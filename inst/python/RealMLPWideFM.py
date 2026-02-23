import torch
from torch import nn
from torch.nn import functional as F

from RealMLP import RealMLP


class RealMLPWideFM(RealMLP):
    """Wide + FM (+ optional deep residual) variant for sparse tabular tokens."""

    def __init__(
        self,
        *args,
        fm_rank: int = 16,
        fm_use_deep: bool = False,
        fm_alpha_init: float = 0.0,
        fm_alpha_trainable: bool = True,
        fm_norm_mode: str = "sqrt_k",
        fm_alpha_nonnegative: bool = False,
        wide_alpha_nonnegative: bool = False,
        numeric_residual_enabled: bool = False,
        numeric_residual_hidden_dim: int = 64,
        numeric_residual_num_layers: int = 2,
        numeric_residual_dropout: float = 0.0,
        numeric_residual_alpha_init: float = 0.0,
        numeric_residual_alpha_trainable: bool = True,
        numeric_residual_alpha_nonnegative: bool = False,
        **kwargs,
    ):
        model_init_path = kwargs.pop("model_init_path", None)
        model_init_strict = bool(kwargs.pop("model_init_strict", True))
        kwargs.setdefault("model_type", "RealMLPWideFM")
        kwargs.setdefault("wide_enabled", True)
        super().__init__(*args, **kwargs)
        self.fm_use_deep = bool(fm_use_deep)
        self.fm_rank = int(fm_rank)
        self.fm_alpha_trainable = bool(fm_alpha_trainable)
        self.fm_norm_mode = str(fm_norm_mode)
        self.fm_alpha_nonnegative = bool(fm_alpha_nonnegative)
        self.wide_alpha_nonnegative = bool(wide_alpha_nonnegative)
        self.numeric_residual_enabled = bool(numeric_residual_enabled)
        self.numeric_residual_alpha_trainable = bool(numeric_residual_alpha_trainable)
        self.numeric_residual_alpha_nonnegative = bool(numeric_residual_alpha_nonnegative)
        vocabulary_size = int(self.feature_info.get_vocabulary_size())
        self.fm_embedding = nn.Embedding(
            vocabulary_size + 1,
            self.fm_rank,
            padding_idx=0,
        )
        self.fm_alpha = nn.Parameter(
            torch.tensor(float(fm_alpha_init), dtype=torch.float32),
            requires_grad=self.fm_alpha_trainable,
        )
        # Keep FM near-zero at start so warm-started wide path remains dominant.
        nn.init.normal_(self.fm_embedding.weight, mean=0.0, std=0.002)
        with torch.no_grad():
            self.fm_embedding.weight[0].zero_()
        setattr(self.fm_embedding.weight, "_is_fm_param", True)
        setattr(self.fm_alpha, "_is_fm_param", True)
        self.numerical_feature_count = int(
            getattr(self.embedding, "n_num", 0)
        )
        if self.numeric_residual_enabled and self.numerical_feature_count > 0:
            hidden_dim = int(max(1, numeric_residual_hidden_dim))
            num_layers = int(max(1, numeric_residual_num_layers))
            dropout = float(max(0.0, numeric_residual_dropout))
            layers = []
            in_dim = self.numerical_feature_count
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.numeric_residual = nn.Sequential(*layers)
            with torch.no_grad():
                last_linear = self.numeric_residual[-1]
                if isinstance(last_linear, nn.Linear):
                    last_linear.weight.zero_()
                    if last_linear.bias is not None:
                        last_linear.bias.zero_()
            self.numeric_residual_alpha = nn.Parameter(
                torch.tensor(float(numeric_residual_alpha_init), dtype=torch.float32),
                requires_grad=self.numeric_residual_alpha_trainable,
            )
            for parameter in self.numeric_residual.parameters():
                setattr(parameter, "_is_num_residual_param", True)
            setattr(self.numeric_residual_alpha, "_is_num_residual_param", True)
        else:
            self.numeric_residual = None
            self.numeric_residual_alpha = None

        if model_init_path is not None and str(model_init_path).strip() != "":
            self.load_model_from_file(
                path=str(model_init_path),
                strict=model_init_strict,
            )
        self.set_branch_trainability(
            freeze_wide=self.freeze_wide,
            freeze_deep=self.freeze_deep,
        )

    def set_branch_trainability(self, freeze_wide=None, freeze_deep=None):
        super().set_branch_trainability(freeze_wide=freeze_wide, freeze_deep=freeze_deep)
        # RealMLP base init may call this before FM params are created.
        if not hasattr(self, "fm_embedding") or not hasattr(self, "fm_alpha"):
            return
        if self.freeze_deep:
            self.fm_embedding.weight.requires_grad_(False)
            self.fm_alpha.requires_grad_(False)
            if self.numeric_residual is not None:
                for parameter in self.numeric_residual.parameters():
                    parameter.requires_grad_(False)
            if self.numeric_residual_alpha is not None:
                self.numeric_residual_alpha.requires_grad_(False)
        else:
            self.fm_embedding.weight.requires_grad_(True)
            self.fm_alpha.requires_grad_(self.fm_alpha_trainable)
            if self.numeric_residual is not None:
                for parameter in self.numeric_residual.parameters():
                    parameter.requires_grad_(True)
            if self.numeric_residual_alpha is not None:
                self.numeric_residual_alpha.requires_grad_(self.numeric_residual_alpha_trainable)

    @staticmethod
    def _constrain_alpha(alpha, nonnegative: bool):
        if alpha is None:
            return None
        if nonnegative:
            return F.softplus(alpha)
        return alpha

    def _deep_score(self, x):
        feature_ids = x["feature_ids"]
        tokens = self.embedding(x)
        scales = self.feature_scale(feature_ids)
        hidden = self._aggregate_tokens(tokens * scales, feature_ids)
        for module in self.blocks:
            hidden = module(hidden)
        deep_logits = self.head(hidden)
        return self.prediction_scores(deep_logits)

    def _fm_logit(self, feature_ids, feature_values):
        if feature_values is None:
            feature_values = torch.ones_like(feature_ids, dtype=self.fm_embedding.weight.dtype)
        else:
            feature_values = feature_values.to(dtype=self.fm_embedding.weight.dtype)

        weighted_tokens = self.fm_embedding(feature_ids) * feature_values.unsqueeze(-1)
        summed = weighted_tokens.sum(dim=1)
        sum_square = (summed * summed).sum(dim=1)
        square_sum = (weighted_tokens * weighted_tokens).sum(dim=(1, 2))
        fm_logit = 0.5 * (sum_square - square_sum)

        if self.fm_norm_mode == "sqrt_k":
            token_count = feature_ids.ne(0).sum(dim=1).to(dtype=fm_logit.dtype)
            norm = torch.sqrt(token_count.clamp_min(1.0))
            fm_logit = fm_logit / norm
        elif self.fm_norm_mode == "k":
            token_count = feature_ids.ne(0).sum(dim=1).to(dtype=fm_logit.dtype)
            norm = token_count.clamp_min(1.0)
            fm_logit = fm_logit / norm
        elif self.fm_norm_mode != "none":
            raise ValueError(f"Unsupported fm_norm_mode: {self.fm_norm_mode}")
        return fm_logit

    def _numeric_dense_values(self, feature_ids, feature_values):
        if feature_values is None:
            return None
        if self.numerical_feature_count <= 0:
            return None

        numeric_lookup = self.embedding.input_to_numeric[feature_ids]
        numeric_mask = numeric_lookup != 0
        if not torch.any(numeric_mask):
            return feature_values.new_zeros(
                (feature_ids.shape[0], self.numerical_feature_count)
            )

        dense = feature_values.new_zeros(
            (feature_ids.shape[0], self.numerical_feature_count)
        )
        scatter_index = (numeric_lookup - 1).clamp_min(0)
        values = torch.where(
            numeric_mask,
            feature_values.to(dtype=dense.dtype),
            torch.zeros_like(feature_values, dtype=dense.dtype),
        )
        dense.scatter_add_(1, scatter_index, values)
        return dense

    def forward(self, x):
        feature_ids = x["feature_ids"]
        feature_values = x.get("feature_values", None)

        wide_logit = self._wide_logit(
            feature_ids=feature_ids,
            feature_values=feature_values,
            dtype=self.fm_embedding.weight.dtype,
        )
        self._last_wide_logit = wide_logit
        fm_logit = self._fm_logit(feature_ids=feature_ids, feature_values=feature_values)
        self._last_fm_logit = fm_logit
        fm_alpha = self._constrain_alpha(self.fm_alpha, self.fm_alpha_nonnegative)

        deep_component = torch.zeros_like(wide_logit)
        if self.fm_use_deep:
            wide_alpha = self._constrain_alpha(self.wide_alpha, self.wide_alpha_nonnegative)
            if wide_alpha is None:
                wide_alpha = torch.zeros((), device=deep_component.device, dtype=deep_component.dtype)
            deep_component = wide_alpha * self._deep_score(x)
        else:
            wide_alpha = self._constrain_alpha(self.wide_alpha, self.wide_alpha_nonnegative)
            if wide_alpha is None:
                wide_alpha = torch.zeros((), device=deep_component.device, dtype=deep_component.dtype)

        self._last_wide_alpha = float(wide_alpha.detach().item())
        self._last_fm_alpha = float(fm_alpha.detach().item())
        numeric_component = torch.zeros_like(wide_logit)
        numeric_alpha = torch.zeros_like(fm_alpha)
        if self.numeric_residual is not None:
            numeric_dense = self._numeric_dense_values(
                feature_ids=feature_ids,
                feature_values=feature_values,
            )
            if numeric_dense is not None:
                numeric_component = self.numeric_residual(numeric_dense).squeeze(-1)
                numeric_alpha = self._constrain_alpha(
                    self.numeric_residual_alpha,
                    self.numeric_residual_alpha_nonnegative,
                )
        self._last_numeric_alpha = float(numeric_alpha.detach().item())
        combined = wide_logit + fm_alpha * fm_logit + deep_component + numeric_alpha * numeric_component
        if self.use_two_logit_ce:
            return torch.stack((-0.5 * combined, 0.5 * combined), dim=1)
        return combined
