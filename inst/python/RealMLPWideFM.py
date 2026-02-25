import csv
import math
import pathlib

import torch
from torch import nn
from torch.nn import functional as F

from RealMLP import LinearNTP, ParametricActivation, RealMLP, ScheduledDropout


class GLUNumericResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        input_dim = int(max(1, input_dim))
        hidden_dim = int(max(1, hidden_dim))
        num_layers = int(max(1, num_layers))
        dropout = float(max(0.0, dropout))

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            if self.out.bias is not None:
                self.out.bias.zero_()

    def forward(self, x):
        h = x
        i = 0
        while i < len(self.layers):
            value_proj = self.layers[i]
            gate_proj = self.layers[i + 1]
            norm = self.layers[i + 2]
            act = self.layers[i + 3]
            h = value_proj(h) * torch.sigmoid(gate_proj(h))
            h = act(norm(h))
            i += 4
            if i < len(self.layers) and isinstance(self.layers[i], nn.Dropout):
                h = self.layers[i](h)
                i += 1
        return self.out(h).squeeze(-1)


class DenseNumericRealMLPBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mode: str = "scale",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_frequencies: int = 8,
        periodic_init_std: float = 0.1,
        pbld_hidden_dim: int = 16,
        pbld_embedding_dim: int = 4,
        use_missing_indicators: bool = False,
        threshold_numeric_indices=None,
        threshold_values=None,
    ):
        super().__init__()
        self.input_dim = int(max(1, input_dim))
        self.mode = str(mode)
        self.use_missing_indicators = bool(use_missing_indicators)

        if self.mode not in ("scale", "pl", "pbld"):
            raise ValueError("dense_realmlp_mode must be one of: scale, pl, pbld")

        self.feature_scale = nn.Parameter(torch.ones(self.input_dim))
        setattr(self.feature_scale, "_is_scale_param", True)

        self.num_frequencies = int(max(1, num_frequencies))
        self.periodic_init_std = float(max(0.0, periodic_init_std))
        self.pbld_hidden_dim = int(max(1, pbld_hidden_dim))
        self.pbld_embedding_dim = int(max(1, pbld_embedding_dim))

        self.freq = None
        self.phase = None
        self.pbld_w1 = None
        self.pbld_b1 = None
        self.pbld_w2 = None
        self.pbld_b2 = None

        self.register_buffer("threshold_numeric_indices", None, persistent=False)
        self.register_buffer("threshold_values", None, persistent=False)
        self.threshold_feature_count = 0
        if threshold_numeric_indices is not None and threshold_values is not None:
            idx = torch.as_tensor(threshold_numeric_indices, dtype=torch.long)
            vals = torch.as_tensor(threshold_values, dtype=torch.float32)
            if idx.numel() > 0 and vals.numel() == idx.numel():
                self.register_buffer("threshold_numeric_indices", idx, persistent=False)
                self.register_buffer("threshold_values", vals, persistent=False)
                self.threshold_feature_count = int(idx.numel())

        encoded_dim = self.input_dim
        if self.mode in ("pl", "pbld"):
            self.freq = nn.Parameter(
                torch.empty(self.input_dim, self.num_frequencies).normal_(
                    mean=0.0,
                    std=self.periodic_init_std,
                )
            )
            self.phase = nn.Parameter(torch.zeros(self.input_dim, self.num_frequencies))
            if self.mode == "pl":
                encoded_dim = self.input_dim * (1 + self.num_frequencies)
            else:
                self.pbld_w1 = nn.Parameter(
                    torch.empty(self.input_dim, self.num_frequencies, self.pbld_hidden_dim)
                )
                self.pbld_b1 = nn.Parameter(torch.zeros(self.input_dim, self.pbld_hidden_dim))
                self.pbld_w2 = nn.Parameter(
                    torch.empty(self.input_dim, self.pbld_hidden_dim, self.pbld_embedding_dim)
                )
                self.pbld_b2 = nn.Parameter(torch.zeros(self.input_dim, self.pbld_embedding_dim))
                nn.init.normal_(self.pbld_w1, mean=0.0, std=0.1)
                nn.init.normal_(self.pbld_w2, mean=0.0, std=0.1)
                encoded_dim = self.input_dim * (1 + self.pbld_embedding_dim)

        encoded_dim += self.threshold_feature_count
        if self.use_missing_indicators:
            encoded_dim += self.input_dim

        hidden_dim = int(max(1, hidden_dim))
        num_layers = int(max(1, num_layers))
        dropout = float(max(0.0, dropout))

        blocks = []
        in_dim = encoded_dim
        for _ in range(num_layers):
            blocks.append(LinearNTP(in_dim, hidden_dim, bias=True))
            blocks.append(nn.BatchNorm1d(hidden_dim))
            blocks.append(ParametricActivation(hidden_dim, base_act="selu"))
            blocks.append(ScheduledDropout(dropout))
            in_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            if self.head.bias is not None:
                self.head.bias.zero_()

    def set_dropout(self, p: float):
        for module in self.blocks:
            if isinstance(module, ScheduledDropout):
                module.set_p(p)

    def _encode(self, values):
        x = values * self.feature_scale.view(1, -1)
        threshold_feats = None
        if self.threshold_feature_count > 0:
            idx = self.threshold_numeric_indices.to(device=x.device)
            thr = self.threshold_values.to(device=x.device)
            threshold_feats = (x.index_select(dim=1, index=idx) > thr.view(1, -1)).to(dtype=x.dtype)
        if self.mode == "scale":
            if threshold_feats is None:
                return x
            return torch.cat((x, threshold_feats), dim=1)

        # x: [B, F], freq/phase: [F, K]
        periodic = torch.sin(x.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0))
        if self.mode == "pl":
            encoded = torch.cat((x.unsqueeze(-1), periodic), dim=-1).flatten(1)
            if threshold_feats is None:
                return encoded
            return torch.cat((encoded, threshold_feats), dim=1)

        hidden = torch.einsum("bfk,fkh->bfh", periodic, self.pbld_w1) + self.pbld_b1.unsqueeze(0)
        hidden = F.silu(hidden)
        emb = torch.einsum("bfh,fhe->bfe", hidden, self.pbld_w2) + self.pbld_b2.unsqueeze(0)
        encoded = torch.cat((x.unsqueeze(-1), emb), dim=-1).flatten(1)
        if threshold_feats is None:
            return encoded
        return torch.cat((encoded, threshold_feats), dim=1)

    def forward(self, values, present_mask=None):
        encoded = self._encode(values)
        if self.use_missing_indicators and present_mask is not None:
            missing = (~present_mask).to(dtype=values.dtype)
            encoded = torch.cat((encoded, missing), dim=1)

        hidden = encoded
        for module in self.blocks:
            hidden = module(hidden)
        return self.head(hidden).squeeze(-1)


class RealMLPWideFM(RealMLP):
    """Wide + FM (+ optional deep/numeric residual) variant for sparse tabular tokens."""

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
        numeric_residual_mode: str = "mlp",
        numeric_residual_hidden_dim: int = 64,
        numeric_residual_num_layers: int = 2,
        numeric_residual_dropout: float = 0.0,
        numeric_residual_alpha_init: float = 0.0,
        numeric_residual_alpha_trainable: bool = True,
        numeric_residual_alpha_nonnegative: bool = False,
        numeric_gate_l1_lambda: float = 0.0,
        numeric_gate_l_1_lambda: float = None,
        numeric_use_missing_indicators: bool = False,
        dense_realmlp_enabled: bool = False,
        dense_realmlp_mode: str = "scale",
        dense_realmlp_hidden_dim: int = 64,
        dense_realmlp_num_layers: int = 2,
        dense_realmlp_dropout: float = 0.0,
        dense_realmlp_num_frequencies: int = 8,
        dense_realmlp_periodic_init_std: float = 0.1,
        dense_realmlp_pbld_hidden_dim: int = 16,
        dense_realmlp_pbld_embedding_dim: int = 4,
        dense_realmlp_threshold_path: str = "",
        dense_realmlp_threshold_top_k: int = 0,
        dense_realmlp_threshold_value_field: str = "medianThreshold",
        dense_realmlp_alpha_init: float = 0.0,
        dense_realmlp_alpha_trainable: bool = True,
        dense_realmlp_alpha_nonnegative: bool = False,
        fwfm_enabled: bool = False,
        fwfm_field: str = "analysisId",
        fwfm_alpha_init: float = 0.0,
        fwfm_alpha_trainable: bool = True,
        fwfm_alpha_nonnegative: bool = False,
        orthogonal_logit_lambda: float = 0.0,
        orthogonal_logit_eps: float = 1e-8,
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
        self.numeric_residual_mode = str(numeric_residual_mode)
        self.numeric_residual_alpha_trainable = bool(numeric_residual_alpha_trainable)
        self.numeric_residual_alpha_nonnegative = bool(numeric_residual_alpha_nonnegative)
        if numeric_gate_l_1_lambda is not None:
            numeric_gate_l1_lambda = numeric_gate_l_1_lambda
        self.numeric_gate_l1_lambda = float(max(0.0, numeric_gate_l1_lambda))
        self.numeric_use_missing_indicators = bool(numeric_use_missing_indicators)

        self.dense_realmlp_enabled = bool(dense_realmlp_enabled)
        self.dense_realmlp_alpha_trainable = bool(dense_realmlp_alpha_trainable)
        self.dense_realmlp_alpha_nonnegative = bool(dense_realmlp_alpha_nonnegative)
        self.dense_realmlp_threshold_path = str(dense_realmlp_threshold_path or "").strip()
        self.dense_realmlp_threshold_top_k = int(max(0, dense_realmlp_threshold_top_k))
        self.dense_realmlp_threshold_value_field = str(
            dense_realmlp_threshold_value_field or "medianThreshold"
        )

        self.fwfm_enabled = bool(fwfm_enabled)
        self.fwfm_field = str(fwfm_field)
        self.fwfm_alpha_trainable = bool(fwfm_alpha_trainable)
        self.fwfm_alpha_nonnegative = bool(fwfm_alpha_nonnegative)
        self.orthogonal_logit_lambda = float(max(0.0, orthogonal_logit_lambda))
        self.orthogonal_logit_eps = float(max(1e-12, orthogonal_logit_eps))

        vocabulary_size = int(self.feature_info.get_vocabulary_size())
        self.fm_embedding = nn.Embedding(vocabulary_size + 1, self.fm_rank, padding_idx=0)
        self.fm_alpha = nn.Parameter(
            torch.tensor(float(fm_alpha_init), dtype=torch.float32),
            requires_grad=self.fm_alpha_trainable,
        )
        nn.init.normal_(self.fm_embedding.weight, mean=0.0, std=0.002)
        with torch.no_grad():
            self.fm_embedding.weight[0].zero_()
        setattr(self.fm_embedding.weight, "_is_fm_param", True)
        setattr(self.fm_alpha, "_is_fm_param", True)

        if self.fwfm_enabled:
            self.fwfm_embedding = nn.Embedding(vocabulary_size + 1, self.fm_rank, padding_idx=0)
            nn.init.normal_(self.fwfm_embedding.weight, mean=0.0, std=0.002)
            with torch.no_grad():
                self.fwfm_embedding.weight[0].zero_()
            self.fwfm_alpha = nn.Parameter(
                torch.tensor(float(fwfm_alpha_init), dtype=torch.float32),
                requires_grad=self.fwfm_alpha_trainable,
            )
            field_lookup, n_fields = self._build_field_lookup(
                vocabulary_size=vocabulary_size,
                field_name=self.fwfm_field,
            )
            self.register_buffer("fwfm_field_lookup", field_lookup)
            self.fwfm_num_fields = int(n_fields)
            self.fwfm_field_weight = nn.Parameter(
                torch.zeros(self.fwfm_num_fields, self.fwfm_num_fields, dtype=torch.float32),
                requires_grad=True,
            )
            nn.init.normal_(self.fwfm_field_weight, mean=0.0, std=0.01)
            fwfm_mask = torch.triu(
                torch.ones(self.fwfm_num_fields, self.fwfm_num_fields, dtype=torch.float32),
                diagonal=1,
            )
            self.register_buffer("fwfm_pair_mask", fwfm_mask)
            setattr(self.fwfm_embedding.weight, "_is_fwfm_param", True)
            setattr(self.fwfm_field_weight, "_is_fwfm_param", True)
            setattr(self.fwfm_alpha, "_is_fwfm_param", True)
        else:
            self.fwfm_embedding = None
            self.fwfm_alpha = None
            self.fwfm_num_fields = 0

        self.numerical_feature_count = int(getattr(self.embedding, "n_num", 0))
        threshold_numeric_indices, threshold_values = self._load_dense_realmlp_threshold_basis(
            path=self.dense_realmlp_threshold_path,
            top_k=self.dense_realmlp_threshold_top_k,
            value_field=self.dense_realmlp_threshold_value_field,
        )
        self.numeric_feature_gate = None
        if self.numerical_feature_count > 0:
            # Shared feature gate for numeric residual branches.
            self.numeric_feature_gate = nn.Parameter(torch.zeros(self.numerical_feature_count))
            setattr(self.numeric_feature_gate, "_is_num_residual_param", True)

        self.numeric_residual = None
        self.numeric_residual_alpha = None
        if self.numeric_residual_enabled and self.numerical_feature_count > 0:
            hidden_dim = int(max(1, numeric_residual_hidden_dim))
            num_layers = int(max(1, numeric_residual_num_layers))
            dropout = float(max(0.0, numeric_residual_dropout))
            numeric_input_dim = self.numerical_feature_count
            if self.numeric_use_missing_indicators:
                numeric_input_dim += self.numerical_feature_count

            if self.numeric_residual_mode == "glu":
                self.numeric_residual = GLUNumericResidual(
                    input_dim=numeric_input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            else:
                layers = []
                in_dim = numeric_input_dim
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

        self.dense_realmlp_branch = None
        self.dense_realmlp_alpha = None
        if self.dense_realmlp_enabled and self.numerical_feature_count > 0:
            self.dense_realmlp_branch = DenseNumericRealMLPBranch(
                input_dim=self.numerical_feature_count,
                mode=str(dense_realmlp_mode),
                hidden_dim=int(max(1, dense_realmlp_hidden_dim)),
                num_layers=int(max(1, dense_realmlp_num_layers)),
                dropout=float(max(0.0, dense_realmlp_dropout)),
                num_frequencies=int(max(1, dense_realmlp_num_frequencies)),
                periodic_init_std=float(max(0.0, dense_realmlp_periodic_init_std)),
                pbld_hidden_dim=int(max(1, dense_realmlp_pbld_hidden_dim)),
                pbld_embedding_dim=int(max(1, dense_realmlp_pbld_embedding_dim)),
                use_missing_indicators=bool(self.numeric_use_missing_indicators),
                threshold_numeric_indices=threshold_numeric_indices,
                threshold_values=threshold_values,
            )
            self.dense_realmlp_alpha = nn.Parameter(
                torch.tensor(float(dense_realmlp_alpha_init), dtype=torch.float32),
                requires_grad=self.dense_realmlp_alpha_trainable,
            )
            for parameter in self.dense_realmlp_branch.parameters():
                setattr(parameter, "_is_num_residual_param", True)
            setattr(self.dense_realmlp_alpha, "_is_num_residual_param", True)

        if model_init_path is not None and str(model_init_path).strip() != "":
            self.load_model_from_file(path=str(model_init_path), strict=model_init_strict)

        self.set_branch_trainability(
            freeze_wide=self.freeze_wide,
            freeze_deep=self.freeze_deep,
        )

    def set_dropout(self, p: float):
        super().set_dropout(p)
        if self.dense_realmlp_branch is not None:
            self.dense_realmlp_branch.set_dropout(p)

    def set_branch_trainability(self, freeze_wide=None, freeze_deep=None):
        super().set_branch_trainability(freeze_wide=freeze_wide, freeze_deep=freeze_deep)
        if not hasattr(self, "fm_embedding") or not hasattr(self, "fm_alpha"):
            return
        if self.freeze_deep:
            self.fm_embedding.weight.requires_grad_(False)
            self.fm_alpha.requires_grad_(False)
            if self.fwfm_embedding is not None:
                self.fwfm_embedding.weight.requires_grad_(False)
            if self.fwfm_alpha is not None:
                self.fwfm_alpha.requires_grad_(False)
            if hasattr(self, "fwfm_field_weight"):
                self.fwfm_field_weight.requires_grad_(False)
            if self.numeric_residual is not None:
                for parameter in self.numeric_residual.parameters():
                    parameter.requires_grad_(False)
            if self.numeric_residual_alpha is not None:
                self.numeric_residual_alpha.requires_grad_(False)
            if self.numeric_feature_gate is not None:
                self.numeric_feature_gate.requires_grad_(False)
            if self.dense_realmlp_branch is not None:
                for parameter in self.dense_realmlp_branch.parameters():
                    parameter.requires_grad_(False)
            if self.dense_realmlp_alpha is not None:
                self.dense_realmlp_alpha.requires_grad_(False)
        else:
            self.fm_embedding.weight.requires_grad_(True)
            self.fm_alpha.requires_grad_(self.fm_alpha_trainable)
            if self.fwfm_embedding is not None:
                self.fwfm_embedding.weight.requires_grad_(True)
            if self.fwfm_alpha is not None:
                self.fwfm_alpha.requires_grad_(self.fwfm_alpha_trainable)
            if hasattr(self, "fwfm_field_weight"):
                self.fwfm_field_weight.requires_grad_(True)
            if self.numeric_residual is not None:
                for parameter in self.numeric_residual.parameters():
                    parameter.requires_grad_(True)
            if self.numeric_residual_alpha is not None:
                self.numeric_residual_alpha.requires_grad_(self.numeric_residual_alpha_trainable)
            if self.numeric_feature_gate is not None:
                self.numeric_feature_gate.requires_grad_(True)
            if self.dense_realmlp_branch is not None:
                for parameter in self.dense_realmlp_branch.parameters():
                    parameter.requires_grad_(True)
            if self.dense_realmlp_alpha is not None:
                self.dense_realmlp_alpha.requires_grad_(self.dense_realmlp_alpha_trainable)

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

    def _build_field_lookup(self, vocabulary_size: int, field_name: str):
        lookup = torch.zeros(vocabulary_size + 1, dtype=torch.long)
        data_ref = getattr(self.feature_info, "data_reference", None)
        if data_ref is None:
            return lookup, 1
        if field_name not in data_ref.columns or "columnId" not in data_ref.columns:
            return lookup, 1

        try:
            column_ids = data_ref.select("columnId").to_series().to_list()
            field_values = data_ref.select(field_name).to_series().to_list()
        except Exception:
            return lookup, 1

        compact_map = {}
        next_field = 1
        for feature_id, raw_value in zip(column_ids, field_values):
            try:
                feature_id = int(feature_id)
            except Exception:
                continue
            if feature_id <= 0 or feature_id > vocabulary_size:
                continue
            try:
                key = int(raw_value)
            except Exception:
                key = str(raw_value)
            if key not in compact_map:
                compact_map[key] = next_field
                next_field += 1
            lookup[feature_id] = compact_map[key]

        n_fields = max(1, next_field - 1)
        return lookup, n_fields

    def _load_dense_realmlp_threshold_basis(self, path: str, top_k: int, value_field: str):
        if self.numerical_feature_count <= 0:
            return [], []
        if path is None or str(path).strip() == "":
            return [], []
        if top_k <= 0:
            return [], []

        file_path = pathlib.Path(str(path))
        if not file_path.exists():
            print(f"Dense threshold basis file not found: {file_path}; disabling threshold basis.")
            return [], []

        input_to_numeric = getattr(self.embedding, "input_to_numeric", None)
        if input_to_numeric is None:
            return [], []
        if torch.is_tensor(input_to_numeric):
            input_to_numeric_cpu = input_to_numeric.detach().cpu()
        else:
            input_to_numeric_cpu = torch.as_tensor(input_to_numeric)

        selected = []
        seen_numeric = set()
        key_value = str(value_field).lower()
        with open(file_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            field_map = {str(x).lower(): x for x in (reader.fieldnames or [])}
            split_key = field_map.get("splitindex")
            count_key = field_map.get("splitcount")
            value_key = field_map.get(key_value)
            if split_key is None or value_key is None:
                print(
                    f"Dense threshold basis file missing required columns splitIndex/{value_field}; "
                    "disabling threshold basis."
                )
                return [], []

            for row in reader:
                try:
                    split_index = int(float(row[split_key]))
                    threshold_value = float(row[value_key])
                except Exception:
                    continue
                split_count = 0.0
                if count_key is not None:
                    try:
                        split_count = float(row[count_key])
                    except Exception:
                        split_count = 0.0
                feature_id = split_index + 1
                if feature_id <= 0 or feature_id >= int(input_to_numeric_cpu.numel()):
                    continue
                numeric_slot = int(input_to_numeric_cpu[feature_id].item()) - 1
                if numeric_slot < 0 or numeric_slot >= self.numerical_feature_count:
                    continue
                selected.append((split_count, numeric_slot, threshold_value))

        if len(selected) == 0:
            return [], []

        selected.sort(key=lambda x: x[0], reverse=True)
        numeric_indices = []
        threshold_values = []
        for _, numeric_slot, threshold_value in selected:
            if numeric_slot in seen_numeric:
                continue
            seen_numeric.add(numeric_slot)
            numeric_indices.append(int(numeric_slot))
            threshold_values.append(float(threshold_value))
            if len(numeric_indices) >= int(top_k):
                break

        if len(numeric_indices) > 0:
            print(
                "Loaded dense threshold basis | "
                f"file={file_path} | selected={len(numeric_indices)}"
            )
        return numeric_indices, threshold_values

    def _fwfm_logit(self, feature_ids, feature_values):
        if self.fwfm_embedding is None or self.fwfm_num_fields <= 0:
            return torch.zeros(feature_ids.shape[0], device=feature_ids.device, dtype=self.fm_embedding.weight.dtype)

        if feature_values is None:
            feature_values = torch.ones_like(feature_ids, dtype=self.fwfm_embedding.weight.dtype)
        else:
            feature_values = feature_values.to(dtype=self.fwfm_embedding.weight.dtype)

        weighted_tokens = self.fwfm_embedding(feature_ids) * feature_values.unsqueeze(-1)
        field_ids = self.fwfm_field_lookup[feature_ids]
        valid = feature_ids.ne(0)

        pooled = weighted_tokens.new_zeros((feature_ids.shape[0], self.fwfm_num_fields + 1, self.fm_rank))
        scatter_index = field_ids.unsqueeze(-1).expand_as(weighted_tokens)
        pooled.scatter_add_(1, scatter_index, weighted_tokens * valid.unsqueeze(-1))
        pooled_fields = pooled[:, 1:, :]

        pair_dot = torch.bmm(pooled_fields, pooled_fields.transpose(1, 2))
        fwfm_logit = (pair_dot * self.fwfm_field_weight * self.fwfm_pair_mask).sum(dim=(1, 2))
        return fwfm_logit

    def _numeric_dense_values(self, feature_ids, feature_values):
        if feature_values is None or self.numerical_feature_count <= 0:
            return None, None

        numeric_lookup = self.embedding.input_to_numeric[feature_ids]
        numeric_mask = numeric_lookup != 0
        dense = feature_values.new_zeros((feature_ids.shape[0], self.numerical_feature_count))
        present_counts = feature_values.new_zeros((feature_ids.shape[0], self.numerical_feature_count))

        if not torch.any(numeric_mask):
            return dense, present_counts.bool()

        scatter_index = (numeric_lookup - 1).clamp_min(0)
        values = torch.where(
            numeric_mask,
            feature_values.to(dtype=dense.dtype),
            torch.zeros_like(feature_values, dtype=dense.dtype),
        )
        dense.scatter_add_(1, scatter_index, values)

        ones = torch.where(
            numeric_mask,
            torch.ones_like(feature_values, dtype=dense.dtype),
            torch.zeros_like(feature_values, dtype=dense.dtype),
        )
        present_counts.scatter_add_(1, scatter_index, ones)
        present_mask = present_counts > 0
        return dense, present_mask

    def _apply_numeric_feature_gate(self, numeric_dense):
        if numeric_dense is None or self.numeric_feature_gate is None:
            return numeric_dense
        gate = torch.sigmoid(self.numeric_feature_gate).view(1, -1)
        return numeric_dense * gate

    def regularization_loss(self):
        reg = super().regularization_loss()
        if self.numeric_gate_l1_lambda > 0.0 and self.numeric_feature_gate is not None:
            gate = torch.sigmoid(self.numeric_feature_gate)
            gate_l1 = self.numeric_gate_l1_lambda * gate.abs().mean()
            reg = gate_l1 if reg is None else (reg + gate_l1)
        if (
            self.orthogonal_logit_lambda > 0.0
            and hasattr(self, "_last_wide_logit")
            and hasattr(self, "_last_deep_total_logit")
        ):
            wide = self._last_wide_logit
            deep_total = self._last_deep_total_logit
            if wide is not None and deep_total is not None:
                wide_centered = wide - wide.mean()
                deep_centered = deep_total - deep_total.mean()
                denom = (
                    torch.sqrt((wide_centered * wide_centered).sum())
                    * torch.sqrt((deep_centered * deep_centered).sum())
                ).clamp_min(self.orthogonal_logit_eps)
                cosine_sq = torch.square((wide_centered * deep_centered).sum() / denom)
                ortho_penalty = self.orthogonal_logit_lambda * cosine_sq
                reg = ortho_penalty if reg is None else (reg + ortho_penalty)
        return reg

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

        fwfm_logit = self._fwfm_logit(feature_ids=feature_ids, feature_values=feature_values)
        fwfm_alpha = self._constrain_alpha(self.fwfm_alpha, self.fwfm_alpha_nonnegative)
        if fwfm_alpha is None:
            fwfm_alpha = torch.zeros_like(fm_alpha)

        deep_component = torch.zeros_like(wide_logit)
        wide_alpha = self._constrain_alpha(self.wide_alpha, self.wide_alpha_nonnegative)
        if wide_alpha is None:
            wide_alpha = torch.zeros((), device=wide_logit.device, dtype=wide_logit.dtype)
        if self.fm_use_deep:
            deep_component = wide_alpha * self._deep_score(x)

        numeric_dense, numeric_present = self._numeric_dense_values(
            feature_ids=feature_ids,
            feature_values=feature_values,
        )
        numeric_dense = self._apply_numeric_feature_gate(numeric_dense)

        numeric_component = torch.zeros_like(wide_logit)
        numeric_alpha = torch.zeros_like(fm_alpha)
        if self.numeric_residual is not None and numeric_dense is not None:
            numeric_input = numeric_dense
            if self.numeric_use_missing_indicators and numeric_present is not None:
                missing = (~numeric_present).to(dtype=numeric_dense.dtype)
                numeric_input = torch.cat((numeric_dense, missing), dim=1)
            numeric_component = self.numeric_residual(numeric_input)
            if numeric_component.ndim > 1:
                numeric_component = numeric_component.squeeze(-1)
            numeric_alpha = self._constrain_alpha(
                self.numeric_residual_alpha,
                self.numeric_residual_alpha_nonnegative,
            )

        dense_realmlp_component = torch.zeros_like(wide_logit)
        dense_realmlp_alpha = torch.zeros_like(fm_alpha)
        if self.dense_realmlp_branch is not None and numeric_dense is not None:
            dense_realmlp_component = self.dense_realmlp_branch(
                numeric_dense,
                present_mask=numeric_present,
            )
            dense_realmlp_alpha = self._constrain_alpha(
                self.dense_realmlp_alpha,
                self.dense_realmlp_alpha_nonnegative,
            )

        self._last_wide_alpha = float(wide_alpha.detach().item())
        self._last_fm_alpha = float(fm_alpha.detach().item())
        self._last_fwfm_alpha = float(fwfm_alpha.detach().item())
        self._last_numeric_alpha = float(numeric_alpha.detach().item())
        self._last_dense_realmlp_alpha = float(dense_realmlp_alpha.detach().item())
        deep_total_component = (
            fm_alpha * fm_logit
            + fwfm_alpha * fwfm_logit
            + deep_component
            + numeric_alpha * numeric_component
            + dense_realmlp_alpha * dense_realmlp_component
        )
        self._last_deep_total_logit = deep_total_component

        combined = (
            wide_logit
            + deep_total_component
        )
        if self.use_two_logit_ce:
            return torch.stack((-0.5 * combined, 0.5 * combined), dim=1)
        return combined
