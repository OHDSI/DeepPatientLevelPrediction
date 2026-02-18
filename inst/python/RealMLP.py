import math
import torch
from torch import nn
from typing import Optional
from Embeddings import Embedding


class TokenFeatureScale(nn.Module):
    """Learned scalar per covariate id, applied before token aggregation."""

    def __init__(self, vocabulary_size: int, scale_dim: int = 1):
        super().__init__()
        self.scale = nn.Embedding(vocabulary_size + 1, int(scale_dim), padding_idx=0)
        nn.init.ones_(self.scale.weight)
        with torch.no_grad():
            self.scale.weight[0].zero_()
        setattr(self.scale.weight, "_is_scale_param", True)  # tag for optimizer group

    def forward(self, feature_ids):
        return self.scale(feature_ids)


class LinearNTP(nn.Linear):
    """Linear layer with Neural Tangent Parametrization scaling."""

    _ntp_factor: torch.Tensor

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("_ntp_factor", torch.tensor(1.0 / math.sqrt(in_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=1.0)

    def forward(self, input):
        return nn.functional.linear(input, self.weight, self.bias) * self._ntp_factor


class ParametricActivation(nn.Module):
    """σα(x) = (1-α)x + α σ(x), per-neuron α."""

    def __init__(self, hidden: int, base_act: str = "selu"):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden))
        setattr(self.alpha, "_is_act_param", True)  # tag for optimizer group
        self.sigma = nn.SELU() if base_act == "selu" else nn.Mish()

    def forward(self, x):
        a = self.alpha.view(1, -1)
        return (1 - a) * x + a * self.sigma(x)


class ScheduledDropout(nn.Module):
    """Dropout with externally settable p."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def set_p(self, p):
        self.p = float(p)

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training)


class RealMLP(nn.Module):
    """
    Minimal RealMLP for binary classification.
    Assumes preprocessing (robust scaling/clipping) is handled upstream.
    """

    def __init__(
        self,
        feature_info,
        size_embedding: int,
        size_hidden: int,
        num_layers: int,
        dropout: float,
        dim_out: int = 1,
        model_type: str = "RealMLP",
        paper_mode: bool = False,
        token_aggregation: str = "mean",
        feature_scale_mode: str = "scalar",
    ):
        super().__init__()
        self.name = model_type
        self.paper_mode = bool(paper_mode)
        self.token_aggregation = str(token_aggregation)
        self.feature_scale_mode = str(feature_scale_mode)

        if self.paper_mode and self.token_aggregation == "mean":
            self.token_aggregation = "sum"

        if self.token_aggregation not in ("mean", "sum", "sum_len_norm"):
            raise ValueError(
                f"Unknown token_aggregation: {self.token_aggregation}. "
                "Expected one of: mean, sum, sum_len_norm."
            )
        if self.feature_scale_mode not in ("scalar", "vector"):
            raise ValueError(
                f"Unknown feature_scale_mode: {self.feature_scale_mode}. "
                "Expected one of: scalar, vector."
            )

        # Use existing Embedding so Dataset interface stays unchanged.
        # numeric_mode="scale" preserves your current MLP behavior.
        self.embedding = Embedding(
            feature_info=feature_info,
            numeric_mode="scale",
            embedding_dim=int(size_embedding),
            aggregate="none",
        )
        for parameter in self.embedding.parameters():
            setattr(parameter, "_is_embedding_param", True)

        in_dim = int(size_embedding)
        scale_dim = 1 if self.feature_scale_mode == "scalar" else in_dim
        self.feature_scale = TokenFeatureScale(
            vocabulary_size=int(feature_info.get_vocabulary_size()),
            scale_dim=scale_dim,
        )

        blocks = []
        last = in_dim
        for _ in range(int(num_layers)):
            blocks += [
                LinearNTP(last, int(size_hidden), bias=True),
                nn.BatchNorm1d(int(size_hidden)),
                ParametricActivation(int(size_hidden), base_act="selu"),
                ScheduledDropout(p=float(dropout)),
            ]
            last = int(size_hidden)
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(last, int(dim_out))

        # cache for schedule updates
        self._sched_drops = [m for m in self.blocks if isinstance(m, ScheduledDropout)]
        self._linear_layers = [m for m in self.blocks if isinstance(m, LinearNTP)]

    def set_dropout(self, p: float):
        for d in self._sched_drops:
            d.set_p(p)

    @staticmethod
    def _resolve_bias_target(
        in_features: int,
        bias_mode: Optional[str],
        bias_scale: float,
    ):
        mode = "none" if bias_mode is None else str(bias_mode).lower()
        if mode in ("none", "off"):
            return None
        if mode in ("zero", "center"):
            return 0.0
        if mode in ("he5", "he+5"):
            return float(bias_scale) * (5.0 / math.sqrt(float(in_features)))
        raise ValueError(f"Unknown data-dependent init bias mode: {bias_mode}")

    def _collect_layer_outputs(self, layer, batches):
        layer_outputs = []

        def _capture_output(_, __, output):
            layer_outputs.append(output.detach())

        hook = layer.register_forward_hook(_capture_output)
        try:
            for features in batches:
                self(features)
        finally:
            hook.remove()
        return layer_outputs

    def data_dependent_init(
        self,
        batches,
        eps: float = 1e-6,
        bias_mode: Optional[str] = "he5",
        bias_scale: float = 1.0,
    ):
        """
        Sequentially rescale each hidden LinearNTP row so pre-activation
        variance is ~1 on a small sample of training batches and optionally
        fit layer biases to a target mean ("he5" or "zero").
        """
        if not self._linear_layers:
            return
        batches = list(batches)
        if len(batches) == 0:
            return

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for layer in self._linear_layers:
                    layer_outputs = self._collect_layer_outputs(layer, batches)

                    if len(layer_outputs) == 0:
                        continue
                    merged = torch.cat(layer_outputs, dim=0)
                    variances = merged.var(dim=0, unbiased=False).clamp_min(eps)
                    gains = torch.rsqrt(variances)
                    layer.weight.mul_(gains.view(-1, 1))
                    if layer.bias is not None:
                        target_value = self._resolve_bias_target(
                            in_features=layer.in_features,
                            bias_mode=bias_mode,
                            bias_scale=bias_scale,
                        )
                        if target_value is None:
                            continue
                        centered_outputs = self._collect_layer_outputs(layer, batches)
                        if len(centered_outputs) == 0:
                            continue
                        centered = torch.cat(centered_outputs, dim=0)
                        means = centered.mean(dim=0)
                        target = torch.full_like(means, float(target_value))
                        output_scale = 1.0
                        if hasattr(layer, "_ntp_factor"):
                            output_scale = float(layer._ntp_factor.item())
                            output_scale = max(output_scale, eps)
                        layer.bias.add_((target - means) / output_scale)
        finally:
            if was_training:
                self.train()

    def _aggregate_tokens(self, scaled_tokens, feature_ids):
        if self.token_aggregation == "mean":
            return scaled_tokens.mean(dim=1)
        if self.token_aggregation == "sum":
            return scaled_tokens.sum(dim=1)
        # sum_len_norm
        lengths = (feature_ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
        lengths = lengths.to(dtype=scaled_tokens.dtype)
        return scaled_tokens.sum(dim=1) / lengths

    def forward(self, x):
        feature_ids = x["feature_ids"]
        tokens = self.embedding(x)  # [B, L, size_embedding]
        scales = self.feature_scale(feature_ids)  # [B, L, 1] or [B, L, size_embedding]
        x = self._aggregate_tokens(tokens * scales, feature_ids)  # [B, size_embedding]
        for m in self.blocks:
            x = m(x)
        x = self.head(x).squeeze(-1)
        return x
