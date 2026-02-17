import math
import torch
from torch import nn
from Embeddings import Embedding


class FeatureScale(nn.Module):
    """Diagonal scaling s_i per feature."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        setattr(self.scale, "_is_scale_param", True)  # tag for optimizer group

    def forward(self, x):
        return x * self.scale


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
    ):
        super().__init__()
        self.name = model_type

        # Use existing Embedding so Dataset interface stays unchanged.
        # numeric_mode="scale" preserves your current MLP behavior.
        self.embedding = Embedding(
            feature_info=feature_info,
            numeric_mode="scale",
            embedding_dim=int(size_embedding),
            aggregate="sum",
        )

        in_dim = int(size_embedding)

        self.scale = FeatureScale(in_dim)

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

    def set_dropout(self, p: float):
        for d in self._sched_drops:
            d.set_p(p)

    def forward(self, x):
        x = self.embedding(x)  # [B, size_embedding]
        x = self.scale(x)  # diagonal feature scaling
        for m in self.blocks:
            x = m(x)
        x = self.head(x).squeeze(-1)
        return x
