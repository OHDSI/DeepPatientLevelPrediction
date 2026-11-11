# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from Dataset import FeatureInfo
from Embeddings import Embedding

class ResNet(nn.Module):
    def __init__(
        self,
        feature_info: FeatureInfo,
        size_embedding: int = 256,
        size_hidden: int = 256,
        num_layers: int = 2,
        hidden_factor: int = 1,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
        hidden_dropout=0,
        residual_dropout=0,
        dim_out: int = 1,
        model_type="ResNet",
    ):
        super(ResNet, self).__init__()
        self.name = model_type
        size_embedding = int(size_embedding)
        size_hidden = int(size_hidden)
        num_layers = int(num_layers)
        hidden_factor = int(hidden_factor)
        dim_out = int(dim_out)

        self.embedding = Embedding(
            feature_info=feature_info,
            embedding_dim=size_embedding,
            aggregate="none",
        )

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        res_hidden = size_hidden * hidden_factor

        self.layers = nn.ModuleList(
            ResLayer(
                size_hidden,
                res_hidden,
                normalization,
                activation,
                hidden_dropout,
                residual_dropout,
            )
            for _ in range(num_layers)
        )

        self.last_norm = normalization(size_hidden)

        self.head = nn.Linear(size_hidden, dim_out)
        self.size_hidden = size_hidden
        self.dim_out = dim_out

        self.last_act = activation()

    def forward(self, x):
        feature_ids = x["feature_ids"]
        x = self.embedding(x)
        x = self._aggregate_tokens(x, feature_ids)
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def reset_head(self):
        self.head = nn.Linear(self.size_hidden, self.dim_out)

    def _aggregate_tokens(self, tokens, feature_ids):
        """The mean of the tokens of each patient, leaving out the padding."""
        if tokens.dim() <= 2:
            # custom embeddings that sum the tokens themselves
            return tokens
        lengths = (feature_ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
        return tokens.sum(dim=1) / lengths.to(dtype=tokens.dtype)

    def set_dropout(self, p: float):
        for layer in self.layers:
            if hasattr(layer, "hidden_dropout") and layer.hidden_dropout is not None:
                layer.hidden_dropout.p = float(p)
            if hasattr(layer, "residual_dropout") and layer.residual_dropout is not None:
                layer.residual_dropout.p = float(p)


class ResLayer(nn.Module):
    def __init__(
        self,
        size_hidden,
        res_hidden,
        normalization,
        activation,
        hidden_dropout=None,
        residual_dropout=None,
    ):
        super(ResLayer, self).__init__()

        self.norm = normalization(size_hidden)
        self.linear0 = nn.Linear(size_hidden, res_hidden)
        self.linear1 = nn.Linear(res_hidden, size_hidden)

        self.hidden_dropout = None if hidden_dropout is None else nn.Dropout(p=hidden_dropout)
        self.residual_dropout = (
            None if residual_dropout is None else nn.Dropout(p=residual_dropout)
        )
        self.activation = activation()

    def forward(self, input):
        z = input
        z = self.norm(z)
        z = self.linear0(z)
        z = self.activation(z)
        if self.hidden_dropout is not None:
            z = self.hidden_dropout(z)
        z = self.linear1(z)
        if self.residual_dropout is not None:
            z = self.residual_dropout(z)
        z = z + input
        return z
