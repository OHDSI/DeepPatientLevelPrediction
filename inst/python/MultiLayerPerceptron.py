import torch
from torch import nn

from Dataset import FeatureInfo
from Embeddings import Embedding


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        feature_info: FeatureInfo,
        size_embedding: int,
        size_hidden: int,
        num_layers: int,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
        dropout=0.0,
        dim_out: int = 1,
        concat_num=False,
        token_aggregation: str = "mean",
        numerical_bias: bool = True,
        numerical_bias_scale: float = 1.0,
        numerical_bias_normalization: str = "none",
        model_type="MultiLayerPerceptron"
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.name = model_type
        size_embedding = int(size_embedding)
        size_hidden = int(size_hidden)
        num_layers = int(num_layers)
        dim_out = int(dim_out)
        self.token_aggregation = str(token_aggregation)
        if self.token_aggregation not in ("mean", "sum", "sum_sqrt_len", "sum_len_norm"):
            raise ValueError(
                "token_aggregation must be one of: mean, sum, sum_sqrt_len, sum_len_norm"
            )


        self.embedding = Embedding(
            feature_info=feature_info,
            numeric_mode="concatenate" if concat_num else "scale",
            embedding_dim=size_embedding,
            numerical_bias=bool(numerical_bias),
            numerical_bias_scale=float(numerical_bias_scale),
            numerical_bias_normalization=str(numerical_bias_normalization),
            aggregate="none",
        )

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        self.layers = nn.ModuleList(
            MlpLayer(
                size_hidden=size_hidden,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
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
        if tokens.dim() <= 2:
            return tokens
        token_sum = tokens.sum(dim=1)
        if self.token_aggregation == "sum":
            return token_sum
        lengths = (feature_ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
        lengths = lengths.to(dtype=tokens.dtype)
        if self.token_aggregation in ("sum_sqrt_len", "sum_len_norm"):
            return token_sum / torch.sqrt(lengths)
        return token_sum / lengths

    def set_dropout(self, p: float):
        for layer in self.layers:
            if layer.dropout is not None:
                layer.dropout.p = float(p)

    def collect_batch_diagnostics(self, batch):
        return self.embedding.get_aggregation_diagnostics(
            batch,
            mode=self.token_aggregation,
        )

    def get_optimizer_param_groups(self, estimator_settings):
        embedding_lr_mult = float(estimator_settings.get("embedding_lr_mult", 1.0))
        bias_lr_mult = float(estimator_settings.get("bias_lr_mult", 1.0))
        norm_lr_mult = float(estimator_settings.get("norm_lr_mult", 1.0))

        exclude_embedding_from_wd = bool(estimator_settings.get("exclude_embedding_from_wd", False))
        exclude_bias_from_wd = bool(estimator_settings.get("exclude_bias_from_wd", False))
        exclude_norm_from_wd = bool(estimator_settings.get("exclude_norm_from_wd", False))

        embedding_wd_factor = float(
            estimator_settings.get(
                "embedding_wd_factor",
                0.0 if exclude_embedding_from_wd else 1.0,
            )
        )
        bias_wd_factor = float(
            estimator_settings.get(
                "bias_wd_factor",
                0.0 if exclude_bias_from_wd else 1.0,
            )
        )
        norm_wd_factor = float(
            estimator_settings.get(
                "norm_wd_factor",
                0.0 if exclude_norm_from_wd else 1.0,
            )
        )

        embedding_param_ids = {id(p) for p in self.embedding.parameters() if p.requires_grad}
        norm_param_ids = set()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        norm_param_ids.add(id(p))

        embedding_params = []
        norm_params = []
        bias_params = []
        other_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in embedding_param_ids:
                embedding_params.append(parameter)
            elif parameter_id in norm_param_ids:
                norm_params.append(parameter)
            elif name.endswith(".bias"):
                bias_params.append(parameter)
            else:
                other_params.append(parameter)

        groups = []
        if other_params:
            groups.append(
                {"params": other_params, "name": "other", "lr_factor": 1.0, "wd_factor": 1.0}
            )
        if embedding_params:
            groups.append(
                {
                    "params": embedding_params,
                    "name": "embed",
                    "lr_factor": embedding_lr_mult,
                    "wd_factor": embedding_wd_factor,
                }
            )
        if norm_params:
            groups.append(
                {
                    "params": norm_params,
                    "name": "norm",
                    "lr_factor": norm_lr_mult,
                    "wd_factor": norm_wd_factor,
                }
            )
        if bias_params:
            groups.append(
                {
                    "params": bias_params,
                    "name": "bias",
                    "lr_factor": bias_lr_mult,
                    "wd_factor": bias_wd_factor,
                }
            )
        return groups


class MlpLayer(nn.Module):
    def __init__(
        self,
        size_hidden=64,
        normalization=nn.BatchNorm1d,
        activation=nn.ReLU,
        dropout=0.0,
        bias=True,
    ):
        super(MlpLayer, self).__init__()
        self.norm = normalization(size_hidden)
        self.activation = activation()
        self.linear = nn.Linear(size_hidden, size_hidden, bias=bias)

        self.dropout = nn.Dropout(p=dropout) if dropout is not None and dropout > 0.0 else None

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(x)
