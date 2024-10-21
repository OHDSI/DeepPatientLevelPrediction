import math

import torch
from torch import nn
import torch.nn.functional as F

from ResNet import NumericalEmbedding

class LogMap0(nn.Module):
    def forward(self, y):
        curvature=1.0
        norm_y = torch.norm(y, dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(torch.tensor(curvature, dtype=y.dtype, device=y.device))
        scale = torch.arctanh(sqrt_c * norm_y) / (sqrt_c * norm_y)
        scale[torch.isnan(scale)] = 1.0
        return scale * y

def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class Transformer(nn.Module):
    def __init__(
        self,
        feature_info,
        num_blocks: int,
        dim_token: int,
        num_heads: int,
        att_dropout,
        ffn_dropout,
        res_dropout,
        dim_hidden: int,
        dim_out: int = 1,
        head_activation=nn.ReLU,
        activation=ReGLU,
        ffn_norm=nn.LayerNorm,
        head_norm=nn.LayerNorm,
        att_norm=nn.LayerNorm,
        model_type="Transformer"
    ):
        super(Transformer, self).__init__()
        self.name = model_type
        num_blocks = int(num_blocks)
        dim_token = int(dim_token)
        num_heads = int(num_heads)
        dim_hidden = int(dim_hidden)
        dim_out = int(dim_out)
        cat_features = feature_info["categorical_features"]
        num_features = feature_info["numerical_features"]
        cat_feature_size = len(cat_features)
        num_feature_size = len(num_features)

        self.embedding = nn.Embedding(
            cat_feature_size + 1, dim_token, padding_idx=0
        )

        if num_feature_size != 0 and num_feature_size is not None:
            self.numerical_embedding = NumericalEmbedding(num_feature_size, dim_token)
            self.use_numerical = True
        else:
            self.use_numerical = False
        self.class_token = ClassToken(dim_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(num_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": nn.MultiheadAttention(
                        dim_token, num_heads, dropout=att_dropout
                    ),
                    "ffn": FeedForwardBlock(
                        dim_token,
                        dim_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=activation,
                    ),
                    "attention_res_dropout": nn.Dropout(res_dropout),
                    "ffn_res_dropout": nn.Dropout(res_dropout),
                    "ffn_norm": ffn_norm(dim_token),
                }
            )
            if layer_idx != 0:
                layer["attention_norm"] = att_norm(dim_token)
            self.layers.append(layer)

        self.head = Head(
            dim_token,
            bias=True,
            activation=head_activation,
            normalization=head_norm,
            dim_out=dim_out,
        )
        self.dim_token = dim_token
        self.head_activation = head_activation
        self.head_normalization = head_norm
        self.dim_out = dim_out
        self.logmap0 = LogMap0()

    def forward(self, x):
        mask = torch.where(x["cat"] == 0, True, False)
        mask = torch.cat([mask], dim=1) # dim 0 may be batch size, dim 1 should it be
		
        cat = self.embedding(x["cat"])

        if self.use_numerical:
            num = self.numerical_embedding(x["num"])
            x = torch.cat([cat, num], dim=1)
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        [x.shape[0], num.shape[1]], device=mask.device, dtype=mask.dtype
                    ),
                ],
                dim=1,
            )
        else:
            x = cat
        x = self.class_token(x)
        mask = torch.cat(
            [mask, torch.zeros([x.shape[0], 1], device=mask.device, dtype=mask.dtype)],
            dim=1,
        )

        for i, layer in enumerate(self.layers):
            x_residual = self.start_residual(layer, "attention", x)

            if i == len(self.layers) - 1:
                dims = x_residual.shape
                x_residual = layer["attention"](
                    x_residual[:, -1].view([dims[0], 1, dims[2]]).transpose(0, 1),
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    mask,
                )
                x_residual = x_residual[0]
                x = x[:, -1].view([dims[0], 1, dims[2]])
            else:
                x_residual = layer["attention"](
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    mask,
                )[0]
            x = self.end_residual(layer, "attention", x, x_residual.transpose(0, 1))

            x_residual = self.start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self.end_residual(layer, "ffn", x, x_residual)

        x = self.head(x)[:, 0]
        return x

    def reset_head(self):
        self.head = Head(
            self.dim_token,
            bias=True,
            activation=self.head_activation,
            normalization=self.head_normalization,
            dim_out=self.dim_out
        )

    @staticmethod
    def start_residual(layer, stage, x):
        norm = f"{stage}_norm"
        if norm in layer.keys():
            x = layer[stage + "_norm"](x)
        return x

    @staticmethod
    def end_residual(layer, stage, x, x_residual):
        x_residual = layer[f"{stage}_res_dropout"](x_residual)
        return x + x_residual


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim_token,
        dim_hidden,
        bias_first=True,
        bias_second=True,
        dropout=0.0,
        activation=ReGLU,
    ):
        super(FeedForwardBlock, self).__init__()
        self.linear0 = nn.Linear(dim_token, int(dim_hidden * 2), bias=bias_first)
        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(dim_hidden, dim_token, bias=bias_second)

    def forward(self, x):
        x = self.linear0(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x


class Head(nn.Module):
    def __init__(self, dim_in, bias, activation, normalization, dim_out):
        super(Head, self).__init__()
        self.normalization = normalization(dim_in)
        self.activation = activation()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class ClassToken(nn.Module):
    def __init__(self, dim_token):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.empty(dim_token, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def expand(self, dims):
        new_dims = [1] * (len(dims) - 1)
        return self.weight.view(new_dims + [-1]).expand(dims + [-1])

    def forward(self, x):
        return torch.cat([x, self.expand([x.shape[0], 1])], dim=1)
