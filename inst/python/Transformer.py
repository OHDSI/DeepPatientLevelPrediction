import math

import torch
from torch import nn
import torch.nn.functional as F

from ResNet import NumericalEmbedding


def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class Transformer(nn.Module):

    def __init__(self,
                 cat_features,
                 num_features,
                 num_blocks,
                 dim_token,
                 num_heads,
                 att_dropout,
                 ffn_dropout,
                 res_dropout,
                 dim_hidden,
                 dim_out=1,
                 head_activation=nn.ReLU,
                 activation=ReGLU,
                 ffn_norm=nn.LayerNorm,
                 head_norm=nn.LayerNorm,
                 att_norm=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.name = "Transformer"
        self.categorical_embedding = nn.Embedding(cat_features + 1, dim_token, padding_idx=0)

        if num_features != 0 and num_features is not None:
            self.numerical_embedding = NumericalEmbedding(num_features, dim_token)
        self.class_token = ClassToken(dim_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(num_blocks):
            layer = nn.ModuleDict({
                "attention": nn.MultiheadAttention(dim_token, num_heads,
                                                   dropout=att_dropout),
                "ffn": FeedForwardBlock(dim_token, dim_hidden,
                                        bias_first=True,
                                        bias_second=True,
                                        dropout=ffn_dropout,
                                        activation=activation),
                "attention_res_dropout": nn.Dropout(res_dropout),
                "ffn_res_dropout": nn.Dropout(res_dropout),
                "ffn_norm": ffn_norm(dim_token)
            })
            if layer_idx != 0:
                layer["attention_norm"] = att_norm(dim_token)
            self.layers.append(layer)

        self.head = Head(dim_token,
                         bias=True,
                         activation=head_activation,
                         normalization=head_norm,
                         dim_out=dim_out)

    def forward(self, x):
        mask = torch.where(x["cat"] == 0, True, False)
        cat = self.categorical_embedding(x["cat"])
        if "num" in x.keys() and self.numerical_embedding is not None:
            num = self.numerical_embedding(x["num"])
            x = torch.cat([cat, num], dim=1)
            mask = torch.cat([mask, torch.zeros([x.shape[0],
                                                 num.shape[1]],
                                                device=mask.device,
                                                dtype=mask.dtype)],
                             dim=1)
        else:
            x = cat
        x = self.class_token(x)
        mask = torch.cat([mask, torch.zeros([x.shape[0], 1],
                                            device=mask.device,
                                            dtype=mask.dtype)],
                         dim=1)

        for i, layer in enumerate(self.layers):
            x_residual = self.start_residual(layer, "attention", x)

            if i == len(self.layers)-1:
                dims = x_residual.shape
                x_residual = layer["attention"](
                    x_residual[:, -1].view([dims[0], 1, dims[2]]).transpose(0, 1),
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    mask
                )
                attn_weights = x_residual[1]
                x_residual = x_residual[0]
                x = x[:, -1].view([dims[0], 1, dims[2]])
            else:
                x_residual = layer["attention"](
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    x_residual.transpose(0, 1),
                    mask
                )[0]
            x = self.end_residual(layer, "attention", x, x_residual.transpose(0, 1))

            x_residual = self.start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self.end_residual(layer, "ffn", x, x_residual)

        x = self.head(x)[:, 0]
        return x

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

    def __init__(self,
                 dim_token,
                 dim_hidden,
                 bias_first=True,
                 bias_second=True,
                 dropout=0.0,
                 activation=ReGLU):
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

    def __init__(self,
                 dim_in,
                 bias,
                 activation,
                 normalization,
                 dim_out):
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

    def __init__(self,
                 dim_token):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.empty(dim_token, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def expand(self, dims):
        new_dims = [1] * (len(dims) - 1)
        return self.weight.view(new_dims + [-1]).expand(dims +[-1])

    def forward(self, x):
        return torch.cat([x, self.expand([x.shape[0], 1])], dim=1)
