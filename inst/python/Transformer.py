import math

import torch
from torch import nn
import torch.nn.functional as F

from inst.python.Embeddings import Embedding, ClassToken


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
        num_blocks: int, # model architecture
        dim_token: int, # embedding size
        num_heads: int, # number of heads in multihead attention
        att_dropout, #
        ffn_dropout,
        res_dropout,
        dim_hidden: int, # hidden layer size
        dim_out: int = 1,
        head_activation=nn.ReLU,
        activation=ReGLU,
        ffn_norm=nn.LayerNorm,
        head_norm=nn.LayerNorm,
        att_norm=nn.LayerNorm,
        model_type="Transformer",
        temporal = False
    ):
        super(Transformer, self).__init__()
        self.name = model_type
        num_blocks = int(num_blocks)
        dim_token = int(dim_token)
        num_heads = int(num_heads)
        dim_hidden = int(dim_hidden)
        dim_out = int(dim_out)

        self.embedding = Embedding(embedding_dim=dim_token,
                                   feature_info=feature_info)

        self.class_token = ClassToken(dim_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(num_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": MultiHeadAttention(
                        E_q=dim_token,
                        E_k=dim_token,
                        E_v=dim_token,
                        E_total=dim_token,
                        nheads=num_heads,
                        dropout_p=att_dropout
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

    def forward(self, x):
        x = self.embedding(x)
        x = self.class_token(x)
        mask = (x != 0).any(dim=-1)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x_residual = self.start_residual(layer, "attention", x)
                x_residual = layer["attention"](
                    x_residual,
                    x_residual,
                    x_residual,
                    mask)
                x = self.end_residual(layer, "attention", x, x_residual)
            else:
                x_residual = self.start_residual(layer, "attention", x)
                x_residual = layer["attention"](
                    x_residual[:, :1],
                    x_residual,
                    x_residual,
                    mask[:, :1]
                )
                x = self.end_residual(layer, "attention", x[:, :1], x_residual)
            x_residual = self.start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self.end_residual(layer, "ffn", x, x_residual)
        x = self.head(x)
        return x.squeeze()

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
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout_p (float, optional): Dropout probability. Default: 0.0
    """
    def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int,
                 nheads: int, dropout_p: float = 0.0):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.query_proj = nn.Linear(E_q, E_total)
        self.key_proj = nn.Linear(E_k, E_total)
        self.value_proj = nn.Linear(E_v, E_total)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_t, E_q)
            key (torch.Tensor): key of shape (N, L_s, E_k)
            value (torch.Tensor): value of shape (N, L_s, E_v)
            mask (torch.Tensor): mask of shape (N, L_t, E_q)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        # TODO: demonstrate packed projection
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        mask = mask.unsqueeze(1)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout_p, is_causal=False,
            attn_mask=mask
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output



