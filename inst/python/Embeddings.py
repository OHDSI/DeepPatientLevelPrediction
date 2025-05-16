import math
from functools import partial

import torch
from torch import nn
import polars as pl

from inst.python.Dataset import FeatureInfo


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int, feature_info: FeatureInfo):
        super(Embedding, self).__init__()
        self.embedding_dim = int(embedding_dim)

        self.vocabulary_size = (
            feature_info["data_reference"]
            .filter(pl.col("isBinary") == "Y")
            .select("columnId")
            .max()
            .item()
        )
        self.numerical_feature_ids = (
            feature_info["data_reference"]
            .filter(pl.col("isBinary") == "N")
            .select("columnId")
            .sort("columnId")
            .to_torch()
            .squeeze(1)
        )

        self.embedding = nn.Embedding(
            self.vocabulary_size + 1 - self.numerical_feature_ids.shape[0],
            embedding_dim,
            padding_idx=0,
        )

        if self.numerical_feature_ids.shape[0] != 0:
            self.numerical_embedding = NumericalEmbedding(
                self.numerical_feature_ids.shape[0], embedding_dim
            )

        # create a router to router the input to the correct embedding such that
        # input_to_numeric[input] will give the index of the numerical feature 
        # in numerical_embedding
        input_to_numeric = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        input_to_numeric[self.numerical_feature_ids] = torch.arange(
            1, self.numerical_feature_ids.shape[0] + 1
        )
        self.register_buffer("input_to_numeric", input_to_numeric)

        input_to_categorical = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        categorical_feature_ids = torch.where(input_to_numeric == 0)[0]
        input_to_categorical[categorical_feature_ids[1:]] = torch.arange(
            1, categorical_feature_ids.numel()
        )
        self.register_buffer("input_to_categorical", input_to_categorical)

    def forward(self, x):
        numerical_mask = torch.isin(
            x["feature_ids"],
            self.numerical_feature_ids.to(x["feature_ids"].device)
        )
        numerical_features = torch.where(
            numerical_mask, x["feature_ids"], 
            torch.tensor(0)
        )
        numerical_mapped_features = self.input_to_numeric[numerical_features]
        numerical_values = torch.where(
            numerical_mask, x["feature_values"], torch.tensor(0.0)
        )
        numerical_embeddings = self.numerical_embedding(
            numerical_mapped_features, numerical_values
        )
        categorical_features = torch.where(
            ~numerical_mask, x["feature_ids"], torch.tensor(0)
        )
        categorical_mapped_features = self.input_to_categorical[categorical_features]
        categorical_embeddings = self.embedding(categorical_mapped_features)
        merge_embeddings = torch.where(
            numerical_mask.unsqueeze(-1), 
            numerical_embeddings, 
            categorical_embeddings
        )
        return merge_embeddings


# class NumericalEmbedding(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, bias=True):
#         super(NumericalEmbedding, self).__init__()
#         # +1 for padding
#         self.weight = nn.Parameter(torch.empty(num_embeddings + 1, embedding_dim))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(num_embeddings + 1, embedding_dim))
#         else:
#             self.bias = None
#
#         for parameter in [self.weight, self.bias]:
#             if parameter is not None:
#                 nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
#
#         with torch.no_grad():
#             self.weight[0].fill_(0)
#             if self.bias is not None:
#                 self.bias[0].fill_(0)
#
#     def forward(self, *inputs):
#         values = inputs[1]
#         ids = inputs[0]
#         # ids are 1-indexed
#         emb = self.weight[ids] * values.unsqueeze(-1)
#         if self.bias is not None:
#             emb = emb + self.bias[ids]
#         return emb
#

# class NumericalEmbedding2(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim):
#         super(NumericalEmbedding2, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings + 1, embedding_dim - 1, padding_idx=0
#         )
#
#     def forward(self, ids, values):
#         x = self.embedding(ids)
#         return torch.cat((x, values[:, None, None].expand(-1, x.shape[1], -1)), dim=-1)


class ClassTokenNested(nn.Module):
    """
    Prepend a class token to the input tensor
    """

    def __init__(self, dim_token):
        super(ClassTokenNested, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, dim_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch._dynamo.disable
    def add_token(self, x):
        return torch.nested.as_nested_tensor(
            [torch.cat([self.weight, seq]) for seq in x.unbind(0)],
            device=x.device,
            layout=x.layout,
        )

    def forward(self, x):
        return self.add_token(x)


class ClassToken(nn.Module):
    def __init__(self, dim_token):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.empty(dim_token, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def expand(self, dims):
        new_dims = [1] * (len(dims) - 1)
        return self.weight.view(new_dims + [-1]).expand(dims + [-1])

    def forward(self, x):
        return torch.cat([self.expand([x.shape[0], 1]), x], dim=1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function that rotates every two elements in the final dimension.
    Works on a tensor with shape (..., head_dim). It splits the last dimension 
    into pairs, then rotates them by replacing (a, b) with (-b, a).

    Args:
        x (torch.Tensor): Input tensor with shape (..., head_dim).

    Returns:
        torch.Tensor: Rotated tensor with the same shape as input.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x_rotated


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (ROPE) that can optionally be scaled.

    Args:
        head_dim (int): the dimension for each attention head.
        base (float): used to compute the inverse frequencies.
        max_time_id (int): the maximum time_id to be supported.
    """

    def __init__(self, head_dim: int, base: float, max_time_id: int = 512):
        super(RotaryEmbedding, self).__init__()
        self.head_dim = head_dim
        self.base = base

        half_dim = head_dim // 2
        inv_freq = 1 / (
            base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        pos = torch.arange(max_time_id, dtype=torch.float32).unsqueeze(
            1
        )  # shape: (max_seq_len, 1)
        angles = pos * inv_freq.unsqueeze(0)  # shape: (max_seq_len, half_dim)

        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # (max_seq_len, head_dim)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("precomputed_sin", sin)
        self.register_buffer("precomputed_cos", cos)

    def forward(self, x, time_ids):
        """

        Applies the rotary embedding to the input tensor.

        Args:
               x (torch.Tensor): Tensor of shape (batch, nheads, seq_len, head_dim).
               time_ids (torch.Tensor): Discrete time IDs of shape (batch, seq_len).
        Returns:
            torch.tensor: Tensor of the same shape as input x with rotary 
            embeddings applied.
        """
        max_pos = self.precomputed_sin.shape[0]
        if time_ids.max() >= max_pos:
            raise ValueError("time_ids exceed precomputed maximum sequence length!")

        sin_emb = self.precomputed_sin[time_ids].unsqueeze(
            1
        )  # (batch, 1, seq_len, head_dim)
        cos_emb = self.precomputed_cos[time_ids].unsqueeze(
            1
        )  # (batch, 1, seq_len, head_dim)

        # Apply the rotary transformation: x_rotated = x * cos + rotate_every_two(x) * sin.
        return (x * cos_emb) + (rotate_every_two(x) * sin_emb)


class NumericalEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 mode: str="scale",
                 bias: bool=True,
                 aggregate: bool=False):
        """
        Merged Numerical Embedding Layer that supports two modes:

        - 'scale':
          Uses an nn.Embedding to lookup a learned vector, which is then
          scaled by the provided numerical value. Optionally, a separate
          bias embedding is added.

        - 'concatenate':
          Uses an nn.Embedding (with output dimension embedding_dim - 1) to
          look up a learned vector and then concatenates the provided numerical
          value (expanded to match dimensions) so that the final output has
          dimension embedding_dim.


        Args:
            num_embeddings (int): Number of embeddings (excluding padding).
            embedding_dim (int): Final embedding dimension.
            mode (str): Either 'scale', 'concatenate' or 'average'.
            aggregate (bool): Whether to use an nn.EmbeddingBag with a sum for the lookup
            bias (bool): Whether to include a bias embedding (only applies to scale mode).
        """
        super(NumericalEmbedding, self).__init__()
        if mode not in ["scale", "concatenate"]:
            raise ValueError("mode must be either 'scale' or 'concatenate'")

        self.mode = mode
        self.aggregate = aggregate
        if self.aggregate:
            embedding_layer = partial(nn.EmbeddingBag, mode="sum")
        else:
            embedding_layer = nn.Embedding
        if self.mode == "scale":
            self.embedding = embedding_layer(num_embeddings + 1,
                                             embedding_dim,
                                             padding_idx=0)
            if bias:
                self.bias_embedding = embedding_layer(num_embeddings + 1,
                                                      embedding_dim,
                                                      padding_idx=0)
            else:
                self.bias_embedding = None
        elif self.mode == "concatenate":
            self.embedding = embedding_layer(num_embeddings + 1,
                                          embedding_dim - 1,
                                          padding_idx=0)


    def forward(self, ids: torch.Tensor, values: torch.Tensor):
        """
        Args:
            ids (LongTensor): Tensor of ids (with index 0 reserved for padding).
            values (Tensor): Numerical values to be integrated into the embedding.

        Returns:
            Tensor: Output embeddings.
        """
        if self.mode == "scale":
            x = self.embedding(ids)
            out = x * values.unsqueeze(-1)
            if self.bias_embedding is not None:
                out = out + self.bias_embedding(ids)
            return out
        else:
            x = self.embedding(ids)
            values_expanded = values.sum(dim=1).unsqueeze(-1)
            return torch.cat([x, values_expanded],
                             dim=-1)