import math

import torch
from torch import nn

from Dataset import FeatureInfo


class Embedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_info: FeatureInfo,
        numeric_mode: str = "scale",
        aggregate: str = "none",
    ):
        super(Embedding, self).__init__()
        assert aggregate in ("none", "sum")
        self.embedding_dim = int(embedding_dim)
        self.aggregate = aggregate

        self.feature_info = feature_info
        self.vocabulary_size = feature_info.get_vocabulary_size()
        self.numerical_feature_ids = feature_info.get_numerical_feature_ids()

        n_num = self.numerical_feature_ids.numel()
        self.n_num = n_num
        n_cat = self.vocabulary_size  - n_num
        self.n_cat = n_cat

        if n_num > 0:
            self.numerical_embedding = NumericalEmbedding(
                num_embeddings=n_num,
                embedding_dim=embedding_dim,
                mode=numeric_mode,
                aggregate=(aggregate != "none"),
            )
        else:
            self.numerical_embedding = None

        if aggregate == "none":
            self.embedding = nn.Embedding(
                num_embeddings=n_cat + 1, 
                embedding_dim=embedding_dim, 
                padding_idx=0
            )
        else:
            self.embedding = nn.EmbeddingBag(
                num_embeddings=n_cat + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
                mode=aggregate,
            )

        # create a router to router the input to the correct embedding such that
        # input_to_numeric[input] will give the index of the numerical feature
        # in numerical_embedding
        input_to_numeric = torch.zeros(n_cat + n_num + 1, dtype=torch.long)
        input_to_numeric[self.numerical_feature_ids] = torch.arange(
            1, self.numerical_feature_ids.shape[0] + 1
        )
        self.register_buffer("input_to_numeric", input_to_numeric)

        input_to_categorical = torch.zeros(n_cat + n_num + 1, dtype=torch.long)
        categorical_feature_ids = torch.where(input_to_numeric == 0)[0]
        input_to_categorical[categorical_feature_ids[1:]] = torch.arange(
            1, categorical_feature_ids.numel()
        )
        self.register_buffer("input_to_categorical", input_to_categorical)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        numerical_mask = self.input_to_numeric[x["feature_ids"]] != 0
        categorical_features = x["feature_ids"].clone()
        categorical_features[numerical_mask] = 0
        categorical_mapped_features = self.input_to_categorical[categorical_features]
        categorical_embeddings = self.embedding(categorical_mapped_features)

        if self.numerical_embedding is None:
            if self.aggregate == "none":
                return categorical_embeddings
            else:
                return categorical_embeddings / numerical_mask.shape[1]

        numerical_features = x["feature_ids"].clone()
        numerical_features[~numerical_mask] = 0

        numerical_mapped_features = self.input_to_numeric[numerical_features]

        numerical_values = x["feature_values"].clone()
        numerical_values[~numerical_mask] = 0.0
        numerical_embeddings = self.numerical_embedding(
            numerical_mapped_features, numerical_values
        )

        if self.aggregate == "none":
            merged_embeddings = torch.where(
                numerical_mask.unsqueeze(-1),
                numerical_embeddings,
                categorical_embeddings,
            )
        else:
            merged_embeddings = (
                categorical_embeddings + numerical_embeddings
            ) / numerical_mask.shape[1]
        return merged_embeddings


class ClassToken(nn.Module):
    def __init__(self, dim_token):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, 1, dim_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        B, L, D = x.shape
        out = x.new_empty(B, L + 1, D)
        out[:, 0] = self.weight
        out[:, 1:] = x
        return out


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
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "scale",
        bias: bool = True,
        aggregate: bool = False,
    ):
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

        self.embedding = nn.Embedding(
            num_embeddings + 1,
            embedding_dim if mode == "scale" else embedding_dim - 1,
            padding_idx=0,
        )

        if mode == "scale" and bias:
            self.bias_embedding = nn.Embedding(
                num_embeddings + 1, embedding_dim, padding_idx=0
            )
        else:
            self.bias_embedding = None

    def forward(self, ids: torch.Tensor, values: torch.Tensor):
        """
        Args:
            ids (LongTensor): Tensor of ids (with index 0 reserved for padding).
            values (Tensor): Numerical values to be integrated into the embedding.

        Returns:
            Tensor: Output embeddings.
        """
        if self.mode == "scale":
            out = self.embedding(ids)
            out.mul_(values.unsqueeze(-1))

            if self.bias_embedding is not None:
                out.add_(self.bias_embedding(ids))
            if self.aggregate:
                out = out.sum(dim=1)
            return out

        x = self.embedding(ids)
        if self.aggregate:
            x = x.sum(dim=1)
            values = values.sum(dim=1)
            B, Dm1 = x.shape
            out = x.new_empty(B, Dm1 + 1)  # (B, D)

            out[:, :Dm1] = x
            out[:, Dm1] = values  # broadcast along B
            return out
        else:
            values = values.unsqueeze(-1)
            B, L, Dm1 = x.shape
            out = x.new_empty(B, L, Dm1 + 1)  # (B, L, D)
            out[..., :Dm1] = x
            out[..., Dm1] = values.squeeze(-1)
            return out
