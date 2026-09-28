# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional as F

from Dataset import FeatureInfo


class Embedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_info: FeatureInfo,
        numeric_mode: str = "scale",
        numerical_num_frequencies: int = 8,
        numerical_periodic_init_std: float = 0.1,
        numerical_pbld_hidden_dim: int = 16,
        numerical_pbld_embedding_dim: int = 4,
        numerical_bias: bool = True,
        numerical_bias_scale: float = 1.0,
        numerical_bias_normalization: str = "none",
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
                num_frequencies=int(numerical_num_frequencies),
                periodic_init_std=float(numerical_periodic_init_std),
                pbld_hidden_dim=int(numerical_pbld_hidden_dim),
                pbld_embedding_dim=int(numerical_pbld_embedding_dim),
                bias=numerical_bias,
                bias_scale=float(numerical_bias_scale),
                bias_normalization=str(numerical_bias_normalization),
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

    def _categorical_token_embeddings(self, categorical_mapped_features):
        if isinstance(self.embedding, nn.EmbeddingBag):
            return F.embedding(
                categorical_mapped_features,
                self.embedding.weight,
                padding_idx=0,
            )
        return self.embedding(categorical_mapped_features)

    def _split_feature_components(self, x: dict[str, torch.Tensor]):
        feature_ids = x["feature_ids"]
        feature_values = x["feature_values"]
        numerical_mask = self.input_to_numeric[feature_ids] != 0

        categorical_features = feature_ids.clone()
        categorical_features[numerical_mask] = 0
        categorical_mapped_features = self.input_to_categorical[categorical_features]

        numerical_features = feature_ids.clone()
        numerical_features[~numerical_mask] = 0
        numerical_mapped_features = self.input_to_numeric[numerical_features]

        numerical_values = feature_values.clone()
        numerical_values[~numerical_mask] = 0.0
        return (
            numerical_mask,
            categorical_mapped_features,
            numerical_mapped_features,
            numerical_values,
        )

    def get_aggregation_diagnostics(self, x: dict[str, torch.Tensor], mode: str = "sum"):
        (
            numerical_mask,
            categorical_mapped_features,
            numerical_mapped_features,
            numerical_values,
        ) = self._split_feature_components(x)
        valid_mask = x["feature_ids"] != 0

        categorical_tokens = self._categorical_token_embeddings(categorical_mapped_features)
        sum_categorical = categorical_tokens.sum(dim=1)
        if self.numerical_embedding is None:
            sum_num_value = torch.zeros_like(sum_categorical)
            sum_num_bias = torch.zeros_like(sum_categorical)
        elif (
            self.numerical_embedding.mode == "scale"
            and self.numerical_embedding.bias_embedding is not None
        ):
            value_tokens, bias_tokens = self.numerical_embedding.scale_components(
                numerical_mapped_features,
                numerical_values,
            )
            sum_num_value = value_tokens.sum(dim=1)
            sum_num_bias = bias_tokens.sum(dim=1)
        else:
            numerical_tokens = self.numerical_embedding(
                numerical_mapped_features,
                numerical_values,
            )
            if numerical_tokens.dim() == 2:
                sum_num_value = numerical_tokens
            else:
                sum_num_value = numerical_tokens.sum(dim=1)
            sum_num_bias = torch.zeros_like(sum_num_value)

        sum_total = sum_categorical + sum_num_value + sum_num_bias

        counts = valid_mask.sum(dim=1).clamp_min(1).to(dtype=sum_total.dtype)
        if mode == "mean":
            aggregated = sum_total / counts.unsqueeze(-1)
        elif mode in ("sum_sqrt_len", "sum_len_norm"):
            aggregated = sum_total / torch.sqrt(counts).unsqueeze(-1)
        else:
            aggregated = sum_total

        num_counts = numerical_mask.sum(dim=1).to(dtype=sum_total.dtype)
        cat_counts = (valid_mask & ~numerical_mask).sum(dim=1).to(dtype=sum_total.dtype)
        return {
            "tokens_per_sample_mean": counts.mean().item(),
            "numeric_tokens_per_sample_mean": num_counts.mean().item(),
            "categorical_tokens_per_sample_mean": cat_counts.mean().item(),
            "sum_norm_mean": torch.linalg.norm(sum_total, dim=1).mean().item(),
            "sum_value_norm_mean": torch.linalg.norm(sum_num_value, dim=1).mean().item(),
            "sum_bias_norm_mean": torch.linalg.norm(sum_num_bias, dim=1).mean().item(),
            "aggregated_norm_mean": torch.linalg.norm(aggregated, dim=1).mean().item(),
        }

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        (
            numerical_mask,
            categorical_mapped_features,
            numerical_mapped_features,
            numerical_values,
        ) = self._split_feature_components(x)
        categorical_embeddings = self.embedding(categorical_mapped_features)

        if self.numerical_embedding is None:
            if self.aggregate == "none":
                return categorical_embeddings
            else:
                return categorical_embeddings / numerical_mask.shape[1]
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
        num_frequencies: int = 8,
        periodic_init_std: float = 0.1,
        pbld_hidden_dim: int = 16,
        pbld_embedding_dim: int = 4,
        bias: bool = True,
        bias_scale: float = 1.0,
        bias_normalization: str = "none",
        aggregate: bool = False,
    ):
        """
        Merged Numerical Embedding Layer that supports multiple modes:

        - 'scale':
          Uses an nn.Embedding to lookup a learned vector, which is then
          scaled by the provided numerical value. Optionally, a separate
          bias embedding is added.

        - 'concatenate':
          Uses an nn.Embedding (with output dimension embedding_dim - 1) to
          look up a learned vector and then concatenates the provided numerical
          value (expanded to match dimensions) so that the final output has
          dimension embedding_dim.

        - 'pl':
          Periodic + linear numerical embeddings inspired by RealMLP/RTDL.

        - 'pbld':
          Periodic biased low-dimensional numerical embeddings (PBLD-inspired)
          with a per-feature two-layer block followed by projection.

        Args:
            num_embeddings (int): Number of embeddings (excluding padding).
            embedding_dim (int): Final embedding dimension.
            mode (str): One of 'scale', 'concatenate', 'pl', 'pbld'.
            aggregate (bool): Whether to use an nn.EmbeddingBag with a sum for the lookup
            bias (bool): Whether to include a bias embedding (only applies to scale mode).
        """
        super(NumericalEmbedding, self).__init__()
        if mode not in ["scale", "concatenate", "pl", "pbld"]:
            raise ValueError(
                "mode must be one of: 'scale', 'concatenate', 'pl', 'pbld'"
            )
        if bias_normalization not in ("none", "mean", "sqrt_len"):
            raise ValueError(
                "bias_normalization must be one of: none, mean, sqrt_len"
            )
        if int(num_frequencies) <= 0:
            raise ValueError("num_frequencies must be > 0")
        if int(pbld_hidden_dim) <= 0:
            raise ValueError("pbld_hidden_dim must be > 0")
        if int(pbld_embedding_dim) <= 0:
            raise ValueError("pbld_embedding_dim must be > 0")
        if mode == "concatenate" and int(embedding_dim) <= 1:
            raise ValueError("embedding_dim must be > 1 when mode='concatenate'")

        self.mode = mode
        self.aggregate = aggregate
        self.bias_scale = float(bias_scale)
        self.bias_normalization = bias_normalization
        self.embedding_dim = int(embedding_dim)
        self.num_frequencies = int(num_frequencies)
        self.pbld_hidden_dim = int(pbld_hidden_dim)
        self.pbld_embedding_dim = int(pbld_embedding_dim)

        self.embedding = None
        self.bias_embedding = None
        self.periodic_frequency = None
        self.periodic_phase = None
        self.pl_projection = None
        self.pbld_w1 = None
        self.pbld_b1 = None
        self.pbld_w2 = None
        self.pbld_b2 = None
        self.pbld_activation = None
        self.pbld_projection = None

        if mode in ("scale", "concatenate"):
            self.embedding = nn.Embedding(
                num_embeddings + 1,
                embedding_dim if mode == "scale" else embedding_dim - 1,
                padding_idx=0,
            )
            if mode == "scale" and bias:
                self.bias_embedding = nn.Embedding(
                    num_embeddings + 1,
                    embedding_dim,
                    padding_idx=0,
                )
        else:
            periodic_dim = int(self.num_frequencies)
            self.periodic_frequency = nn.Embedding(
                num_embeddings + 1,
                periodic_dim,
                padding_idx=0,
            )
            self.periodic_phase = nn.Embedding(
                num_embeddings + 1,
                periodic_dim,
                padding_idx=0,
            )
            with torch.no_grad():
                nn.init.normal_(
                    self.periodic_frequency.weight[1:],
                    mean=0.0,
                    std=float(periodic_init_std),
                )
                self.periodic_frequency.weight[0].zero_()
                nn.init.uniform_(
                    self.periodic_phase.weight[1:],
                    -math.pi,
                    math.pi,
                )
                self.periodic_phase.weight[0].zero_()

            base_dim = 1 + 2 * periodic_dim
            if mode == "pl":
                self.pl_projection = nn.Linear(base_dim, embedding_dim)
            else:
                hidden_dim = int(self.pbld_hidden_dim)
                out_dim = int(self.pbld_embedding_dim)
                self.pbld_w1 = nn.Embedding(
                    num_embeddings + 1,
                    base_dim * hidden_dim,
                    padding_idx=0,
                )
                self.pbld_b1 = nn.Embedding(
                    num_embeddings + 1,
                    hidden_dim,
                    padding_idx=0,
                )
                self.pbld_w2 = nn.Embedding(
                    num_embeddings + 1,
                    hidden_dim * out_dim,
                    padding_idx=0,
                )
                self.pbld_b2 = nn.Embedding(
                    num_embeddings + 1,
                    out_dim,
                    padding_idx=0,
                )
                self.pbld_activation = nn.SiLU()
                self.pbld_projection = nn.Linear(1 + out_dim, embedding_dim)
                self._reset_pbld_feature_layers(base_dim, hidden_dim, out_dim)

    def _reset_pbld_feature_layers(self, in_dim: int, hidden_dim: int, out_dim: int):
        std_1 = 1.0 / math.sqrt(max(1, in_dim))
        std_2 = 1.0 / math.sqrt(max(1, hidden_dim))
        with torch.no_grad():
            nn.init.normal_(self.pbld_w1.weight[1:], mean=0.0, std=std_1)
            nn.init.normal_(self.pbld_w2.weight[1:], mean=0.0, std=std_2)
            nn.init.zeros_(self.pbld_b1.weight)
            nn.init.zeros_(self.pbld_b2.weight)
            self.pbld_w1.weight[0].zero_()
            self.pbld_w2.weight[0].zero_()
            self.pbld_b1.weight[0].zero_()
            self.pbld_b2.weight[0].zero_()

    def _featurewise_linear(
        self,
        inputs: torch.Tensor,
        ids: torch.Tensor,
        weight_embedding: nn.Embedding,
        bias_embedding: nn.Embedding,
        out_dim: int,
    ):
        in_dim = inputs.shape[-1]
        weights = weight_embedding(ids).view(*ids.shape, in_dim, int(out_dim))
        biases = bias_embedding(ids)
        return torch.einsum("bli,blio->blo", inputs, weights) + biases

    def _periodic_basis(self, ids: torch.Tensor, values: torch.Tensor):
        values = values.unsqueeze(-1)
        freq = self.periodic_frequency(ids)
        phase = self.periodic_phase(ids)
        angles = values * freq + phase
        periodic = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        basis = torch.cat((values, periodic), dim=-1)
        valid_mask = (ids != 0).unsqueeze(-1).to(dtype=basis.dtype)
        return basis * valid_mask

    def _pl_forward(self, ids: torch.Tensor, values: torch.Tensor):
        basis = self._periodic_basis(ids, values)
        out = self.pl_projection(basis)
        valid_mask = (ids != 0).unsqueeze(-1).to(dtype=out.dtype)
        return out * valid_mask

    def _pbld_forward(self, ids: torch.Tensor, values: torch.Tensor):
        basis = self._periodic_basis(ids, values)
        hidden = self._featurewise_linear(
            inputs=basis,
            ids=ids,
            weight_embedding=self.pbld_w1,
            bias_embedding=self.pbld_b1,
            out_dim=self.pbld_hidden_dim,
        )
        hidden = self.pbld_activation(hidden)
        low_dim = self._featurewise_linear(
            inputs=hidden,
            ids=ids,
            weight_embedding=self.pbld_w2,
            bias_embedding=self.pbld_b2,
            out_dim=self.pbld_embedding_dim,
        )
        values_column = values.unsqueeze(-1)
        merged = torch.cat((values_column, low_dim), dim=-1)
        out = self.pbld_projection(merged)
        valid_mask = (ids != 0).unsqueeze(-1).to(dtype=out.dtype)
        return out * valid_mask

    def scale_components(self, ids: torch.Tensor, values: torch.Tensor):
        out = self.embedding(ids)
        out.mul_(values.unsqueeze(-1))

        bias = None
        if self.bias_embedding is not None:
            bias = self.bias_embedding(ids)
            if self.bias_normalization != "none":
                lengths = (ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
                lengths = lengths.to(dtype=bias.dtype)
                if self.bias_normalization == "mean":
                    divisor = lengths
                else:
                    divisor = torch.sqrt(lengths)
                bias = bias / divisor.unsqueeze(-1)
            if self.bias_scale != 1.0:
                bias = bias * self.bias_scale
        return out, bias

    def forward(self, ids: torch.Tensor, values: torch.Tensor):
        """
        Args:
            ids (LongTensor): Tensor of ids (with index 0 reserved for padding).
            values (Tensor): Numerical values to be integrated into the embedding.

        Returns:
            Tensor: Output embeddings.
        """
        if self.mode == "scale":
            out, bias = self.scale_components(ids, values)
            if bias is not None:
                out.add_(bias)
            if self.aggregate:
                out = out.sum(dim=1)
            return out

        if self.mode == "pl":
            out = self._pl_forward(ids, values)
            if self.aggregate:
                out = out.sum(dim=1)
            return out

        if self.mode == "pbld":
            out = self._pbld_forward(ids, values)
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
