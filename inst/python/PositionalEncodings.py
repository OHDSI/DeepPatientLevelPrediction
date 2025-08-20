from typing import Optional, Callable
import math

import torch
from torch import nn
import torch.nn.functional as F

from Embeddings import RotaryEmbedding


class PositionalEncoding(nn.Module):
    """Abstract base class for all positional encoding methods"""

    def __init__(self, dim_token: int | float, max_time_id: int = 512):
        super().__init__()
        self.dim_token = int(dim_token)
        self.max_time_id = max_time_id

    def apply_additive_pe(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies PE by adding it to the input token embeddings.
        For attention-based PEs, this is typically an identity operation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            **kwargs: Can include 'time_ids' or other necessary data.

        Returns:
            torch.Tensor: Tensor with positional encoding added, shape (B, L, D).
        """
        return x

    def apply_attention_pe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies PE by manipulating query and key vectors within the attention module.
        For additive PEs, this is typically an identity operation.

        Args:
            q (torch.Tensor): Query tensor of shape (B, H, L, D_h).
            k (torch.Tensor): Key tensor of shape (B, H, L, D_h).
            time_ids (Optional[torch.Tensor]): The time indices for the query, shape (B, L_q).
            **kwargs: Other necessary data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified query and key tensors.
        """
        return q, k

    def get_attention_bias(
        self, q_len: int, k_len: int, time_ids: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor | None:
        """
        Calculates a bias tensor to be added directly to the attention scores (logits).
        For PEs that do not modify attention scores, this returns None.

        Args:
            q_len (int): The sequence length of the query.
            k_len (int): The sequence length of the key.
            time_ids (Optional[torch.Tensor]): The time indices for the query, shape (B, L_q).
            **kwargs: Can include 'num_heads' or other necessary data.

        Returns:
            An optional tensor of shape that can be broadcast to (H, L_q, L_k).
            Common shapes are (L_q, L_k) or (H, L_q, L_k).
        """
        return None

    def get_positional_scores(
        self, query: torch.Tensor, time_ids: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Calculates the positional attention scores directly, as in Shaw et al.
        This is typically a 'content-to-position' interaction (e.g., Q @ R^T).

        Args:
            query (torch.Tensor): The query tensor of shape (B, H, L_q, D_h).
            time_ids (torch.Tensor): The time indices for the query, shape (B, L_q).
            **kwargs: Can include k_len or other necessary data.

        Returns:
            An optional tensor of attention scores with shape (B, H, L_q, L_k),
            or None if this method is not implemented.
        """
        return None

    def get_post_softmax_bias(self, *args, **kwargs) -> torch.Tensor | None:
        """
        Calculates a bias to be added to the attention weights *after* softmax.
        This is a highly specialized method for specific PE types.
        """
        return None

    def get_attention_module_class(self) -> type:
        """
        Returns the attention module class that uses this positional encoding.
        This is used for TUPE and similar methods that require a custom attention module.
        Standard positional encoding modules will return the default MultiHeadAttention class.
        """
        from Transformer import MultiHeadAttention
        return MultiHeadAttention


class NoPositionalEncoding(PositionalEncoding):
    """A no-op positional encoding class.
    This is used when no positional encoding is needed, e.g., in some
    non-sequential models or when the model is not time-aware.
    """

    def __init__(self, dim_token: int | float = 0):
        super().__init__(dim_token)

class SinusoidalPE(PositionalEncoding):
    """Sinusoidal positional encoding
    This class pre-computes a sinusoidal embedding for every possible discrete
    timestamp up to `max_time_id`. During the forward pass, it uses the
    provided `time_ids` to perform a direct lookup into this pre-computed table.
    """

    def __init__(
        self,
        dim_token: int | float,
        max_time_id: int = 512,
        base: int = 10000,
        dropout: float = 0.0,
    ):
        super().__init__(dim_token, max_time_id)
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, self.max_time_id).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_token, 2)
            * -(torch.log(torch.tensor(base)) / self.dim_token)
        )
        pe = torch.zeros(self.max_time_id, self.dim_token)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def apply_additive_pe(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Looks up the pre-computed sinusoidal positional encoding and adds it to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            **kwargs: Must include 'time_ids' to index into the pre-computed table.
        Returns:
            torch.Tensor: Tensor with positional encoding added, shape (B, L, D).
        """
        time_ids = kwargs.get("time_ids")
        if time_ids is None:
            raise ValueError("time_ids must be provided for SinusoidalPE")
        x = x + self.pe[time_ids]
        return self.dropout(x)


class LearnablePE(PositionalEncoding):
    """
    Learnable absolute positional encoding.This class uses an `nn.Embedding`
    layer as a lookup table. The size of this table is determined by
    `max_time_id`. During the forward pass, it uses the provided `time_ids` as
    indices to retrieve the learned positional vectors.
    """

    def __init__(
        self, dim_token: int | float, max_time_id: int = 512, dropout: float = 0.1
    ):
        super().__init__(dim_token, max_time_id)
        self.pe = nn.Embedding(self.max_time_id, self.dim_token)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.uniform_(self.pe.weight, -0.02, 0.02)

    def apply_additive_pe(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Looks up the positional embedding and adds it to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            **kwargs: Must include 'time_ids' to index into the embedding layer.
        Returns:
            torch.Tensor: Tensor with positional encoding added, shape (B, L, D)."""
        time_ids = kwargs.get("time_ids")
        if time_ids is None:
            raise ValueError("time_ids must be provided for LearnablePE")
        x = x + self.pe(time_ids)
        return self.dropout(x)


class TapePE(PositionalEncoding):
    """
    time Absolute Positional Encoding (tAPE).
    This method modifies the sinusoidal frequency based on the embedding
    dimension (d_model) and the sequence length (L) to improve distance
    awareness and isotropy, especially when d_model and L are mismatched.

    Note: Due to the dependency on the dynamic sequence length `L`, this
    encoding is calculated on-the-fly during the forward pass.

    from: https://arxiv.org/abs/2305.16642
    """

    def __init__(self, dim_token: int | float, dropout: float = 0.1, base: int = 10000):
        super().__init__(dim_token)
        self.dropout = nn.Dropout(p=dropout)
        self.base = base

        if self.dim_token % 2 != 0:
            raise ValueError(
                f"dim_token must be even for TapePE, but got {self.dim_token}"
            )

    def apply_additive_pe(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates and adds tAPE on the fly using the provided `time_ids`.
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            **kwargs: Must include 'time_ids' and 'lengths' for dynamic sequence length.
        Returns:
            torch.Tensor: Tensor with positional encoding added, shape (B, L, D).
        """
        time_ids = kwargs.get("time_ids")
        lengths = kwargs.get("lengths")
        if time_ids is None:
            raise ValueError("time_ids must be provided for TapePE")
        if lengths is None:
            raise ValueError(
                "TapePE needs dynamic sequence length, lengths should be provided"
            )

        batch_size, seq_len, d_model = x.shape
        L = lengths.float()

        half_dim = d_model // 2
        k_indices = torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
        omega_k = 1.0 / (self.base ** (2 * k_indices / d_model))

        L_reshaped = L.view(-1, 1, 1)
        omega_new = omega_k * (d_model / L_reshaped)

        angles = time_ids.unsqueeze(-1) * omega_new  # (B, L, D/2)

        pe = torch.empty(batch_size, seq_len, d_model, device=x.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        return self.dropout(x + pe)


class RotaryPE(PositionalEncoding):
    """
    Rotary Positional Embedding (RoPE).
    Applies positional information by rotating query and key vectors based on
    their absolute time_ids. This is an attention-based PE.
    """

    def __init__(
        self,
        dim_token: int | float,
        base: int = 10000,
        num_heads: int = 8,
        max_time_id: int = 512,
    ):
        super().__init__(dim_token, max_time_id)
        self.base = base
        self.num_heads = num_heads
        self.head_dim = self.dim_token // self.num_heads
        self.rotary_embedding = RotaryEmbedding(
            head_dim=self.head_dim,
            base=self.base,
            max_time_id=self.max_time_id,
        )

    def apply_attention_pe(
        self, q: torch.Tensor, k: torch.Tensor, time_ids: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to the query and key tensors.
        Args:
            q (torch.Tensor): Query tensor of shape (B, H, L, D_h).
            k (torch.Tensor): Key tensor of shape (B, H, L, D_h).
            **kwargs: Must include 'time_ids' to apply the rotary embedding.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified query and key tensors.
        """
        if time_ids is None:
            raise ValueError("time_ids must be provided for RotaryPE")
        q_len = q.shape[2]
        k_len = k.shape[2]
        q_rotated = self.rotary_embedding(q, time_ids[:, :q_len])
        k_rotated = self.rotary_embedding(k, time_ids[:, :k_len])
        return q_rotated, k_rotated


class RelativePE(PositionalEncoding):
    """
    Relative Positional Encoding (RPE) as in Shaw et al. 2018

    https://arxiv.org/abs/1803.02155.

    This creates a learnable bias that is added to the attention scores based on
    the relative distance between query and key positions.
    """

    def __init__(
        self,
        dim_token: int | float,
        max_relative_position: int = 16,
        num_heads: int = 8,
    ):
        super().__init__(dim_token)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_token // self.num_heads
        self.max_relative_position = max_relative_position

        vocab_size = 2 * self.max_relative_position + 1
        self.relative_embeddings = nn.Embedding(vocab_size, self.head_dim)

    def get_positional_scores(
        self, query: torch.tensor, time_ids: torch.Tensor = None, **kwargs
    ) -> torch.tensor:
        """
        calculates the content-to-position attention scores.
        """
        q_len, k_len = query.shape[2], kwargs.get("k_len")
        if k_len is None:
            raise ValueError("k_len must be provided for shawrelativepe")

        time_ids_slice = time_ids[:, :k_len]

        time_ids_q = time_ids_slice[:, :q_len].unsqueeze(2)  # Shape: (B, L_q, 1)
        time_ids_k = time_ids_slice.unsqueeze(1)  # Shape: (B, 1, L_k)
        relative_pos = time_ids_k - time_ids_q  # Shape: (B, L_q, L_k)
        clipped_pos = torch.clamp(
            relative_pos, -self.max_relative_position, self.max_relative_position
        )
        indices = clipped_pos + self.max_relative_position

        rel_embeddings = self.relative_embeddings(indices)

        positional_scores = torch.einsum("bhqd,bqkd->bhqk", query, rel_embeddings)
        return positional_scores


class EfficientRPE(PositionalEncoding):
    """
    Implements the novel 'eRPE' from the paper: https://arxiv.org/abs/2305.16642

    It adds a learnable, input-independent, scalar bias to the attention weights
    *after* the softmax operation.
    """

    def __init__(
        self, dim_token: int | float, num_heads: int = 8, max_time_id: int = 512
    ):
        super().__init__(dim_token, max_time_id)
        self.num_heads = int(num_heads)

        vocab_size = 2 * self.max_time_id + 1

        self.relative_bias_table = nn.Embedding(vocab_size, self.num_heads)
        torch.nn.init.zeros_(self.relative_bias_table.weight)

    def get_post_softmax_bias(
        self,
        q_len: int,
        k_len: int,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Creates the bias matrix to be added to the attention weights post-softmax.
        """
        if time_ids is None:
            raise ValueError("time_ids must be provided for time-aware EfficientRPE")

        q_t = time_ids[:, :q_len]
        k_t = time_ids[:, :k_len]

        if q_len == 1:
            rel = q_t - k_t
            rel = torch.clamp(rel, -self.max_time_id, self.max_time_id)
            idx = (rel + self.max_time_id).long()
            bias = self.relative_bias_table(idx).permute(0, 2, 1).unsqueeze(2)
            return bias

        rel = q_t.unsqueeze(2) - k_t.unsqueeze(1)
        rel = torch.clamp(rel, -self.max_time_id, self.max_time_id)
        idx = (rel + self.max_time_id).long()
        bias = self.relative_bias_table(idx).permute(0, 3, 1, 2)
        return bias


class TemporalPE(PositionalEncoding):
    """This method combines two components:
    1. A standard time-aware sinusoidal encoding added to the input embeddings
       (`apply_additive_pe`).
    2. A dynamic, content-aware "semantic" component that calculates a similarity
       score (RBF kernel) between token embeddings and adds it as a pre-softmax
       attention bias (`get_attention_bias`).
    """

    def __init__(
        self,
        dim_token: int | float,
        max_time_id: int = 512,
        num_heads: int = 8,
        base: int = 10000,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
    ):
        super().__init__(dim_token, max_time_id)

        self.sinusoidal_pe = SinusoidalPE(dim_token, max_time_id, base)

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(sigma).log())
        else:
            self.register_buffer("log_sigma", torch.tensor(sigma))

        self._original_embeddings = None

    def apply_additive_pe(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies the geometric (sinusoidal) component and caches the original
        embeddings for the semantic component.
        """
        self._original_embeddings = x.detach()
        return self.sinusoidal_pe.apply_additive_pe(x, **kwargs)

    def get_attention_bias(
        self,
        q_len: int,
        k_len: int,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor | None:
        """
        Calculates the semantic (RBF kernel) component as a pre-softmax bias.
        """
        if self._original_embeddings is None:
            return None

        x = self._original_embeddings
        x_q = x[:, :q_len, :]
        x_k = x[:, :k_len, :]

        x_q_norm_sq = torch.sum(x_q**2, dim=-1, keepdim=True)  # Shape: (B, L_q, 1)
        x_k_norm_sq = torch.sum(x_k**2, dim=-1, keepdim=True)  # Shape: (B, L_k, 1)

        dot_product = torch.bmm(x_q, x_k.transpose(1, 2))

        dist_sq = x_q_norm_sq - 2 * dot_product + x_k_norm_sq.transpose(1, 2)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        sigma_val = (
            torch.exp(self.log_sigma) if hasattr(self, "log_sigma") else self.sigma
        )

        similarity_matrix = torch.exp(
            -dist_sq / (2 * sigma_val**2)
        )  # Shape: (B, L_q, L_k)
        return similarity_matrix.unsqueeze(1)


class StochasticConvPE(PositionalEncoding):
    """
    Implementation of  'convSPE' from Liutkus et al. (2021).

    This method generates a relative positional encoding by applying learnable
    convolutions to a tensor of random Gaussian noise. The resulting positional
    encodings are then added to the content-based queries and keys.
    """

    def __init__(
        self,
        dim_token: int | float,
        num_heads: int,
        kernel_size: int = 15,
    ):
        super().__init__(dim_token)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_token // self.num_heads
        self.kernel_size = kernel_size
        padding = (self.kernel_size - 1) // 2

        self.query_conv = nn.Conv1d(
            in_channels=self.dim_token,
            out_channels=self.dim_token,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_heads,
        )

        self.key_conv = nn.Conv1d(
            in_channels=self.dim_token,
            out_channels=self.dim_token,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_heads,
        )
        noise = torch.randn(1, self.max_time_id, self.dim_token)
        self.register_buffer("noise_buffer", noise)

    def apply_attention_pe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates stochastic PEs and adds them to the content-based Q and K.
        """
        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k.shape[2]

        noise_for_q = self.noise_buffer[:, :q_len, :].expand(batch_size, -1, -1)
        noise_for_k = self.noise_buffer[:, :k_len, :].expand(batch_size, -1, -1)

        q_pos_conv = self.query_conv(noise_for_q.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # -> (B, L, D)
        k_pos_conv = self.key_conv(noise_for_k.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # -> (B, L, D)

        q_pos = q_pos_conv.view(batch_size, q_len, num_heads, head_dim).permute(
            0, 2, 1, 3
        )

        k_pos = k_pos_conv.view(batch_size, k_len, num_heads, head_dim).permute(
            0, 2, 1, 3
        )

        q_final = q + q_pos
        k_final = k + k_pos
        return q_final, k_final


class HybridRoPEConvPE(PositionalEncoding):
    """
    A hybrid positional encoding method that combines two powerful concepts:
    1. Rotary Positional Embedding (RoPE) to encode the absolute, irregular
       timestamp of each token via rotation.
    2. Stochastic Convolutional PE (convSPE) to encode relative, shape-based
       patterns by adding a filtered noise vector.

    This allows the model to be simultaneously aware of "when" an event happened
    and "what shape" the local sequence of events has.
    """

    def __init__(
        self,
        dim_token: int | float,
        num_heads: int,
        max_time_id: int,
        kernel_size: int = 15,
        base: int = 10000,
    ):
        super().__init__(dim_token, max_time_id)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_token // self.num_heads

        self.kernel_size = kernel_size
        padding = (self.kernel_size - 1) // 2

        self.query_conv = nn.Conv1d(
            in_channels=self.dim_token,
            out_channels=self.dim_token,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_heads,
        )
        self.key_conv = nn.Conv1d(
            in_channels=self.dim_token,
            out_channels=self.dim_token,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.num_heads,
        )

        self.rope_module = RotaryPE(dim_token=dim_token, base=base, num_heads=num_heads)

        noise = torch.randn(1, self.max_time_id, self.dim_token)
        self.register_buffer("noise_buffer", noise)

    def apply_attention_pe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the sequential transformation: first RoPE, then StochasticConvPE.
        """
        if time_ids is None:
            raise ValueError("time_ids are required for the HybridRoPEConvPE method.")

        q_rotated, k_rotated = self.rope_module.apply_attention_pe(
            q, k, time_ids=time_ids
        )

        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k.shape[2]

        noise_for_q = self.noise_buffer[:, :q_len, :].expand(batch_size, -1, -1)
        noise_for_k = self.noise_buffer[:, :k_len, :].expand(batch_size, -1, -1)

        q_pos_conv = self.query_conv(noise_for_q.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # -> (B, L, D)
        k_pos_conv = self.key_conv(noise_for_k.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # -> (B, L, D)

        q_pos = q_pos_conv.view(batch_size, q_len, num_heads, head_dim).permute(
            0, 2, 1, 3
        )

        k_pos = k_pos_conv.view(batch_size, k_len, num_heads, head_dim).permute(
            0, 2, 1, 3
        )

        q_final = q_rotated + q_pos
        k_final = k_rotated + k_pos

        return q_final, k_final


class TUPE(PositionalEncoding):
    """
    A 'signal' class for the Transformer with Untied Positional Encoding (TUPE).

    This class holds the learnable projection layers for the positional queries
    and keys. It uses a standard time-aware absolute PE as its base.
    """

    def __init__(
        self,
        dim_token: int | float,
        max_time_id: int,
        base_pe_config: dict = {"name": "SinusoidalPE", "dropout": 0.1},
        model_parameters: Optional[dict] = None,
    ):
        super().__init__(dim_token, max_time_id)

        if model_parameters is None:
            raise ValueError(
                "TUPE requires the full `model_parameters` for initialization."
            )

        # 1. Create the base absolute positional encoding module (time-aware)
        factory_params = model_parameters.copy()
        factory_params["positional_encoding"] = base_pe_config
        self.base_pos_encoder = create_positional_encoding_module(factory_params)

        # 2. Create the untied projection layers for the positional Q/K
        # This is the `pos_kq` linear layer from the review's code.
        self.pos_kq_proj = nn.Linear(self.dim_token, 2 * self.dim_token, bias=False)

    def get_positional_queries_keys(
        self, time_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the untied positional queries and keys for the attention module.
        """
        # Get the time-aware absolute positional embeddings. Shape: (B, L, D)
        # We need a dummy tensor of the right shape and device to pass to apply_additive_pe.
        dummy_x = torch.zeros(
            time_ids.shape[0], time_ids.shape[1], self.dim_token, device=time_ids.device
        )
        pos_embed = self.base_pos_encoder.apply_additive_pe(dummy_x, time_ids=time_ids)

        # Project them to get positional queries and keys
        pos_key, pos_query = self.pos_kq_proj(pos_embed).chunk(2, dim=-1)

        return pos_query, pos_key

    def get_attention_module_class(self) -> type:
        # Import here to avoid circular dependency
        return TUPEMultiHeadAttention


class TUPEMultiHeadAttention(nn.Module):
    """
    Implements the Untied Positional Encoding (TUPE) attention mechanism.

    The attention score is a sum of a content-content term and a
    position-position term, calculated with separate projection matrices.
    """

    def __init__(
        self, dim_token: int, nheads: int, dropout_p: float, pe_module: "TUPE"
    ):
        super().__init__()
        self.nheads = nheads
        self.E_head = dim_token // nheads
        self.pe_module = pe_module
        self.dropout_p = dropout_p

        self.content_qkv_proj = nn.Linear(dim_token, 3 * dim_token, bias=False)
        self.out_proj = nn.Linear(dim_token, dim_token)
        self.scale = math.sqrt(self.E_head)

    def forward(
        self,
        x: torch.Tensor,
        query_selector: Callable,
        mask: torch.Tensor,
        time_ids: Optional[torch.Tensor],
    ):
        if time_ids is None:
            raise ValueError("TUPEMultiHeadAttention requires time_ids.")

        x_query = query_selector(x)
        Lq, Lk = x_query.size(1), x.size(1)

        # 1. --- Content Path ---
        content_query, content_key, content_value = self.content_qkv_proj(x).chunk(
            3, dim=-1
        )
        content_query = query_selector(content_query)

        content_query = content_query.view(
            x_query.size(0), Lq, self.nheads, self.E_head
        ).transpose(1, 2)
        content_key = content_key.view(
            x.size(0), Lk, self.nheads, self.E_head
        ).transpose(1, 2)
        content_value = content_value.view(
            x.size(0), Lk, self.nheads, self.E_head
        ).transpose(1, 2)

        # 2. --- Position Path ---
        pos_query, pos_key = self.pe_module.get_positional_queries_keys(time_ids)
        pos_query = query_selector(pos_query)

        pos_query = pos_query.view(
            x_query.size(0), Lq, self.nheads, self.E_head
        ).transpose(1, 2)
        pos_key = pos_key.view(x.size(0), Lk, self.nheads, self.E_head).transpose(1, 2)

        # 3. --- Combine Scores ---
        # Term 1: Content-Content Score
        content_score = torch.einsum("bhqd,bhkd->bhqk", content_query, content_key)

        # Term 2: Position-Position Score
        pos_score = torch.einsum("bhqd,bhkd->bhqk", pos_query, pos_key)

        attn_scores = (content_score + pos_score) / self.scale

        # 4. --- Standard Attention Finale ---
        attn_mask = mask[:, None, None, :Lk].contiguous()
        attn_scores = attn_scores.masked_fill(~attn_mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        attn_output = attn_weights @ content_value
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        return self.out_proj(attn_output)


def create_positional_encoding_module(model_parameters: dict) -> PositionalEncoding:
    """
    Factory function to create a positional encoding module.

    Args:
        model_parameters (dict):
            The main model parameters dictionary, used to fetch
            shared values like dim_token and max_time_id.

    Returns:
        An instantiated PositionalEncoding module.
    """
    PE_MODULE_REGISTRY = {
        "SinusoidalPE": SinusoidalPE,
        "LearnablePE": LearnablePE,
        "TapePE": TapePE,
        "RotaryPE": RotaryPE,
        "RelativePE": RelativePE,
        "EfficientRPE": EfficientRPE,
        "TemporalPE": TemporalPE,
        "StochasticConvPE": StochasticConvPE,
        "HybridRoPEConvPE": HybridRoPEConvPE,
        "TUPE": TUPE,
    }
    pe_config = model_parameters["positional_encoding"]

    if not isinstance(pe_config, dict) or "name" not in pe_config:
        raise ValueError("Positional encoding config must be a dict with a 'name' key.")

    pe_name = pe_config["name"]

    if pe_name not in PE_MODULE_REGISTRY:
        raise ValueError(
            f"Unknown positional encoding module: {pe_name}. "
            f"Available modules: {list(PE_MODULE_REGISTRY.keys())}"
        )

    pe_class = PE_MODULE_REGISTRY[pe_name]

    constructor_args = {
        "dim_token": model_parameters["dim_token"],
        "max_time_id": model_parameters["feature_info"].get_max_time_id(),
        "num_heads": model_parameters["num_heads"],
    }
    SPECIAL_CONTENT_CLASES = (TemporalPE, TUPE)

    if issubclass(pe_class, SPECIAL_CONTENT_CLASES):
        constructor_args["model_parameters"] = model_parameters

    constructor_args.update(pe_config)

    # Filter args to only those accepted by the specific PE class constructor
    # This is robust and prevents errors if extra args are passed
    import inspect

    sig = inspect.signature(pe_class.__init__)
    valid_args = {k: v for k, v in constructor_args.items() if k in sig.parameters}
    return pe_class(**valid_args)
