from typing import Optional

import torch
from torch import nn

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

    def __init__(self, dim_token: int | float, base: int = 10000, num_heads: int = 8):
        super().__init__(dim_token)
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
        self, 
        dim_token: int | float, 
        num_heads: int = 8, 
        max_relative_position: int = 128
    ):
        super().__init__(dim_token)
        self.num_heads = int(num_heads)
        self.max_relative_position = max_relative_position
        
        vocab_size = 2 * self.max_relative_Position - 1

        self.relative_bias_table = nn.Embedding(
            vocab_size, self.num_heads
        )
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
        time_ids_slice = time_ids[:, :k_len]
        time_ids_q = time_ids_slice[:, :q_len].unsqueeze(2)  # Shape: (B, L_q, 1)
        time_ids_k = time_ids_slice.unsqueeze(1)  # Shape: (B, 1, L_k)
        relative_pos = time_ids_k - time_ids_q  # Shape: (B, L_q, L_k)

        clipped_pos = torch.clamp(
            relative_pos, -self.max_relative_position, self.max_relative_position
        )

        indices = clipped_pos + self.max_relative_position
        post_softmax_bias = self.relative_bias_table(indices)
        post_softmax_bias = post_softmax_bias.permute(0, 3, 1, 2)
        return post_softmax_bias


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

    constructor_args.update(pe_config)

    # Filter args to only those accepted by the specific PE class constructor
    # This is robust and prevents errors if extra args are passed
    import inspect

    sig = inspect.signature(pe_class.__init__)
    valid_args = {k: v for k, v in constructor_args.items() if k in sig.parameters}
    return pe_class(**valid_args)
