from typing import Optional, Type

import torch
import torch.nn.functional as F
from torch import nn

from Embeddings import ClassToken, Embedding, RotaryEmbedding
from Dataset import FeatureInfo

def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class Transformer(nn.Module):
    def __init__(
        self,
        feature_info: FeatureInfo,
        num_blocks: int,
        dim_token: int,
        num_heads: int,
        att_dropout: float,
        ffn_dropout: float,
        dim_hidden: int,
        dim_out: int = 1,
        head_activation: Type[nn.Module] = nn.ReLU,
        activation: Type[nn.Module] = ReGLU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        head_norm: Type[nn.Module] = nn.LayerNorm,
        use_rope: bool = False,
        model_type="Transformer",
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

        self.layers = nn.ModuleList()
        for layer_idx in range(num_blocks):
            block = TransformerBlock(
                dim_token=dim_token,
                num_heads=num_heads,
                att_dropout=att_dropout,
                ffn_dropout=ffn_dropout,
                dim_hidden=dim_hidden,
                norm_layer=norm_layer,
                activation=activation,
                skip_attn_norm=(layer_idx == 0),
                only_class_token=(layer_idx == num_blocks - 1),
                use_rope=use_rope,
                max_time_id=feature_info.get_max_time_id() if use_rope else None,
            )
            self.layers.append(block)

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
        if "time_ids" in x.keys() and x["time_ids"] is not None:
            time_ids = x["time_ids"]
        else:
            time_ids = None
        mask = x["feature_ids"] != 0
        mask = torch.cat(
            (mask.new_full((mask.size(0), 1), True),  # (B, 1)
             mask),  # (B, L)
            dim=1
        )
        x = self.embedding(x)
        x = self.class_token(x)
        for layer in self.layers:
            x = layer(x, mask, time_ids)
        x = self.head(x)
        return x.squeeze()

    def reset_head(self):
        self.head = Head(
            self.dim_token,
            bias=True,
            activation=self.head_activation,
            normalization=self.head_normalization,
            dim_out=self.dim_out,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_token: int,
        num_heads: int,
        att_dropout: float,
        ffn_dropout: float,
        dim_hidden: int,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        activation: Type[nn.Module] = ReGLU,
        skip_attn_norm: bool = False,
        only_class_token: bool = False,
        use_rope: bool = False,
        max_time_id: Optional[int] = 512,
    ):
        super(TransformerBlock, self).__init__()
        if skip_attn_norm:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = norm_layer(dim_token)

        self.query_selector = (
            (lambda x: x[:, :1, :]) if only_class_token else (lambda x: x)
        )
        self.residual_selector = (
            (lambda x: x[:, :1, :]) if only_class_token else (lambda x: x)
        )
        self.mask_selector = (
            (lambda m: m[:, :1]) if only_class_token else (lambda m: m)
        )
        self.time_ids_selector = (
            (lambda t: t[:, :1]) if only_class_token else (lambda t: t)
        )

        self.attn = MultiHeadAttention(
            dim_token=dim_token,
            nheads=num_heads,
            dropout_p=att_dropout,
            use_rope=use_rope,
            max_time_id=max_time_id,
        )
        self.attn_dropout = nn.Dropout(att_dropout)

        self.ffn_norm = norm_layer(dim_token)
        self.ffn = FeedForwardBlock(
            dim_token=dim_token,
            dim_hidden=dim_hidden,
            dropout=ffn_dropout,
            activation=activation,
        )
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.attn_norm(x)
        attn_out = self.attn(
            x=x_norm,
            query_selector=self.query_selector,
            mask=mask,
            time_ids=self.time_ids_selector(time_ids) if time_ids is not None else None,
        )
        attn_out = self.attn_dropout(attn_out)
        x = self.residual_selector(x) + attn_out

        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.ffn_dropout(ffn_out)
        x = self.residual_selector(x) + ffn_out
        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_hidden: int,
        bias_first: bool=True,
        bias_second: bool=True,
        dropout: float=0.0,
        activation: Type[nn.Module]=ReGLU,
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
        dim_token (int): Size of embedding dim for query, key and value
        nheads (int): Number of heads
        dropout_p (float, optional): Dropout probability. Default: 0.0
        use_rope (bool, optional): Whether to use RoPE (rotary positional
            embedding). Default: False
        max_time_id (int, optional): Maximum time_id for RoPE.
            Default: 512
    """

    def __init__(
        self,
        dim_token: int,
        nheads: int,
        dropout_p: float = 0.0,
        use_rope: bool = False,
        max_time_id: Optional[int] = 512,
    ):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.in_proj = nn.Linear(dim_token, 3*dim_token)
        self.out_proj = nn.Linear(dim_token, dim_token)
        assert dim_token % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = dim_token // nheads

        self.use_rope = use_rope
        if self.use_rope and max_time_id is not None:
            self.rope = RotaryEmbedding(
                head_dim=self.E_head, base=10000, max_time_id=max_time_id
            )
        else:
            self.rope = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        query_selector: callable,
        mask: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            x (torch.Tensor): query of shape (N, L_t, dim_token)
            query_selector (callable): function to select query from x
            mask (torch.Tensor): mask of shape (N, L_t)
            time_ids (torch.Tensor, optional): time ids of shape (N, L_t) for RoPE

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        query, key ,value = self.in_proj(x).chunk(3, dim=-1)
        query = query_selector(query)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        if self.use_rope and time_ids is not None:
            query = self.rope(query, time_ids)
            key = self.rope(key, time_ids)
        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_mask = mask[:, None, None, :].contiguous()
        attn_output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout_p, is_causal=False, attn_mask=attn_mask
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
