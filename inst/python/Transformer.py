from typing import Optional, Type, Callable

import torch
import torch.nn.functional as F
from torch import nn

from Embeddings import ClassToken, Embedding
from PositionalEncodings import EfficientRPE
from Dataset import FeatureInfo
from PositionalEncodings import PositionalEncoding, NoPositionalEncoding


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
        pe_module: PositionalEncoding = NoPositionalEncoding(),
        model_type="Transformer",
        **kwargs,
    ):
        super(Transformer, self).__init__()
        self.name = model_type
        num_blocks = int(num_blocks)
        dim_token = int(dim_token)
        num_heads = int(num_heads)
        dim_hidden = int(dim_hidden)
        dim_out = int(dim_out)

        self.embedding = Embedding(embedding_dim=dim_token, feature_info=feature_info)
        self.pe_module = pe_module

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
                pe_module=self.pe_module,
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

    def forward(self, input):
        if "time_ids" in input.keys() and input["time_ids"] is not None:
            time_ids = input["time_ids"]
            sequence_lengths = input["sequence_lengths"]
        else:
            time_ids, sequence_lengths = None, None
        mask = input["feature_ids"] != 0
        mask = torch.cat(
            (
                mask.new_full((mask.size(0), 1), True),  # (B, 1)
                mask,
            ),  # (B, L)
            dim=1,
        )
        x = self.embedding(input)
        x = self.class_token(x)
        x = self.pe_module.apply_additive_pe(
            x, time_ids=time_ids, lengths=sequence_lengths
        )
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
        pe_module: PositionalEncoding = NoPositionalEncoding(),
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
        self.mask_selector = (lambda m: m[:, :1]) if only_class_token else (lambda m: m)

        self.only_class_token = only_class_token
        AttentionModuleClass = pe_module.get_attention_module_class()

        self.attn = AttentionModuleClass(
            dim_token=dim_token,
            nheads=num_heads,
            dropout_p=att_dropout,
            pe_module=pe_module,
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
            time_ids=time_ids if time_ids is not None else None,
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
        bias_first: bool = True,
        bias_second: bool = True,
        dropout: float = 0.0,
        activation: Type[nn.Module] = ReGLU,
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
        pe_module: PositionalEncoding = NoPositionalEncoding(),
    ):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.in_proj = nn.Linear(dim_token, 3 * dim_token)
        self.out_proj = nn.Linear(dim_token, dim_token)
        assert dim_token % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = dim_token // nheads
        self.pe_module = pe_module

    def forward(
        self,
        x: torch.Tensor,
        query_selector: Callable,
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
        query, key, value = self.in_proj(x).chunk(3, dim=-1)
        query = query_selector(query)
        Lk = key.size(1)
        Lq = query.size(1)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        pos_scores, bias = None, None
        use_manual_path = False
        use_erpe = False
        if time_ids is not None:
            query, key = self.pe_module.apply_attention_pe(
                query, key, time_ids=time_ids
            )
            pos_scores = self.pe_module.get_positional_scores(
                query, k_len=Lk, time_ids=time_ids
            )
            bias = self.pe_module.get_attention_bias(
                q_len=Lq, k_len=Lk, time_ids=time_ids
            )
            if pos_scores is not None or bias is not None:
                use_manual_path = True
            use_erpe = isinstance(self.pe_module, EfficientRPE)

        attn_mask = mask[:, None, None, :].contiguous()

        if use_manual_path:
            # TODO : explore more efficient way to handle bias
            # --- Manual Attention Path (for RPE, eRPE) ---
            scale = self.E_head**-0.5
            attn_scores = (query @ key.transpose(-2, -1)) * scale

            if pos_scores is not None:
                # Add positional scores if available
                attn_scores += pos_scores * scale
            if bias is not None:
                attn_scores += bias
            attn_scores = attn_scores.masked_fill(~attn_mask, -torch.inf)

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout_p, training=self.training
            )

            attn_output = attn_weights @ value
        else:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0
                if use_erpe
                else (self.dropout_p if self.training else 0.0),
                is_causal=False,
                attn_mask=attn_mask,
            )
        if use_erpe:
            # eRPE uses post-softmax bias, so we need to apply it after the attention
            post_bias = self.pe_module.get_post_softmax_bias(Lq, Lk, time_ids=time_ids)
            post_bias = (
                post_bias.masked_fill(~attn_mask, 0.0)
                if post_bias is not None
                else None
            )
            if post_bias is not None:
                delta = torch.einsum("bhql,bhle->bhqe", post_bias, value)
                attn_output = attn_output + delta
                if self.training and self.dropout_p > 0.0:
                    attn_output = F.dropout(attn_output, p=self.dropout_p)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
