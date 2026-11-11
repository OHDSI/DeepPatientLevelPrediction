# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from Embeddings import Embedding


class TokenFeatureScale(nn.Module):
    """Learned scalar per covariate id, applied before token aggregation."""

    def __init__(self, vocabulary_size: int, scale_dim: int = 1):
        super().__init__()
        self.scale = nn.Embedding(vocabulary_size + 1, int(scale_dim), padding_idx=0)
        nn.init.ones_(self.scale.weight)
        with torch.no_grad():
            self.scale.weight[0].zero_()
        setattr(self.scale.weight, "_is_scale_param", True)  # tag for optimizer group

    def forward(self, feature_ids):
        return self.scale(feature_ids)


class LinearNTP(nn.Linear):
    """Linear layer with Neural Tangent Parametrization scaling."""

    _ntp_factor: torch.Tensor

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("_ntp_factor", torch.tensor(1.0 / math.sqrt(in_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=1.0)

    def forward(self, input):
        return nn.functional.linear(input, self.weight, self.bias) * self._ntp_factor


class ParametricActivation(nn.Module):
    """σα(x) = (1-α)x + α σ(x), per-neuron α."""

    def __init__(self, hidden: int, base_act: str = "selu"):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden))
        setattr(self.alpha, "_is_act_param", True)  # tag for optimizer group
        self.sigma = nn.SELU() if base_act == "selu" else nn.Mish()

    def forward(self, x):
        a = self.alpha.view(1, -1)
        return (1 - a) * x + a * self.sigma(x)


class ScheduledDropout(nn.Module):
    """Dropout with externally settable p."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def set_p(self, p):
        self.p = float(p)

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training)


class RealMLP(nn.Module):
    """
    Minimal RealMLP for binary classification.
    Assumes preprocessing (robust scaling/clipping) is handled upstream.
    """

    def __init__(
        self,
        feature_info,
        size_embedding: int,
        size_hidden: int,
        num_layers: int,
        dropout: float,
        dim_out: int = 1,
        model_type: str = "RealMLP",
        paper_mode: bool = False,
        token_aggregation: str = "mean",
        feature_scale_mode: str = "scalar",
        numeric_embedding_mode: str = "scale",
        numeric_num_frequencies: int = 8,
        numeric_periodic_init_std: float = 0.1,
        numeric_pbld_hidden_dim: int = 16,
        numeric_pbld_embedding_dim: int = 4,
    ):
        super().__init__()
        self.name = model_type
        self.paper_mode = bool(paper_mode)
        self.use_two_logit_ce = self.paper_mode
        self.token_aggregation = str(token_aggregation)
        self.feature_scale_mode = str(feature_scale_mode)
        self.numeric_embedding_mode = str(numeric_embedding_mode)

        if self.paper_mode and self.token_aggregation == "mean":
            self.token_aggregation = "sum"

        if self.token_aggregation not in ("mean", "sum", "sum_len_norm", "sum_sqrt_len"):
            raise ValueError(
                f"Unknown token_aggregation: {self.token_aggregation}. "
                "Expected one of: mean, sum, sum_len_norm, sum_sqrt_len."
            )
        if self.feature_scale_mode not in ("scalar", "vector"):
            raise ValueError(
                f"Unknown feature_scale_mode: {self.feature_scale_mode}. "
                "Expected one of: scalar, vector."
            )

        if self.numeric_embedding_mode not in ("scale", "pl", "pbld"):
            raise ValueError(
                "numeric_embedding_mode must be one of: scale, pl, pbld"
            )

        # Use existing Embedding so Dataset interface stays unchanged.
        self.embedding = Embedding(
            feature_info=feature_info,
            numeric_mode=self.numeric_embedding_mode,
            numerical_num_frequencies=int(numeric_num_frequencies),
            numerical_periodic_init_std=float(numeric_periodic_init_std),
            numerical_pbld_hidden_dim=int(numeric_pbld_hidden_dim),
            numerical_pbld_embedding_dim=int(numeric_pbld_embedding_dim),
            embedding_dim=int(size_embedding),
            aggregate="none",
        )
        for parameter in self.embedding.parameters():
            setattr(parameter, "_is_embedding_param", True)

        in_dim = int(size_embedding)
        scale_dim = 1 if self.feature_scale_mode == "scalar" else in_dim
        self.feature_scale = TokenFeatureScale(
            vocabulary_size=int(feature_info.get_vocabulary_size()),
            scale_dim=scale_dim,
        )

        blocks = []
        last = in_dim
        for _ in range(int(num_layers)):
            blocks += [
                LinearNTP(last, int(size_hidden), bias=True),
                nn.BatchNorm1d(int(size_hidden)),
                ParametricActivation(int(size_hidden), base_act="selu"),
                ScheduledDropout(p=float(dropout)),
            ]
            last = int(size_hidden)
        self.blocks = nn.ModuleList(blocks)

        self.output_dim = 2 if self.use_two_logit_ce else int(dim_out)
        self.head = nn.Linear(last, self.output_dim)

        # cache for schedule updates
        self._sched_drops = [m for m in self.blocks if isinstance(m, ScheduledDropout)]
        self._linear_layers = [m for m in self.blocks if isinstance(m, LinearNTP)]

    def set_dropout(self, p: float):
        for d in self._sched_drops:
            d.set_p(p)

    def reset_head(self):
        self.head = nn.Linear(self.head.in_features, self.head.out_features)

    def log_training_config(self, estimator):
        print(
            "RealMLP config | "
            f"paper_mode={self.paper_mode} | "
            f"two_logit_ce={self.use_two_logit_ce} | "
            f"token_aggregation={self.token_aggregation} | "
            f"feature_scale_mode={self.feature_scale_mode} | "
            f"numeric_embedding_mode={self.numeric_embedding_mode} | "
            f"label_smoothing={estimator.label_smoothing} | "
            f"lr_schedule={estimator.lr_schedule_name} | "
            f"dropout_schedule={estimator.dropout_schedule_name} | "
            f"weight_decay_schedule={estimator.weight_decay_schedule_name} | "
            f"data_dependent_init={estimator.use_data_dependent_init} | "
            f"data_dependent_init_mode={estimator.data_dependent_init_mode} | "
            f"data_dependent_init_bias_mode={estimator.data_dependent_init_bias_mode}"
        )

    def get_optimizer_param_groups(self, estimator_settings):
        scale_params = []
        embedding_params = []
        act_params = []
        bias_params = []
        other_params = []

        scaling_lr_mult = float(estimator_settings.get("scaling_lr_mult", 1.0))
        bias_lr_mult = float(estimator_settings.get("bias_lr_mult", 1.0))
        act_lr_mult = float(estimator_settings.get("act_lr_mult", 1.0))
        embedding_lr_mult = float(estimator_settings.get("embedding_lr_mult", 1.0))
        bias_wd_factor = float(estimator_settings.get("bias_wd_factor", 0.0))

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if getattr(param, "_is_scale_param", False):
                scale_params.append(param)
            elif getattr(param, "_is_act_param", False):
                act_params.append(param)
            elif getattr(param, "_is_embedding_param", False):
                embedding_params.append(param)
            elif name.endswith(".bias"):
                bias_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append(
                {"params": other_params, "name": "other", "lr_factor": 1.0, "wd_factor": 1.0}
            )
        if scale_params:
            groups.append(
                {
                    "params": scale_params,
                    "name": "scale",
                    "lr_factor": scaling_lr_mult,
                    "wd_factor": 1.0,
                }
            )
        if embedding_params:
            groups.append(
                {
                    "params": embedding_params,
                    "name": "embed",
                    "lr_factor": embedding_lr_mult,
                    "wd_factor": 1.0,
                }
            )
        if bias_params:
            groups.append(
                {
                    "params": bias_params,
                    "name": "bias",
                    "lr_factor": bias_lr_mult,
                    "wd_factor": bias_wd_factor,
                }
            )
        if act_params:
            groups.append(
                {"params": act_params, "name": "act", "lr_factor": act_lr_mult, "wd_factor": 1.0}
            )
        return groups

    def apply_dynamic_schedule(
        self,
        optimizer,
        base_learning_rate: float,
        base_weight_decay: float,
        lr_scale: float,
        wd_scale: float,
        drop_scale: float,
        base_dropout: float,
    ):
        for group in optimizer.param_groups:
            group["lr"] = base_learning_rate * group.get("lr_factor", 1.0) * lr_scale
            group["weight_decay"] = (
                base_weight_decay * group.get("wd_factor", 1.0) * wd_scale
            )
        self.set_dropout(base_dropout * drop_scale)

    def compute_loss(self, predictions, targets, criterion, label_smoothing: float = 0.0):
        if self.use_two_logit_ce:
            return self._cross_entropy_loss(
                predictions=predictions,
                targets=targets,
                label_smoothing=label_smoothing,
            )
        return self._binary_loss(
            predictions=predictions,
            targets=targets,
            label_smoothing=label_smoothing,
        )

    def prediction_scores(self, predictions):
        if self.use_two_logit_ce:
            return predictions[:, 1] - predictions[:, 0]
        return predictions.squeeze()

    def predict_proba_from_output(self, predictions):
        if self.use_two_logit_ce:
            return F.softmax(predictions, dim=1)[:, 1]
        return torch.sigmoid(predictions.squeeze())

    @staticmethod
    def _resolve_bias_target(
        in_features: int,
        bias_mode: Optional[str],
        bias_scale: float,
    ):
        mode = "none" if bias_mode is None else str(bias_mode).lower()
        if mode in ("none", "off"):
            return None
        if mode in ("zero", "center"):
            return 0.0
        if mode in ("he5", "he+5"):
            return float(bias_scale) * (5.0 / math.sqrt(float(in_features)))
        raise ValueError(f"Unknown data-dependent init bias mode: {bias_mode}")

    def _collect_layer_outputs(self, layer, batches):
        layer_outputs = []

        def _capture_output(_, __, output):
            layer_outputs.append(output.detach())

        hook = layer.register_forward_hook(_capture_output)
        try:
            for features in batches:
                self(features)
        finally:
            hook.remove()
        return layer_outputs

    @staticmethod
    def _slice_batch_rows(features, keep_rows: int):
        sliced = {}
        for key, value in features.items():
            if torch.is_tensor(value) and value.shape[0] >= keep_rows:
                sliced[key] = value[:keep_rows]
            else:
                sliced[key] = value
        return sliced

    def _prepare_ddi_batches(self, batches, max_rows: int):
        prepared = []
        if max_rows is None or int(max_rows) <= 0:
            return list(batches)
        remaining = int(max_rows)
        for features in batches:
            if remaining <= 0:
                break
            rows = int(features["feature_ids"].shape[0])
            if rows <= remaining:
                prepared.append(features)
                remaining -= rows
                continue
            prepared.append(self._slice_batch_rows(features, remaining))
            remaining = 0
            break
        return prepared

    def data_dependent_init(
        self,
        batches,
        eps: float = 1e-6,
        init_mode: str = "current",
        target_var: float = 1.0,
        max_rows: int = 0,
        gain_clip: float = 0.0,
        bias_mode: Optional[str] = "he5",
        bias_scale: float = 1.0,
        bias_refit_steps: int = 2,
        verbose: bool = False,
    ):
        """
        Sequentially rescale each hidden LinearNTP row so pre-activation
        variance is ~1 on a small sample of training batches and optionally
        fit layer biases to a target mean ("he5" or "zero").
        """
        if not self._linear_layers:
            return
        mode = str(init_mode).lower()
        if mode not in ("current", "paper_lsuv"):
            raise ValueError(
                f"Unknown data-dependent init mode: {init_mode}. "
                "Expected one of: current, paper_lsuv."
            )
        batches = self._prepare_ddi_batches(list(batches), max_rows=max_rows)
        if len(batches) == 0:
            return

        # frozen weights, such as those of a fine-tuned model, must stay unchanged
        trainable_layers = [
            layer for layer in self._linear_layers if layer.weight.requires_grad
        ]
        if len(trainable_layers) == 0:
            return

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for layer in trainable_layers:
                    layer_outputs = self._collect_layer_outputs(layer, batches)

                    if len(layer_outputs) == 0:
                        continue
                    merged = torch.cat(layer_outputs, dim=0)
                    variances = merged.var(dim=0, unbiased=False).clamp_min(eps)
                    gains = torch.sqrt(float(target_var) / variances)
                    gain_clip = float(gain_clip)
                    if gain_clip > 1.0:
                        gains = gains.clamp(min=1.0 / gain_clip, max=gain_clip)
                    layer.weight.mul_(gains.view(-1, 1))
                    if bool(verbose):
                        post_outputs = self._collect_layer_outputs(layer, batches)
                        post_var = torch.cat(post_outputs, dim=0).var(dim=0, unbiased=False)
                        print(
                            "DDI layer stats | "
                            f"mode={mode} | "
                            f"layer_in={layer.in_features} | "
                            f"layer_out={layer.out_features} | "
                            f"pre_var_mean={variances.mean().item():.6f} | "
                            f"post_var_mean={post_var.mean().item():.6f}"
                        )
                    if layer.bias is not None and layer.bias.requires_grad:
                        target_value = self._resolve_bias_target(
                            in_features=layer.in_features,
                            bias_mode=bias_mode,
                            bias_scale=bias_scale,
                        )
                        if target_value is None:
                            continue
                        target = None
                        for _ in range(max(1, int(bias_refit_steps))):
                            centered_outputs = self._collect_layer_outputs(layer, batches)
                            if len(centered_outputs) == 0:
                                break
                            centered = torch.cat(centered_outputs, dim=0)
                            means = centered.mean(dim=0)
                            if target is None:
                                target = torch.full_like(means, float(target_value))
                            output_scale = 1.0
                            if hasattr(layer, "_ntp_factor"):
                                output_scale = float(layer._ntp_factor.item())
                                output_scale = max(output_scale, eps)
                            layer.bias.add_((target - means) / output_scale)
        finally:
            if was_training:
                self.train()

    def _aggregate_tokens(self, scaled_tokens, feature_ids):
        if self.token_aggregation == "sum":
            return scaled_tokens.sum(dim=1)
        lengths = (feature_ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
        lengths = lengths.to(dtype=scaled_tokens.dtype)
        if self.token_aggregation == "mean":
            return scaled_tokens.sum(dim=1) / lengths
        # sum_len_norm / sum_sqrt_len
        return scaled_tokens.sum(dim=1) / torch.sqrt(lengths)

    def _binary_loss(self, predictions, targets, label_smoothing: float = 0.0):
        logits = predictions.squeeze()
        targets = targets.float()
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return losses.sum()

    def _cross_entropy_loss(self, predictions, targets, label_smoothing: float = 0.0):
        losses = F.cross_entropy(
            predictions,
            targets.long().view(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        )
        return losses.sum()

    def forward(self, x):
        feature_ids = x["feature_ids"]
        tokens = self.embedding(x)  # [B, L, size_embedding]
        scales = self.feature_scale(feature_ids)  # [B, L, 1] or [B, L, size_embedding]
        hidden = self._aggregate_tokens(tokens * scales, feature_ids)  # [B, size_embedding]
        for m in self.blocks:
            hidden = m(hidden)
        return self.head(hidden).squeeze(-1)

