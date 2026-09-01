# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import math
import csv
import json
from pathlib import Path
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
        numerical_bias: bool = True,
        numerical_bias_scale: float = 1.0,
        numerical_bias_normalization: str = "none",
        wide_enabled: bool = False,
        wide_alpha_init: float = 0.0,
        wide_alpha_trainable: bool = True,
        wide_alpha_mode: str = "trainable",
        freeze_wide: bool = False,
        freeze_deep: bool = False,
        l1_wide_lambda: float = 0.0,
        l_1_wide_lambda: Optional[float] = None,
        wide_init_path: Optional[str] = None,
        wide_init_strict: bool = True,
        model_init_path: Optional[str] = None,
        model_init_strict: bool = True,
        deep_zero_init: bool = False,
        sample_weight_mode: str = "none",
    ):
        super().__init__()
        self.name = model_type
        self.paper_mode = bool(paper_mode)
        self.use_two_logit_ce = self.paper_mode
        self.token_aggregation = str(token_aggregation)
        self.feature_scale_mode = str(feature_scale_mode)
        self.numeric_embedding_mode = str(numeric_embedding_mode)
        self.wide_enabled = bool(wide_enabled)
        if l_1_wide_lambda is not None:
            l1_wide_lambda = l_1_wide_lambda
        self.l1_wide_lambda = float(l1_wide_lambda)
        self.freeze_wide = bool(freeze_wide)
        self.freeze_deep = bool(freeze_deep)
        self.feature_info = feature_info
        self.wide_alpha_mode = str(wide_alpha_mode)
        self.deep_zero_init = bool(deep_zero_init)
        self.sample_weight_mode = str(sample_weight_mode)
        if self.sample_weight_mode not in ("none", "uncertainty"):
            raise ValueError(
                f"Unknown sample_weight_mode: {self.sample_weight_mode}. "
                "Expected one of: none, uncertainty."
            )

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

        if self.numeric_embedding_mode not in ("scale", "concatenate", "pl", "pbld"):
            raise ValueError(
                "numeric_embedding_mode must be one of: scale, concatenate, pl, pbld"
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
            numerical_bias=bool(numerical_bias),
            numerical_bias_scale=float(numerical_bias_scale),
            numerical_bias_normalization=str(numerical_bias_normalization),
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
        if self.deep_zero_init:
            with torch.no_grad():
                self.head.weight.zero_()
                if self.head.bias is not None:
                    self.head.bias.zero_()
        vocabulary_size = int(feature_info.get_vocabulary_size())
        if self.wide_enabled:
            self.wide_embedding = nn.Embedding(
                vocabulary_size + 1,
                1,
                padding_idx=0,
            )
            nn.init.zeros_(self.wide_embedding.weight)
            with torch.no_grad():
                self.wide_embedding.weight[0].zero_()
            setattr(self.wide_embedding.weight, "_is_wide_param", True)
            self.wide_bias = nn.Parameter(torch.zeros(1))
            setattr(self.wide_bias, "_is_wide_param", True)
            alpha_mode = self.wide_alpha_mode.lower()
            if alpha_mode == "fixed1":
                alpha_value = 1.0
                alpha_trainable = False
            elif alpha_mode == "fixed0":
                alpha_value = 0.0
                alpha_trainable = False
            elif alpha_mode == "trainable":
                alpha_value = float(wide_alpha_init)
                alpha_trainable = bool(wide_alpha_trainable)
            else:
                raise ValueError(
                    f"Unknown wide_alpha_mode: {self.wide_alpha_mode}. "
                    "Expected one of: trainable, fixed1, fixed0."
                )
            self.wide_alpha = nn.Parameter(torch.tensor(alpha_value))
            setattr(self.wide_alpha, "_is_wide_param", True)
            if not alpha_trainable:
                self.wide_alpha.requires_grad_(False)
        else:
            self.wide_embedding = None
            self.wide_bias = None
            self.wide_alpha = None

        # cache for schedule updates
        self._sched_drops = [m for m in self.blocks if isinstance(m, ScheduledDropout)]
        self._linear_layers = [m for m in self.blocks if isinstance(m, LinearNTP)]
        self.set_branch_trainability(
            freeze_wide=self.freeze_wide,
            freeze_deep=self.freeze_deep,
        )
        if model_init_path is not None and str(model_init_path).strip() != "":
            self.load_model_from_file(
                path=str(model_init_path),
                strict=bool(model_init_strict),
            )
        if self.wide_enabled and wide_init_path is not None and str(wide_init_path).strip() != "":
            self.load_wide_from_file(str(wide_init_path), strict=bool(wide_init_strict))

    def set_dropout(self, p: float):
        for d in self._sched_drops:
            d.set_p(p)

    def log_training_config(self, estimator):
        print(
            "RealMLP config | "
            f"paper_mode={self.paper_mode} | "
            f"two_logit_ce={self.use_two_logit_ce} | "
            f"token_aggregation={self.token_aggregation} | "
            f"feature_scale_mode={self.feature_scale_mode} | "
            f"numeric_embedding_mode={self.numeric_embedding_mode} | "
            f"wide_enabled={self.wide_enabled} | "
            f"wide_alpha={self.get_wide_alpha():.6f} | "
            f"wide_alpha_mode={self.wide_alpha_mode} | "
            f"freeze_wide={self.freeze_wide} | "
            f"freeze_deep={self.freeze_deep} | "
            f"deep_zero_init={self.deep_zero_init} | "
            f"l1_wide_lambda={self.l1_wide_lambda} | "
            f"sample_weight_mode={self.sample_weight_mode} | "
            f"label_smoothing={estimator.label_smoothing} | "
            f"lr_schedule={estimator.lr_schedule_name} | "
            f"dropout_schedule={estimator.dropout_schedule_name} | "
            f"weight_decay_schedule={estimator.weight_decay_schedule_name} | "
            f"data_dependent_init={estimator.use_data_dependent_init} | "
            f"data_dependent_init_mode={estimator.data_dependent_init_mode} | "
            f"data_dependent_init_bias_mode={estimator.data_dependent_init_bias_mode}"
        )

    def set_branch_trainability(
        self,
        freeze_wide: Optional[bool] = None,
        freeze_deep: Optional[bool] = None,
    ):
        if freeze_wide is not None:
            self.freeze_wide = bool(freeze_wide)
        if freeze_deep is not None:
            self.freeze_deep = bool(freeze_deep)

        if self.freeze_deep:
            deep_modules = (self.embedding, self.feature_scale, self.blocks, self.head)
            for module in deep_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        if self.freeze_wide and self.wide_enabled:
            # Keep alpha trainable for residual-stage runs unless explicitly disabled.
            for parameter in (self.wide_embedding.weight, self.wide_bias):
                parameter.requires_grad_(False)

    def get_wide_alpha(self) -> float:
        if not self.wide_enabled or self.wide_alpha is None:
            return 0.0
        return float(self.wide_alpha.detach().item())

    def regularization_loss(self):
        if (not self.wide_enabled) or self.l1_wide_lambda <= 0.0:
            return None
        # Ignore padding row 0.
        return self.l1_wide_lambda * self.wide_embedding.weight[1:].abs().sum()

    def _feature_map_covariate_to_column(self):
        mapping = {}
        data_reference = getattr(self.feature_info, "data_reference", None)
        if data_reference is None:
            return mapping
        columns = getattr(data_reference, "columns", [])
        if ("covariateId" not in columns) or ("columnId" not in columns):
            return mapping
        covariate_ids = data_reference["covariateId"].to_list()
        column_ids = data_reference["columnId"].to_list()
        for covariate_id, column_id in zip(covariate_ids, column_ids):
            key = self._normalize_covariate_key(covariate_id)
            mapping[key] = int(column_id)
        return mapping

    @staticmethod
    def _normalize_covariate_key(value):
        text = str(value).strip()
        if text == "(Intercept)":
            return text
        try:
            numeric = float(text)
            if math.isfinite(numeric):
                rounded = int(round(numeric))
                if abs(numeric - rounded) < 1e-9:
                    return str(rounded)
        except ValueError:
            pass
        return text

    @staticmethod
    def _pick_first_existing(row, keys):
        for key in keys:
            if key in row and row[key] is not None and str(row[key]).strip() != "":
                return row[key]
        return None

    def _parse_wide_init_json(self, path: Path):
        content = json.loads(path.read_text(encoding="utf-8"))
        coefficients = content.get("coefficients", content)
        if not isinstance(coefficients, dict):
            raise ValueError(f"Expected dict-like coefficients in {path}")
        covariate_ids = coefficients.get("covariateIds")
        betas = coefficients.get("betas")
        if covariate_ids is None or betas is None:
            raise ValueError(f"Expected covariateIds and betas in {path}")
        if len(covariate_ids) != len(betas):
            raise ValueError(f"Mismatched covariateIds/betas length in {path}")
        return covariate_ids, betas

    def _parse_wide_init_csv(self, path: Path):
        covariate_ids = []
        betas = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                covariate_id = self._pick_first_existing(
                    row,
                    ("covariateId", "covariate_id", "columnId", "column_id"),
                )
                beta = self._pick_first_existing(
                    row,
                    ("weight", "coefficient", "beta", "covariateValue"),
                )
                if covariate_id is None or beta is None:
                    continue
                covariate_ids.append(covariate_id)
                betas.append(float(beta))
        return covariate_ids, betas

    def load_wide_from_file(self, path: str, strict: bool = True):
        if not self.wide_enabled:
            raise ValueError("Cannot load wide initialization when wide_enabled=False")
        init_path = Path(path)
        if not init_path.exists():
            raise FileNotFoundError(f"Wide init file not found: {init_path}")

        if init_path.suffix.lower() == ".json":
            covariate_ids, betas = self._parse_wide_init_json(init_path)
        else:
            covariate_ids, betas = self._parse_wide_init_csv(init_path)

        covariate_to_column = self._feature_map_covariate_to_column()
        vocabulary_size = self.wide_embedding.weight.shape[0] - 1
        loaded = 0
        missing = 0
        intercept = None

        with torch.no_grad():
            self.wide_embedding.weight.zero_()
            self.wide_embedding.weight[0].zero_()
            self.wide_bias.zero_()
            for covariate_id, beta in zip(covariate_ids, betas):
                covariate_str = self._normalize_covariate_key(covariate_id)
                if covariate_str == "(Intercept)":
                    intercept = float(beta)
                    continue

                column_id = covariate_to_column.get(covariate_str)
                if column_id is None:
                    try:
                        parsed = int(float(covariate_str))
                    except ValueError:
                        parsed = None
                    if parsed is not None and 1 <= parsed <= vocabulary_size:
                        column_id = parsed

                if column_id is None or not (1 <= int(column_id) <= vocabulary_size):
                    missing += 1
                    continue
                self.wide_embedding.weight[int(column_id), 0] = float(beta)
                loaded += 1

            if intercept is not None:
                self.wide_bias.fill_(float(intercept))

        if strict and loaded == 0:
            raise ValueError(
                f"Failed to map any coefficients from {init_path} to current feature vocabulary."
            )
        print(
            f"Loaded wide init from {init_path} | loaded={loaded} | missing={missing} "
            f"| intercept={'set' if intercept is not None else 'unset'}"
        )

    def load_model_from_file(self, path: str, strict: bool = True):
        init_path = Path(path)
        if not init_path.exists():
            raise FileNotFoundError(f"Model init file not found: {init_path}")

        try:
            payload = torch.load(str(init_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(init_path), map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported model init payload in {init_path}")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            raise ValueError(
                f"Model init strict load failed for {init_path} | "
                f"missing={missing_keys} | unexpected={unexpected_keys}"
            )
        print(
            f"Loaded model init from {init_path} | "
            f"missing={len(missing_keys)} | unexpected={len(unexpected_keys)}"
        )

    def get_optimizer_param_groups(self, estimator_settings):
        scale_params = []
        embedding_params = []
        fm_params = []
        wide_params = []
        act_params = []
        bias_params = []
        other_params = []

        scaling_lr_mult = float(estimator_settings.get("scaling_lr_mult", 1.0))
        bias_lr_mult = float(estimator_settings.get("bias_lr_mult", 1.0))
        act_lr_mult = float(estimator_settings.get("act_lr_mult", 1.0))
        embedding_lr_mult = float(estimator_settings.get("embedding_lr_mult", 1.0))
        fm_lr_mult = float(estimator_settings.get("fm_lr_mult", 1.0))
        fm_wd_factor = float(estimator_settings.get("fm_wd_factor", 1.0))
        wide_lr_mult = float(estimator_settings.get("wide_lr_mult", 1.0))
        wide_wd_factor = float(estimator_settings.get("wide_wd_factor", 0.0))
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
            elif getattr(param, "_is_fm_param", False):
                fm_params.append(param)
            elif getattr(param, "_is_wide_param", False):
                wide_params.append(param)
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
        if fm_params:
            groups.append(
                {
                    "params": fm_params,
                    "name": "fm",
                    "lr_factor": fm_lr_mult,
                    "wd_factor": fm_wd_factor,
                }
            )
        if wide_params:
            groups.append(
                {
                    "params": wide_params,
                    "name": "wide",
                    "lr_factor": wide_lr_mult,
                    "wd_factor": wide_wd_factor,
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

    def collect_batch_diagnostics(self, batch):
        return self.embedding.get_aggregation_diagnostics(
            batch,
            mode=self.token_aggregation,
        )

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

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for layer in self._linear_layers:
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
                    if layer.bias is not None:
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
        if self.token_aggregation == "mean":
            return scaled_tokens.mean(dim=1)
        if self.token_aggregation == "sum":
            return scaled_tokens.sum(dim=1)
        # sum_len_norm / sum_sqrt_len
        lengths = (feature_ids != 0).sum(dim=1, keepdim=True).clamp_min(1)
        lengths = lengths.to(dtype=scaled_tokens.dtype)
        return scaled_tokens.sum(dim=1) / torch.sqrt(lengths)

    def _wide_logit(self, feature_ids, feature_values, dtype):
        if not self.wide_enabled:
            return None
        if feature_values is None:
            feature_values = torch.ones_like(feature_ids, dtype=dtype)
        else:
            feature_values = feature_values.to(dtype=dtype)
        wide_terms = self.wide_embedding(feature_ids).squeeze(-1) * feature_values
        return wide_terms.sum(dim=1) + self.wide_bias

    def _sample_weights(self):
        if self.sample_weight_mode == "none":
            return None
        if self.sample_weight_mode == "uncertainty":
            if self.wide_enabled and hasattr(self, "_last_wide_logit"):
                p = torch.sigmoid(self._last_wide_logit.detach())
                w = 4.0 * p * (1.0 - p)
                return w / w.mean().clamp_min(1e-6)
            return None
        return None

    def _binary_loss(self, predictions, targets, label_smoothing: float = 0.0):
        logits = predictions.squeeze()
        targets = targets.float()
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = self._sample_weights()
        if weights is not None:
            losses = losses * weights
        return losses.sum()

    def _cross_entropy_loss(self, predictions, targets, label_smoothing: float = 0.0):
        losses = F.cross_entropy(
            predictions,
            targets.long().view(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        )
        weights = self._sample_weights()
        if weights is not None:
            losses = losses * weights
        return losses.sum()

    def forward(self, x):
        feature_ids = x["feature_ids"]
        tokens = self.embedding(x)  # [B, L, size_embedding]
        scales = self.feature_scale(feature_ids)  # [B, L, 1] or [B, L, size_embedding]
        hidden = self._aggregate_tokens(tokens * scales, feature_ids)  # [B, size_embedding]
        for m in self.blocks:
            hidden = m(hidden)
        deep_logits = self.head(hidden)
        if not self.wide_enabled:
            return deep_logits.squeeze(-1)

        deep_logit = self.prediction_scores(deep_logits)
        wide_logit = self._wide_logit(
            feature_ids=feature_ids,
            feature_values=x.get("feature_values", None),
            dtype=deep_logit.dtype,
        )
        self._last_wide_logit = wide_logit
        combined = wide_logit + self.wide_alpha * deep_logit
        if self.use_two_logit_ce:
            # Keep a two-logit output for CE while preserving the intended margin.
            return torch.stack((-0.5 * combined, 0.5 * combined), dim=1)
        return combined
