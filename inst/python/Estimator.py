import time
import pathlib
import inspect
import hashlib
import os
import random
import contextlib
import csv

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from gpu_memory_cleanup import memory_cleanup
from InitStrategy import InitStrategy, DefaultInitStrategy
from schedules import get_schedule
try:
    import numpy as np
except ImportError:
    np = None

class Estimator:
    """
    A class that wraps around pytorch models.
    """

    def __init__(self, model, parameters):
        raw_seed = parameters["estimator_settings"].get("seed", 0)
        if raw_seed is None:
            raw_seed = 0
        self.seed = int(raw_seed)
        if callable(parameters["estimator_settings"]["device"]):
            self.device = parameters["estimator_settings"]["device"]()
        else:
            self.device = parameters["estimator_settings"]["device"]
        torch.manual_seed(seed=self.seed)
        random.seed(self.seed)
        if np is not None:
            np.random.seed(self.seed)

        self.deterministic = bool(parameters["estimator_settings"].get("deterministic", False))
        self.deterministic_warn_only = bool(
            parameters["estimator_settings"].get("deterministic_warn_only", True)
        )
        self.cublas_workspace_config = parameters["estimator_settings"].get(
            "cublas_workspace_config"
        )
        if self.cublas_workspace_config:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(self.cublas_workspace_config)
        self._configure_determinism()

        if "init_strategy" in parameters["estimator_settings"]:
            self.model = parameters["estimator_settings"]["init_strategy"].initialize(model, parameters)
        else:
            self.model = DefaultInitStrategy().initialize(model, parameters)
            
        self.model_parameters = parameters["model_parameters"]
        self.estimator_settings = parameters["estimator_settings"]

        self.epochs = int(parameters["estimator_settings"].get("epochs", 5))
        if parameters["estimator_settings"]["find_l_r"]:
            self.learning_rate = 3e-4
        else:
            self.learning_rate = parameters["estimator_settings"].get("learning_rate", 3e-4)
        self.weight_decay = parameters["estimator_settings"].get("weight_decay", 1e-5)
        self.batch_size = int(parameters["estimator_settings"].get("batch_size", 1024))
        self.num_workers = int(parameters["estimator_settings"].get("num_workers", 0))
        self.persistent_workers = bool(
            parameters["estimator_settings"].get(
                "persistent_workers",
                self.num_workers > 0,
            )
        )
        if self.num_workers <= 0:
            self.num_workers = 0
            self.persistent_workers = False
        self.dataloader_seed = int(
            parameters["estimator_settings"].get("dataloader_seed", self.seed)
        )
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.dataloader_seed)
        self.prefix = parameters["estimator_settings"].get("prefix", self.model.name)
        self.base_learning_rate = float(self.learning_rate)
        self.base_weight_decay = float(self.weight_decay)
        grad_clip_norm = parameters["estimator_settings"].get("grad_clip_norm")
        self.grad_clip_norm = (
            None if grad_clip_norm is None else float(grad_clip_norm)
        )
        self.sam_enabled = bool(parameters["estimator_settings"].get("sam_enabled", False))
        self.sam_rho = float(parameters["estimator_settings"].get("sam_rho", 0.05))
        self.sam_adaptive = bool(parameters["estimator_settings"].get("sam_adaptive", False))
        self.sam_eps = float(parameters["estimator_settings"].get("sam_eps", 1e-12))
        self.ema_enabled = bool(parameters["estimator_settings"].get("ema_enabled", False))
        self.ema_decay = float(parameters["estimator_settings"].get("ema_decay", 0.999))
        self.ema_start_step = int(parameters["estimator_settings"].get("ema_start_step", 0))
        ema_start_fraction = parameters["estimator_settings"].get("ema_start_fraction")
        self.ema_start_fraction = None if ema_start_fraction is None else float(ema_start_fraction)
        self.ema_eval_use = bool(parameters["estimator_settings"].get("ema_eval_use", True))
        self.ema_state = dict()
        self.ema_initialized = False
        self.ema_active = False
        self.ema_first_active_step = None
        self.ema_active_by_epoch = list()
        self.best_epoch_index = None
        self.ema_active_at_best_epoch = False

        if "accumulation_steps" in parameters["estimator_settings"].keys() \
        and parameters["estimator_settings"]["accumulation_steps"]:
            self.accumulation_steps = int(parameters["estimator_settings"]["accumulation_steps"])
            self.sub_batch_size = self.batch_size // self.accumulation_steps
        else:
            self.accumulation_steps = 1
            self.sub_batch_size = self.batch_size

        self.previous_epochs = int(parameters["estimator_settings"].get("previous_epochs", 0))
        self.model.to(device=self.device)

        self.has_custom_param_groups = hasattr(self.model, "get_optimizer_param_groups")
        self.has_custom_loss = hasattr(self.model, "compute_loss")
        self.has_custom_scores = hasattr(self.model, "prediction_scores")
        self.has_custom_proba = hasattr(self.model, "predict_proba_from_output")
        self.has_custom_regularization = hasattr(self.model, "regularization_loss")
        self.has_custom_schedule = hasattr(self.model, "apply_dynamic_schedule")
        self.has_data_dependent_init = hasattr(self.model, "data_dependent_init")
        self.has_batch_diagnostics = hasattr(self.model, "collect_batch_diagnostics")

        self.beta2 = float(parameters["estimator_settings"].get("beta2", 0.999))
        self.eps = float(parameters["estimator_settings"].get("eps", 1e-8))
        self.label_smoothing = float(parameters["estimator_settings"].get("label_smoothing", 0.0))
        self.base_dropout = float(parameters["model_parameters"].get("dropout", 0.0))
        self.lr_schedule_name = parameters["estimator_settings"].get("lr_schedule")
        self.dropout_schedule_name = parameters["estimator_settings"].get("dropout_schedule")
        self.weight_decay_schedule_name = parameters["estimator_settings"].get("weight_decay_schedule")
        self.lr_schedule = get_schedule(self.lr_schedule_name)
        self.dropout_schedule = get_schedule(self.dropout_schedule_name)
        self.weight_decay_schedule = get_schedule(self.weight_decay_schedule_name)
        self.total_steps = 1
        self.global_step = 0
        self.optimizer_step_count = 0
        self.apply_dynamic_schedule = any(
            x is not None
            for x in (
                self.lr_schedule_name,
                self.dropout_schedule_name,
                self.weight_decay_schedule_name,
            )
        )
        self.use_data_dependent_init = bool(
            parameters["estimator_settings"].get("data_dependent_init", self.has_data_dependent_init)
        )
        self.data_dependent_init_batches = int(
            parameters["estimator_settings"].get("data_dependent_init_batches", 8)
        )
        self.data_dependent_init_bias_mode = parameters["estimator_settings"].get(
            "data_dependent_init_bias_mode",
            "he5",
        )
        self.data_dependent_init_bias_scale = float(
            parameters["estimator_settings"].get("data_dependent_init_bias_scale", 1.0)
        )
        self.data_dependent_init_mode = parameters["estimator_settings"].get(
            "data_dependent_init_mode",
            "current",
        )
        self.data_dependent_init_target_var = float(
            parameters["estimator_settings"].get("data_dependent_init_target_var", 1.0)
        )
        self.data_dependent_init_max_rows = int(
            parameters["estimator_settings"].get("data_dependent_init_max_rows", 0)
        )
        self.data_dependent_init_gain_clip = float(
            parameters["estimator_settings"].get("data_dependent_init_gain_clip", 0.0)
        )
        self.data_dependent_init_bias_refit_steps = int(
            parameters["estimator_settings"].get("data_dependent_init_bias_refit_steps", 2)
        )
        self.data_dependent_init_done = False
        self.enable_batch_diagnostics = bool(
            parameters["estimator_settings"].get("enable_batch_diagnostics", False)
        )
        self.batch_diagnostics_steps = int(
            parameters["estimator_settings"].get("batch_diagnostics_steps", 0)
        )
        if self.enable_batch_diagnostics and self.batch_diagnostics_steps <= 0:
            self.batch_diagnostics_steps = 50
        self.batch_diagnostics_every = int(
            parameters["estimator_settings"].get("batch_diagnostics_every", 10)
        )
        self.batch_diagnostics_counter = 0
        self.batch_diagnostics_totals = dict()

        self.input_fingerprint_enabled = bool(
            parameters["estimator_settings"].get("input_fingerprint_enabled", False)
        )
        self.input_fingerprint_num_rows = int(
            parameters["estimator_settings"].get("input_fingerprint_num_rows", 128)
        )
        self.input_fingerprint_output = parameters["estimator_settings"].get(
            "input_fingerprint_output"
        )
        self.input_fingerprint_logged = False
        self.distill_enabled = bool(
            parameters["estimator_settings"].get("distill_enabled", False)
        )
        self.distill_lambda = float(
            parameters["estimator_settings"].get("distill_lambda", 1.0)
        )
        self.distill_loss = str(
            parameters["estimator_settings"].get("distill_loss", "mse")
        ).lower()
        self.distill_huber_delta = float(
            parameters["estimator_settings"].get("distill_huber_delta", 1.0)
        )
        self.distill_weight_mode = str(
            parameters["estimator_settings"].get("distill_weight_mode", "none")
        ).lower()
        self.distill_teacher_path = parameters["estimator_settings"].get("distill_teacher_path")
        self.distill_teacher_rowid_zero_indexed = bool(
            parameters["estimator_settings"].get(
                "distill_teacher_rowid_zero_indexed", True
            )
        )
        self.teacher_residual_map = None
        self.teacher_residual_max_row_id = -1
        self.teacher_abs_gap_map = None
        self.teacher_residual_device_cache = dict()
        self.teacher_abs_gap_device_cache = dict()
        self._load_teacher_residuals_if_available()
        self._log_model_training_configuration()

        self.metric = None
        if (
            "metric" in parameters["estimator_settings"].keys()
            and parameters["estimator_settings"]["metric"] is not None
        ):
            self.metric = parameters["estimator_settings"]["metric"]
            if isinstance(self.metric, str):
                if self.metric == "auc":
                    self.metric = {"name": "auc", "mode": "max"}
                elif self.metric == "loss":
                    self.metric = {"name": "loss", "mode": "min"}
            if (
                "scheduler" in parameters["estimator_settings"].keys()
                and parameters["estimator_settings"]["scheduler"] is not None
            ):
                parameters["estimator_settings"]["scheduler"]["params"]["mode"] = self.metric["mode"]
            if (
                "early_stopping" in parameters["estimator_settings"].keys()
                and parameters["estimator_settings"]["early_stopping"] is not None
            ):
                parameters["estimator_settings"]["early_stopping"]["params"]["mode"] = self.metric[
                    "mode"
                ]

        self.optimizer = self._create_optimizer(parameters["estimator_settings"]["optimizer"])
        self.criterion = parameters["estimator_settings"]["criterion"](reduction="sum")

        self.use_external_lr_scheduler = (
            "scheduler" in parameters["estimator_settings"].keys()
            and parameters["estimator_settings"]["scheduler"] is not None
            and not (self.apply_dynamic_schedule and self.lr_schedule_name is not None)
        )
        if self.use_external_lr_scheduler:
            self.scheduler = parameters["estimator_settings"]["scheduler"]["fun"](
                self.optimizer, **parameters["estimator_settings"]["scheduler"]["params"]
            )
        else:
            self.scheduler = None

        if (
            "early_stopping" in parameters["estimator_settings"].keys()
            and parameters["estimator_settings"]["early_stopping"] is not None
        ):
            self.early_stopper = EarlyStopping(
                **parameters["estimator_settings"]["early_stopping"]["params"]
            )
        else:
            self.early_stopper = None

        self.best_score = None
        self.best_epoch = None
        self.learn_rate_schedule = None
        torch_compile = parameters["estimator_settings"].get("compile", False)
        if torch_compile:
            self.model = torch.compile(self.model, dynamic=False)

    def _configure_determinism(self):
        if not self.deterministic:
            return
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=self.deterministic_warn_only,
            )
        except TypeError:
            torch.use_deterministic_algorithms(True)

    def _log_model_training_configuration(self):
        if hasattr(self.model, "log_training_config"):
            self.model.log_training_config(self)

    def _seed_worker(self, worker_id):
        worker_seed = self.dataloader_seed + int(worker_id)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed % (2**32 - 1))

    def _maybe_log_input_fingerprint(self, dataset, stage: str = "fit"):
        if not self.input_fingerprint_enabled or self.input_fingerprint_logged:
            return
        if not hasattr(dataset, "data"):
            return
        required_keys = ("row_ids", "feature_ids", "feature_values")
        if not all(key in dataset.data for key in required_keys):
            return
        rows = min(len(dataset), max(1, self.input_fingerprint_num_rows))
        hasher = hashlib.sha256()
        for key in required_keys:
            tensor = dataset.data[key][:rows].detach().cpu().contiguous()
            hasher.update(tensor.numpy().tobytes())
        digest = hasher.hexdigest()
        message = (
            f"Input fingerprint ({stage}) | seed={self.seed} | rows={rows} | sha256={digest}"
        )
        print(message)
        if self.input_fingerprint_output:
            output_path = pathlib.Path(self.input_fingerprint_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        self.input_fingerprint_logged = True

    def _log_batch_diagnostics(self, features):
        if (
            not self.enable_batch_diagnostics
            or not self.has_batch_diagnostics
            or self.batch_diagnostics_counter >= self.batch_diagnostics_steps
        ):
            return
        with torch.no_grad():
            diagnostics = self.model.collect_batch_diagnostics(features)
        if diagnostics is None:
            return
        for key, value in diagnostics.items():
            self.batch_diagnostics_totals[key] = (
                self.batch_diagnostics_totals.get(key, 0.0) + float(value)
            )
        self.batch_diagnostics_counter += 1
        if (
            self.batch_diagnostics_counter % max(1, self.batch_diagnostics_every) == 0
            or self.batch_diagnostics_counter == self.batch_diagnostics_steps
        ):
            averaged = {
                key: value / self.batch_diagnostics_counter
                for key, value in self.batch_diagnostics_totals.items()
            }
            stats = " | ".join(
                f"{key}={value:.4f}" for key, value in sorted(averaged.items())
            )
            print(
                f"Batch diagnostics ({self.batch_diagnostics_counter}/"
                f"{self.batch_diagnostics_steps}) | {stats}"
            )

    def _create_optimizer(self, optimizer_class):
        params = self.model.parameters()
        if self.has_custom_param_groups:
            param_groups = self.model.get_optimizer_param_groups(self.estimator_settings)
            if param_groups is not None and len(param_groups) > 0:
                params = param_groups
        if isinstance(params, list):
            for group in params:
                group.setdefault(
                    "lr",
                    self.learning_rate * group.get("lr_factor", 1.0),
                )
                group.setdefault(
                    "weight_decay",
                    self.weight_decay * group.get("wd_factor", 1.0),
                )

        kwargs = {
            "params": params,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if "beta2" in self.estimator_settings:
            kwargs["betas"] = (0.9, self.beta2)
        if "eps" in self.estimator_settings:
            kwargs["eps"] = self.eps
        try:
            return optimizer_class(**kwargs)
        except TypeError:
            kwargs.pop("betas", None)
            kwargs.pop("eps", None)
            return optimizer_class(**kwargs)

    def _default_prepare_targets_for_loss(self, targets):
        targets = targets.float()
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return targets

    def _compute_loss(self, predictions, targets, features=None, training=False):
        if self.has_custom_loss:
            base_loss = self.model.compute_loss(
                predictions=predictions,
                targets=targets,
                criterion=self.criterion,
                label_smoothing=self.label_smoothing,
            )
        else:
            prepared_targets = self._default_prepare_targets_for_loss(targets)
            base_loss = self.criterion(predictions.squeeze(), prepared_targets)

        if training:
            distill = self._distillation_loss(predictions=predictions, features=features)
            if distill is not None:
                base_loss = base_loss + self.distill_lambda * distill
        return base_loss

    def _load_teacher_residuals_if_available(self):
        if not self.distill_enabled:
            return
        if self.distill_teacher_path is None or str(self.distill_teacher_path).strip() == "":
            print("Distillation enabled but distill_teacher_path missing; disabling distillation.")
            self.distill_enabled = False
            return
        path = pathlib.Path(str(self.distill_teacher_path))
        if not path.exists():
            print(f"Distillation teacher file not found: {path}; disabling distillation.")
            self.distill_enabled = False
            return
        row_to_values = dict()
        duplicate_rows_collapsed = 0
        has_abs_gap = False

        def _evaluation_priority(value):
            if value is None:
                return 3
            text = str(value).strip().lower()
            if text == "train":
                return 0
            if text == "cv":
                return 1
            if text == "test":
                return 2
            return 3

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            field_map = {x.lower(): x for x in (reader.fieldnames or [])}
            row_key = field_map.get("rowid")
            residual_key = (
                field_map.get("teacherresidual")
                or field_map.get("residual")
                or field_map.get("r_teacher")
            )
            abs_gap_key = field_map.get("absprobgap")
            evaluation_type_key = field_map.get("evaluationtype")
            if row_key is None or residual_key is None:
                raise ValueError(
                    "Distillation file must include columns rowId and teacherResidual."
                )
            has_abs_gap = abs_gap_key is not None
            for row in reader:
                row_id = int(float(row[row_key]))
                if not self.distill_teacher_rowid_zero_indexed:
                    row_id -= 1
                residual_value = float(row[residual_key])
                abs_gap_value = (
                    float(row[abs_gap_key]) if has_abs_gap and row.get(abs_gap_key) not in (None, "") else float("nan")
                )
                priority = _evaluation_priority(
                    row.get(evaluation_type_key) if evaluation_type_key is not None else None
                )
                existing = row_to_values.get(row_id)
                if existing is None:
                    row_to_values[row_id] = (priority, residual_value, abs_gap_value)
                else:
                    duplicate_rows_collapsed += 1
                    if priority < existing[0]:
                        row_to_values[row_id] = (priority, residual_value, abs_gap_value)
        if len(row_to_values) == 0:
            print(f"Distillation teacher file is empty: {path}; disabling distillation.")
            self.distill_enabled = False
            return
        row_ids = sorted(row_to_values.keys())
        residuals = [row_to_values[row_id][1] for row_id in row_ids]
        abs_prob_gaps = [row_to_values[row_id][2] for row_id in row_ids]
        max_row_id = max(row_ids)
        teacher_map = torch.full((max_row_id + 1,), float("nan"), dtype=torch.float32)
        teacher_map[torch.as_tensor(row_ids, dtype=torch.long)] = torch.as_tensor(
            residuals, dtype=torch.float32
        )
        self.teacher_residual_map = teacher_map
        self.teacher_residual_max_row_id = max_row_id
        if has_abs_gap and len(abs_prob_gaps) == len(row_ids):
            abs_gap_map = torch.full((max_row_id + 1,), float("nan"), dtype=torch.float32)
            abs_gap_map[torch.as_tensor(row_ids, dtype=torch.long)] = torch.as_tensor(
                abs_prob_gaps, dtype=torch.float32
            )
            finite = torch.isfinite(abs_gap_map)
            if torch.any(finite):
                mean_gap = torch.clamp(abs_gap_map[finite].mean(), min=1e-12)
                abs_gap_map[finite] = abs_gap_map[finite] / mean_gap
            self.teacher_abs_gap_map = abs_gap_map
        elif self.distill_weight_mode != "none":
            print(
                "Distillation weight mode requested, but absProbGap column missing; "
                "falling back to unweighted distillation."
            )
            self.distill_weight_mode = "none"
        if duplicate_rows_collapsed > 0:
            print(
                f"Collapsed duplicate distillation rows: {duplicate_rows_collapsed} "
                f"(kept best-priority evaluation type per rowId)"
            )
        print(
            f"Loaded distillation targets from {path} | rows={len(row_ids)} | "
            f"max_row_id={self.teacher_residual_max_row_id}"
        )

    def _get_teacher_residual_map_on_device(self, device):
        if self.teacher_residual_map is None:
            return None
        cache_key = str(device)
        cached = self.teacher_residual_device_cache.get(cache_key)
        if cached is None:
            cached = self.teacher_residual_map.to(device=device, non_blocking=True)
            self.teacher_residual_device_cache[cache_key] = cached
        return cached

    def _lookup_teacher_residual(self, row_ids):
        if self.teacher_residual_map is None:
            return None, None
        if row_ids is None:
            return None, None
        row_ids = row_ids.reshape(-1).long()
        device_map = self._get_teacher_residual_map_on_device(row_ids.device)
        if device_map is None:
            return None, None
        valid = (row_ids >= 0) & (row_ids <= self.teacher_residual_max_row_id)
        if not torch.any(valid):
            return None, None
        teacher = torch.full_like(row_ids, float("nan"), dtype=torch.float32)
        teacher[valid] = device_map[row_ids[valid]]
        finite = torch.isfinite(teacher)
        if not torch.any(finite):
            return None, None
        return teacher, finite

    def _get_teacher_abs_gap_map_on_device(self, device):
        if self.teacher_abs_gap_map is None:
            return None
        cache_key = str(device)
        cached = self.teacher_abs_gap_device_cache.get(cache_key)
        if cached is None:
            cached = self.teacher_abs_gap_map.to(device=device, non_blocking=True)
            self.teacher_abs_gap_device_cache[cache_key] = cached
        return cached

    def _lookup_teacher_abs_gap(self, row_ids):
        if self.teacher_abs_gap_map is None:
            return None
        if row_ids is None:
            return None
        row_ids = row_ids.reshape(-1).long()
        device_map = self._get_teacher_abs_gap_map_on_device(row_ids.device)
        if device_map is None:
            return None
        valid = (row_ids >= 0) & (row_ids <= self.teacher_residual_max_row_id)
        if not torch.any(valid):
            return None
        values = torch.full_like(row_ids, float("nan"), dtype=torch.float32)
        values[valid] = device_map[row_ids[valid]]
        return values

    def _predicted_residual_logit(self, predictions):
        if hasattr(self.model, "_last_deep_total_logit"):
            deep_total = self.model._last_deep_total_logit
            if deep_total is not None:
                return deep_total.reshape(-1).float()
        if hasattr(self.model, "_last_wide_logit"):
            wide = self.model._last_wide_logit
            if wide is not None:
                total = self._prediction_scores(predictions).reshape(-1)
                return (total - wide.reshape(-1)).float()
        return None

    def _distillation_loss(self, predictions, features):
        if not self.distill_enabled or self.teacher_residual_map is None:
            return None
        row_ids = None if features is None else features.get("row_ids")
        teacher, mask = self._lookup_teacher_residual(row_ids=row_ids)
        if teacher is None or mask is None:
            return None
        pred_residual = self._predicted_residual_logit(predictions)
        if pred_residual is None:
            return None
        pred_residual = pred_residual[mask]
        teacher = teacher[mask]
        row_ids_masked = row_ids.reshape(-1).long()[mask]
        weights = None
        if self.distill_weight_mode == "abs_gap":
            abs_gap = self._lookup_teacher_abs_gap(row_ids=row_ids_masked)
            if abs_gap is not None:
                finite_w = torch.isfinite(abs_gap)
                if torch.any(finite_w):
                    weights = torch.ones_like(abs_gap)
                    weights[finite_w] = torch.clamp(abs_gap[finite_w], min=0.0)
        if self.distill_loss == "huber":
            per_sample = torch.nn.functional.smooth_l1_loss(
                pred_residual,
                teacher,
                beta=self.distill_huber_delta,
                reduction="none",
            )
        else:
            per_sample = torch.nn.functional.mse_loss(
                pred_residual,
                teacher,
                reduction="none",
            )
        if weights is not None:
            denom = torch.clamp(weights.sum(), min=1e-12)
            return (per_sample * weights).sum() / denom
        return per_sample.mean()

    def _regularization_loss(self):
        if not self.has_custom_regularization:
            return None
        value = self.model.regularization_loss()
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(float(value), device=self.device)

    def _prediction_scores(self, predictions):
        if self.has_custom_scores:
            scores = self.model.prediction_scores(predictions)
        else:
            scores = predictions.squeeze()
        return scores.reshape(-1)

    def _prediction_proba(self, predictions):
        if self.has_custom_proba:
            proba = self.model.predict_proba_from_output(predictions)
        else:
            proba = torch.sigmoid(predictions.squeeze())
        return proba.reshape(-1)

    def _apply_step_hparams(self):
        if not self.apply_dynamic_schedule:
            return
        t = self.global_step / max(self.total_steps - 1, 1)
        lr_scale = self.lr_schedule(t)
        wd_scale = self.weight_decay_schedule(t)
        drop_scale = self.dropout_schedule(t)
        if self.has_custom_schedule:
            self.model.apply_dynamic_schedule(
                optimizer=self.optimizer,
                base_learning_rate=self.base_learning_rate,
                base_weight_decay=self.base_weight_decay,
                lr_scale=lr_scale,
                wd_scale=wd_scale,
                drop_scale=drop_scale,
                base_dropout=self.base_dropout,
            )
            return

        for group in self.optimizer.param_groups:
            group["lr"] = self.base_learning_rate * group.get("lr_factor", 1.0) * lr_scale
            group["weight_decay"] = self.base_weight_decay * group.get("wd_factor", 1.0) * wd_scale
        if hasattr(self.model, "set_dropout"):
            self.model.set_dropout(self.base_dropout * drop_scale)

    # backwards compatibility for tests and existing external code
    def _apply_realmlp_step_hparams(self):
        self._apply_step_hparams()

    def _maybe_run_data_dependent_init(self, dataloader):
        if (
            not self.has_data_dependent_init
            or not self.use_data_dependent_init
            or self.data_dependent_init_done
        ):
            return
        sampled_batches = []
        max_batches = max(1, self.data_dependent_init_batches)
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            split_batch = self.split_batch(batch)
            for sub_batch in split_batch:
                sub_batch = batch_to_device(sub_batch, device=self.device)
                sampled_batches.append(sub_batch[0])

        if len(sampled_batches) == 0:
            return
        ddinit_kwargs = {
            "init_mode": self.data_dependent_init_mode,
            "target_var": self.data_dependent_init_target_var,
            "max_rows": self.data_dependent_init_max_rows,
            "gain_clip": self.data_dependent_init_gain_clip,
            "bias_mode": self.data_dependent_init_bias_mode,
            "bias_scale": self.data_dependent_init_bias_scale,
            "bias_refit_steps": self.data_dependent_init_bias_refit_steps,
        }
        signature = inspect.signature(self.model.data_dependent_init)
        accepted_kwargs = {
            key: value for key, value in ddinit_kwargs.items() if key in signature.parameters
        }
        with torch.no_grad():
            self.model.data_dependent_init(sampled_batches, **accepted_kwargs)
        self.data_dependent_init_done = True

    # backwards compatibility for tests and existing external code
    def _maybe_run_realmlp_data_dependent_init(self, dataloader):
        self._maybe_run_data_dependent_init(dataloader)

    def fit(self, dataset, test_dataset):
        self._maybe_log_input_fingerprint(dataset, stage="fit")
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset, generator=self.train_generator),
                batch_size=self.batch_size,
                drop_last=True if len(dataset) > self.batch_size else False,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=SequentialSampler(test_dataset),
                batch_size=self.batch_size,
                drop_last=False,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        self._maybe_run_data_dependent_init(train_dataloader)
        self.total_steps = max(1, self.epochs * len(train_dataloader))
        if self.ema_enabled and self.ema_start_fraction is not None:
            fraction = min(max(self.ema_start_fraction, 0.0), 1.0)
            self.ema_start_step = int(fraction * self.total_steps)
        self.global_step = 0
        self.optimizer_step_count = 0
        self._initialize_ema_if_needed()

        trained_epochs = dict()
        times = list()
        learning_rates = list()
        all_scores = list()
        model_state_dict = dict()
        for epoch in range(self.epochs):
            start_time = time.time()
            training_loss = self.fit_epoch(train_dataloader)
            scores = self.score(test_dataloader)
            end_time = time.time()
            delta_time = end_time - start_time
            current_epoch = epoch + self.previous_epochs
            lr = self.optimizer.param_groups[0]["lr"]
            self.print_progress(scores, training_loss, delta_time, current_epoch)
            if self.scheduler is not None:
                self.scheduler.step(scores["metric"])
            all_scores.append(scores)
            learning_rates.append(lr)
            times.append(round(delta_time, 3))
            self.ema_active_by_epoch.append(bool(self.ema_active))

            if self.early_stopper:
                self.early_stopper(scores["metric"])
                if self.early_stopper.improved:
                    model_state_dict[epoch] = self._current_eval_state_dict()
                    trained_epochs[epoch] = current_epoch
                if self.early_stopper.early_stop:
                    print("Early stopping, validation metric stopped improving")
                    print(
                        f"Average time per epoch was: {torch.mean(torch.as_tensor(times)).item():.2f} seconds"
                    )
                    self.finish_fit(
                        all_scores, model_state_dict, trained_epochs, learning_rates
                    )
                    return
            else:
                model_state_dict[epoch] = self._current_eval_state_dict()
                trained_epochs[epoch] = current_epoch
        print(
            f"Average time per epoch was: {torch.mean(torch.as_tensor(times)).item()} seconds"
        )
        self.finish_fit(all_scores, model_state_dict, trained_epochs, learning_rates)
        return

    def fit_epoch(self, dataloader):
        training_losses = torch.empty(len(dataloader))
        self.model.train()
        index = 0
        self.optimizer.zero_grad()
        for batch in tqdm(dataloader):
            self._apply_step_hparams()
            split_batch = self.split_batch(batch)
            accumulated_loss = 0
            logged_diagnostics = False
            for sub_batch in split_batch:
                sub_batch = batch_to_device(sub_batch, device=self.device)
                if not logged_diagnostics:
                    self._log_batch_diagnostics(sub_batch[0])
                    logged_diagnostics = True
                out = self.model(sub_batch[0])
                loss = self._compute_loss(
                    out,
                    sub_batch[1],
                    features=sub_batch[0],
                    training=True,
                )
                reg = self._regularization_loss()
                if reg is not None:
                    # Main loss uses reduction="sum"; scale regularization accordingly.
                    loss = loss + reg * sub_batch[1].shape[0]
                loss.backward()
                accumulated_loss += loss.detach()

            if self.sam_enabled:
                self._sam_step(split_batch)
            else:
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self._update_ema()
            self.optimizer_step_count += 1
            if self.apply_dynamic_schedule:
                self.global_step += 1
            training_losses[index] = accumulated_loss / self.batch_size
            index += 1
        return training_losses.mean().item()

    def _sam_step(self, split_batch):
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if len(params) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return

        with torch.no_grad():
            grad_norm_sq = torch.zeros((), device=self.device)
            for p in params:
                grad = p.grad
                if self.sam_adaptive:
                    grad = grad * p.abs()
                grad_norm_sq += torch.sum(grad * grad)
            grad_norm = torch.sqrt(grad_norm_sq).clamp_min(self.sam_eps)
            scale = self.sam_rho / grad_norm
            for p in params:
                grad = p.grad
                if self.sam_adaptive:
                    e_w = (p * p) * grad * scale
                else:
                    e_w = grad * scale
                p.add_(e_w)
                p._sam_e_w = e_w

        self.optimizer.zero_grad()
        for sub_batch in split_batch:
            sub_batch = batch_to_device(sub_batch, device=self.device)
            out = self.model(sub_batch[0])
            loss = self._compute_loss(
                out,
                sub_batch[1],
                features=sub_batch[0],
                training=True,
            )
            reg = self._regularization_loss()
            if reg is not None:
                loss = loss + reg * sub_batch[1].shape[0]
            loss.backward()

        with torch.no_grad():
            for p in params:
                e_w = getattr(p, "_sam_e_w", None)
                if e_w is not None:
                    p.sub_(e_w)
                    delattr(p, "_sam_e_w")

        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def score(self, dataloader):
        with self._ema_eval_context(), torch.no_grad():
            loss = torch.empty(len(dataloader))
            predictions = list()
            targets = list()
            self.model.eval()
            index = 0
            for batch in tqdm(dataloader):
                split_batch = self.split_batch(batch)
                accumulated_loss = 0
                for sub_batch in split_batch:
                    sub_batch = batch_to_device(sub_batch, device=self.device)
                    pred = self.model(sub_batch[0])
                    predictions.append(self._prediction_scores(pred))
                    targets.append(sub_batch[1])
                    accumulated_loss += self._compute_loss(
                        pred, sub_batch[1], features=sub_batch[0], training=False
                    ).detach()
                loss[index] = accumulated_loss / self.batch_size

                index += 1
            mean_loss = loss.mean().item()
            predictions = torch.concat(predictions)
            targets = torch.concat(targets)
            auc = compute_auc(targets.cpu(), predictions.cpu())
            scores = dict()
            if self.metric:
                if self.metric["name"] == "auc":
                    scores["metric"] = auc
                elif self.metric["name"] == "loss":
                    scores["metric"] = mean_loss
                else:
                    metric = self.metric["fun"](predictions, targets)
                    scores["metric"] = metric
            scores["auc"] = auc
            scores["loss"] = mean_loss
            return scores

    def _initialize_ema_if_needed(self):
        if not self.ema_enabled:
            return
        self.ema_state = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.ema_initialized = len(self.ema_state) > 0
        self.ema_active = False
        self.ema_first_active_step = None
        self.ema_active_by_epoch = list()

    def _update_ema(self):
        if not self.ema_enabled or not self.ema_initialized:
            return
        if self.optimizer_step_count < self.ema_start_step:
            return
        decay = float(min(max(self.ema_decay, 0.0), 0.999999))
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                ema_parameter = self.ema_state.get(name)
                if ema_parameter is None:
                    self.ema_state[name] = parameter.detach().clone()
                    continue
                ema_parameter.mul_(decay).add_(parameter.detach(), alpha=(1.0 - decay))
        self.ema_active = True
        if self.ema_first_active_step is None:
            self.ema_first_active_step = int(self.optimizer_step_count + 1)

    @contextlib.contextmanager
    def _ema_eval_context(self):
        if not (
            self.ema_enabled
            and self.ema_eval_use
            and self.ema_initialized
            and self.ema_active
        ):
            yield
            return
        backup_state = dict()
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name not in self.ema_state:
                    continue
                backup_state[name] = parameter.detach().clone()
                parameter.copy_(self.ema_state[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    backup_parameter = backup_state.get(name)
                    if backup_parameter is not None:
                        parameter.copy_(backup_parameter)

    def _current_eval_state_dict(self):
        if not (
            self.ema_enabled
            and self.ema_eval_use
            and self.ema_initialized
            and self.ema_active
        ):
            return self.model.state_dict()
        state = self.model.state_dict()
        for name, value in self.ema_state.items():
            if name in state:
                state[name] = value.detach().clone()
        return state

    def finish_fit(self, scores, model_state_dict, epoch, learning_rates):
        metric_values = [x["metric"] for x in scores]
        if self.metric["mode"] in ("max", "min"):
            best_epoch_index = select_best_epoch(metric_values, self.metric["mode"])
        else:
            raise ValueError(f"Unknown metric mode: {self.metric['mode']}")

        self.best_epoch_index = int(best_epoch_index)
        if len(self.ema_active_by_epoch) > best_epoch_index:
            self.ema_active_at_best_epoch = bool(self.ema_active_by_epoch[best_epoch_index])
        else:
            self.ema_active_at_best_epoch = False

        best_model_state_dict = model_state_dict[best_epoch_index]
        self.model.load_state_dict(best_model_state_dict)

        self.best_epoch = epoch[best_epoch_index]
        self.best_score = {
            "loss": scores[best_epoch_index]["loss"],
            "auc": scores[best_epoch_index]["auc"],
        }
        self.learn_rate_schedule = learning_rates[: (best_epoch_index + 1)]
        metric_name = self.metric["name"] if self.metric else "auc"
        print(f"Loaded best model (based on {metric_name}) from epoch {self.best_epoch}")
        print(f"ValLoss: {self.best_score['loss']}")
        print(f"valAUC: {self.best_score['auc']}")
        print(
            "EMA diagnostics | "
            f"ema_enabled={self.ema_enabled} | "
            f"ema_start_step={self.ema_start_step} | "
            f"ema_first_active_step={self.ema_first_active_step} | "
            f"best_epoch={self.best_epoch} | "
            f"best_epoch_index={self.best_epoch_index} | "
            f"ema_active_at_best_epoch={self.ema_active_at_best_epoch}"
        )
        if (
            self.metric
            and self.metric["name"] != "auc"
            and self.metric["name"] != "loss"
        ):
            self.best_score[self.metric["name"]] = scores[best_epoch_index]["metric"]
            print(f"{self.metric['name']}: {self.best_score[self.metric['name']]}")
        return

    def print_progress(self, scores, training_loss, delta_time, current_epoch):
        if (
            self.metric
            and self.metric["name"] != "auc"
            and self.metric["name"] != "loss"
        ):
            print(
                f"Epochs: {current_epoch} | Val {self.metric['name']}: {scores['metric']:.3f} "
                f"| Val AUC: {scores['auc']:.3f} | Val Loss: {scores['loss']:.3f} "
                f"| Train Loss: {training_loss:.3f} | Time: {delta_time:.3f} seconds "
                f"| LR: {self.optimizer.param_groups[0]['lr']}"
            )
        else:
            print(
                f"Epochs: {current_epoch} "
                f"| Val AUC: {scores['auc']:.3f} "
                f"| Val Loss: {scores['loss']:.3f} "
                f"| Train Loss: {training_loss:.3f} "
                f"| Time: {delta_time:.3f} seconds "
                f"| LR: {self.optimizer.param_groups[0]['lr']}"
            )
        return

    def split_batch(self, batch):
        if self.accumulation_steps > 1 and len(batch[0]["feature_ids"]) > self.sub_batch_size:
            data, labels = batch
            split_data = {key: list(torch.split(value, self.sub_batch_size))
                          for key, value in data.items() if value is not None}
            split_labels = list(torch.split(labels, self.sub_batch_size))

            sub_batches = []
            for i in range(len(split_labels)):
                sub_batch = {key: value[i] for key, value in split_data.items()}
                sub_batch = [sub_batch, split_labels[i]]
                sub_batches.append(sub_batch)
        else:
            sub_batches = [batch]
        return sub_batches

    def fit_whole_training_set(self, dataset, learning_rates=None):
        self._maybe_log_input_fingerprint(dataset, stage="fit_whole_training_set")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset, generator=self.train_generator),
                batch_size=self.batch_size,
                drop_last=True,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        self._maybe_run_data_dependent_init(dataloader)
        if self.apply_dynamic_schedule:
            if isinstance(learning_rates, list):
                self.best_epoch = len(learning_rates)
            elif learning_rates is not None:
                self.best_epoch = 1
            else:
                self.best_epoch = self.epochs
            self.total_steps = max(1, self.best_epoch * len(dataloader))
            self.global_step = 0
            for _ in range(self.best_epoch):
                self.fit_epoch(dataloader)
            return

        if isinstance(learning_rates, list):
            self.best_epoch = len(learning_rates)
        elif learning_rates is not None:
            learning_rates = [learning_rates]
            self.best_epoch = len(learning_rates)
        else:
            self.best_epoch = self.epochs
            learning_rates = [self.base_learning_rate] * self.best_epoch

        for epoch in range(self.best_epoch):
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rates[epoch] * group.get("lr_factor", 1.0)
            self.fit_epoch(dataloader)
        return

    def save(self, path, name):
        save_path = pathlib.Path(path).joinpath(name)
        out = dict(
            model_state_dict=self.model.state_dict(),
            model_parameters=self.model_parameters,
            estimator_settings=self.estimator_settings,
            epoch=self.epochs,
        )
        torch.save(out, f=save_path)
        return save_path

    def predict_proba(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        with torch.no_grad():
            predictions = list()
            self.model.eval()
            for batch in tqdm(dataloader):
                split_batch = self.split_batch(batch)
                for sub_batch in split_batch:
                    sub_batch = batch_to_device(sub_batch, device=self.device)
                    pred = self.model(sub_batch[0])
                    predictions.append(self._prediction_proba(pred))
            predictions = torch.concat(predictions).cpu().numpy()
        return predictions

    def predict(self, dataset, threshold=None):
        predictions = self.predict_proba(dataset)

        if threshold is None:
            # use outcome rate
            threshold = dataset.target.sum().item() / len(dataset)
        predicted_class = predictions > threshold
        return predicted_class


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=True, mode="max"):
        self.patience = patience
        self.counter = 0
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.improved = False
        self.delta = delta
        self.previous_score = 0
        self.mode = mode

    def __call__(self, metric):
        if self.mode == "max":
            score = metric
        else:
            score = -1 * metric
        if self.best_score is None:
            self.best_score = score
            self.improved = True
        elif score < (self.best_score + self.delta):
            self.counter += 1
            self.improved = False
            if self.verbose:
                print(
                    f"Early stopping counter: {self.counter}" f" out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.improved = True
        self.previous_score = score


def batch_to_device(batch, device="cpu"):
    if torch.is_tensor(batch):
        batch = batch.to(device=device, non_blocking=True)
    else:
        for ix, b in enumerate(batch):
            if isinstance(b, str):
                key = b
                b = batch[b]
            else:
                key = None
            if b is None:
                continue
            if torch.is_tensor(b):
                b_out = b.to(device=device, non_blocking=True)
            else:
                b_out = batch_to_device(b, device)
            if b_out is not None:
                if key is not None:
                    batch[key] = b_out
                else:
                    batch[ix] = b_out
    return batch


def compute_auc(y_true, y_pred):
    """
    Computes the AUC score for binary classification predictions with a fast algorithm.
    Args:
    y_true (torch.Tensor): True binary labels.
    y_pred (torch.Tensor): Predicted scores.
    Returns:
    float: Computed AUC score.
    """
    # Ensure inputs are sorted by predicted score
    _, sorted_indices = torch.sort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]

    # Get the number of positive and negative examples
    n_pos = y_true_sorted.sum()
    n_neg = (1 - y_true_sorted).sum()

    # for every negative label, count preceding positive labels in sorted labels
    num_crossings = torch.cumsum(y_true_sorted, 0)[y_true_sorted == 0].sum()

    # Compute AUC
    auc = num_crossings / (n_pos * n_neg)
    return auc


def select_best_epoch(metric_values, mode):
    if mode == "max":
        best_metric = max(metric_values)
        return max(i for i, value in enumerate(metric_values) if value == best_metric)
    if mode == "min":
        best_metric = min(metric_values)
        return max(i for i, value in enumerate(metric_values) if value == best_metric)
    raise ValueError(f"Unknown mode: {mode}")


def fit_estimator(estimator, train, test):
    try:
        estimator.fit(train, test)
    except torch.cuda.OutOfMemoryError as e:
        memory_cleanup()
        raise e
    return estimator
