# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import time
import pathlib
import inspect
import hashlib
import os
import random

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

    def _compute_loss(self, predictions, targets):
        if self.has_custom_loss:
            return self.model.compute_loss(
                predictions=predictions,
                targets=targets,
                criterion=self.criterion,
                label_smoothing=self.label_smoothing,
            )
        prepared_targets = self._default_prepare_targets_for_loss(targets)
        return self.criterion(predictions.squeeze(), prepared_targets)

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
        self.global_step = 0

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

            if self.early_stopper:
                self.early_stopper(scores["metric"])
                if self.early_stopper.improved:
                    model_state_dict[epoch] = self.model.state_dict()
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
                model_state_dict[epoch] = self.model.state_dict()
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
                loss = self._compute_loss(out, sub_batch[1])
                reg = self._regularization_loss()
                if reg is not None:
                    # Main loss uses reduction="sum"; scale regularization accordingly.
                    loss = loss + reg * sub_batch[1].shape[0]
                loss.backward()
                accumulated_loss += loss.detach()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.apply_dynamic_schedule:
                self.global_step += 1
            training_losses[index] = accumulated_loss / self.batch_size
            index += 1
        return training_losses.mean().item()

    def score(self, dataloader):
        with torch.no_grad():
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
                    accumulated_loss += self._compute_loss(pred, sub_batch[1]).detach()
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

    def finish_fit(self, scores, model_state_dict, epoch, learning_rates):
        metric_values = [x["metric"] for x in scores]
        if self.metric["mode"] in ("max", "min"):
            best_epoch_index = select_best_epoch(metric_values, self.metric["mode"])
        else:
            raise ValueError(f"Unknown metric mode: {self.metric['mode']}")

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
