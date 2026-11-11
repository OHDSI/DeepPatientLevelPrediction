# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import time
import pathlib
import inspect
import os
import random

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from InitStrategy import DefaultInitStrategy
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
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.seed)
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

        self.model.to(device=self.device)

        self.has_custom_param_groups = hasattr(self.model, "get_optimizer_param_groups")
        self.has_custom_loss = hasattr(self.model, "compute_loss")
        self.has_custom_scores = hasattr(self.model, "prediction_scores")
        self.has_custom_proba = hasattr(self.model, "predict_proba_from_output")
        self.has_custom_schedule = hasattr(self.model, "apply_dynamic_schedule")
        self.has_data_dependent_init = hasattr(self.model, "data_dependent_init")

        # R converts beta2 to beta_2, while Python callers may pass beta2 directly.
        self.beta2_key = next(
            (
                key
                for key in ("beta_2", "beta2")
                if key in parameters["estimator_settings"]
            ),
            None,
        )
        self.beta2 = float(
            parameters["estimator_settings"][self.beta2_key]
            if self.beta2_key is not None
            else 0.999
        )
        self.eps = float(parameters["estimator_settings"].get("eps", 1e-8))
        self.label_smoothing = float(parameters["estimator_settings"].get("label_smoothing", 0.0))
        # settings created before this option existed keep the default initialization
        self.init_output_bias = bool(
            parameters["estimator_settings"].get("init_output_bias", False)
        )
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
        if self.beta2_key is not None:
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

    def _maybe_init_output_bias(self, dataset):
        """Start the output bias at the log-odds of the outcome rate of ``dataset``.

        A newly initialized network predicts a risk of about 50%. With a rare outcome
        and few optimizer steps per epoch, the first epochs go into lowering every
        prediction to the outcome rate. The validation AUROC barely moves meanwhile,
        so early stopping can end training before the model ranks patients at all.
        Models with their own loss or probabilities are left as they are.
        """
        if not self.init_output_bias or self.has_custom_loss or self.has_custom_proba:
            return
        layer = output_layer(self.model)
        targets = dataset_targets(dataset)
        if layer is None or layer.bias is None or layer.out_features != 1 or targets is None:
            return
        rate = targets.float().mean().item()
        # the loss is computed against smoothed labels
        rate = rate * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        if not 0.0 < rate < 1.0:
            return
        with torch.no_grad():
            layer.bias.fill_(math.log(rate / (1.0 - rate)))

    def _train_dataloader(self, dataset):
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty dataset")
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset, generator=self.train_generator),
                batch_size=self.batch_size,
                # keep the only batch when the data is smaller than one batch
                drop_last=len(dataset) > self.batch_size,
            ),
            pin_memory=True,
        )

    def _keep_best_checkpoint(self, checkpoints, all_scores, epoch):
        """Copy the current weights if this epoch is the best one so far.

        state_dict() shares storage with the live parameters, so a checkpoint
        has to be a copy to survive later optimizer steps. Only the best
        checkpoint is retained to avoid holding one model per epoch in memory.
        """
        candidates = [*checkpoints, epoch]
        metric_values = [all_scores[i]["metric"] for i in candidates]
        best = candidates[select_best_epoch(metric_values, self.metric["mode"])]
        if best != epoch:
            return
        checkpoints.clear()
        checkpoints[epoch] = {
            name: value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
            for name, value in self.model.state_dict().items()
        }

    def fit(self, dataset, test_dataset):
        train_dataloader = self._train_dataloader(dataset)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=SequentialSampler(test_dataset),
                batch_size=self.batch_size,
                drop_last=False,
            ),
            pin_memory=True,
        )
        self._maybe_run_data_dependent_init(train_dataloader)
        self._maybe_init_output_bias(dataset)
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
            lr = self.optimizer.param_groups[0]["lr"]
            self.print_progress(scores, training_loss, delta_time, epoch)
            if self.scheduler is not None:
                self.scheduler.step(scores["metric"])
            all_scores.append(scores)
            learning_rates.append(lr)
            times.append(round(delta_time, 3))

            if self.early_stopper:
                self.early_stopper(scores["metric"])
                if self.early_stopper.improved:
                    self._keep_best_checkpoint(model_state_dict, all_scores, epoch)
                    trained_epochs[epoch] = epoch
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
                self._keep_best_checkpoint(model_state_dict, all_scores, epoch)
                trained_epochs[epoch] = epoch
        print(
            f"Average time per epoch was: {torch.mean(torch.as_tensor(times)).item()} seconds"
        )
        self.finish_fit(all_scores, model_state_dict, trained_epochs, learning_rates)
        return

    def fit_epoch(self, dataloader):
        total_loss = 0.0
        num_observations = 0
        self.model.train()
        self.optimizer.zero_grad()
        for batch in tqdm(dataloader):
            self._apply_step_hparams()
            split_batch = self.split_batch(batch)
            accumulated_loss = 0
            for sub_batch in split_batch:
                sub_batch = batch_to_device(sub_batch, device=self.device)
                out = self.model(sub_batch[0])
                loss = self._compute_loss(out, sub_batch[1])
                loss.backward()
                accumulated_loss += loss.detach()
                num_observations += sub_batch[1].numel()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.apply_dynamic_schedule:
                self.global_step += 1
            total_loss += float(accumulated_loss)
        return total_loss / max(num_observations, 1)

    def score(self, dataloader):
        with torch.no_grad():
            total_loss = 0.0
            num_observations = 0
            predictions = list()
            targets = list()
            self.model.eval()
            for batch in tqdm(dataloader):
                split_batch = self.split_batch(batch)
                for sub_batch in split_batch:
                    sub_batch = batch_to_device(sub_batch, device=self.device)
                    pred = self.model(sub_batch[0])
                    predictions.append(self._prediction_scores(pred))
                    # a single-row batch has a 0-d target, which cannot be concatenated
                    targets.append(sub_batch[1].reshape(-1))
                    total_loss += float(self._compute_loss(pred, sub_batch[1]).detach())
                    num_observations += sub_batch[1].numel()
            mean_loss = total_loss / max(num_observations, 1)
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
        if not isinstance(model_state_dict, dict):
            model_state_dict = dict(enumerate(model_state_dict))
        # only epochs with a retained checkpoint can be restored
        checkpointed_epochs = sorted(model_state_dict)
        metric_values = [scores[i]["metric"] for i in checkpointed_epochs]
        if self.metric["mode"] in ("max", "min"):
            best_epoch_index = checkpointed_epochs[
                select_best_epoch(metric_values, self.metric["mode"])
            ]
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
        dataloader = self._train_dataloader(dataset)
        self._maybe_run_data_dependent_init(dataloader)
        self._maybe_init_output_bias(dataset)
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
            pin_memory=True,
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


def output_layer(model):
    """The final linear layer of a model with a ``head``, or None if there is none."""
    model = getattr(model, "_orig_mod", model)  # compiled models wrap the original
    head = getattr(model, "head", None)
    if isinstance(head, torch.nn.Linear):
        return head
    linear = getattr(head, "linear", None)
    return linear if isinstance(linear, torch.nn.Linear) else None


def dataset_targets(dataset):
    """The labels of a dataset or of a Subset of one, or None if they are not available."""
    if isinstance(dataset, torch.utils.data.Subset):
        targets = dataset_targets(dataset.dataset)
        if targets is None:
            return None
        return targets[torch.as_tensor(dataset.indices, dtype=torch.long)]
    return getattr(dataset, "target", None)


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=True, mode="max", min_epochs=0):
        self.patience = patience
        self.counter = 0
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.improved = False
        self.delta = delta
        self.previous_score = 0
        self.mode = mode
        self.min_epochs = min_epochs
        self.epochs_seen = 0

    def __call__(self, metric):
        # called once per completed training epoch
        self.epochs_seen += 1
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
            if self.counter >= self.patience and self.epochs_seen >= self.min_epochs:
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
    Positive-negative pairs with tied scores get half credit.
    Args:
    y_true (torch.Tensor): True binary labels.
    y_pred (torch.Tensor): Predicted scores.
    Returns:
    float: Computed AUC score.
    """
    # Ensure inputs are sorted by predicted score
    sorted_scores, sorted_indices = torch.sort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices].to(dtype=torch.double)

    # Count the positive and negative examples within each group of equal scores
    _, group_index, group_sizes = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    positives = torch.zeros_like(group_sizes, dtype=torch.double).index_add_(
        0, group_index, y_true_sorted
    )
    negatives = group_sizes.to(dtype=torch.double) - positives

    # a negative ranks below the positives in earlier groups and ties with the
    # positives in its own group
    preceding_positives = torch.cumsum(positives, 0) - positives
    num_crossings = (negatives * (preceding_positives + 0.5 * positives)).sum()

    # Compute AUC
    auc = num_crossings / (positives.sum() * negatives.sum())
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
    # after an out-of-memory error, R frees the memory once it has dropped the
    # error, whose traceback still refers to this fit
    estimator.fit(train, test)
    return estimator
