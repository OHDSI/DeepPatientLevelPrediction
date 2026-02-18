import time
import pathlib

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from gpu_memory_cleanup import memory_cleanup
from InitStrategy import InitStrategy, DefaultInitStrategy
from schedules import get_schedule

class Estimator:
    """
    A class that wraps around pytorch models.
    """

    def __init__(self, model, parameters):
        self.seed = parameters["estimator_settings"]["seed"]
        if callable(parameters["estimator_settings"]["device"]):
            self.device = parameters["estimator_settings"]["device"]()
        else:
            self.device = parameters["estimator_settings"]["device"]
        torch.manual_seed(seed=self.seed)

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
        self.prefix = parameters["estimator_settings"].get("prefix", self.model.name)
        self.base_learning_rate = float(self.learning_rate)
        self.base_weight_decay = float(self.weight_decay)

        if "accumulation_steps" in parameters["estimator_settings"].keys() \
        and parameters["estimator_settings"]["accumulation_steps"]:
            self.accumulation_steps = int(parameters["estimator_settings"]["accumulation_steps"])
            self.sub_batch_size = self.batch_size // self.accumulation_steps
        else:
            self.accumulation_steps = 1
            self.sub_batch_size = self.batch_size

        self.previous_epochs = int(parameters["estimator_settings"].get("previous_epochs", 0))
        self.model.to(device=self.device)

        self.realmlp_mode = self.model.name == "RealMLP"
        self.beta2 = float(parameters["estimator_settings"].get("beta2", 0.999))
        self.eps = float(parameters["estimator_settings"].get("eps", 1e-8))
        self.scaling_lr_mult = float(parameters["estimator_settings"].get("scaling_lr_mult", 1.0))
        self.bias_lr_mult = float(parameters["estimator_settings"].get("bias_lr_mult", 1.0))
        self.act_lr_mult = float(parameters["estimator_settings"].get("act_lr_mult", 1.0))
        self.embedding_lr_mult = float(
            parameters["estimator_settings"].get("embedding_lr_mult", 1.0)
        )
        self.bias_wd_factor = float(parameters["estimator_settings"].get("bias_wd_factor", 0.0))
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
        self.apply_realmlp_dynamic_schedule = self.realmlp_mode
        self.use_data_dependent_init = bool(
            parameters["estimator_settings"].get("data_dependent_init", self.realmlp_mode)
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
        self.data_dependent_init_done = False
        self._log_realmlp_configuration()

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
            and not (self.realmlp_mode and self.lr_schedule_name is not None)
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

    def _log_realmlp_configuration(self):
        if not self.realmlp_mode:
            return
        token_aggregation = self.model_parameters.get("token_aggregation", "mean")
        feature_scale_mode = self.model_parameters.get("feature_scale_mode", "scalar")
        paper_mode = bool(self.model_parameters.get("paper_mode", False))
        print(
            "RealMLP config | "
            f"paper_mode={paper_mode} | "
            f"token_aggregation={token_aggregation} | "
            f"feature_scale_mode={feature_scale_mode} | "
            f"label_smoothing={self.label_smoothing} | "
            f"lr_schedule={self.lr_schedule_name} | "
            f"dropout_schedule={self.dropout_schedule_name} | "
            f"weight_decay_schedule={self.weight_decay_schedule_name} | "
            f"data_dependent_init={self.use_data_dependent_init} | "
            f"data_dependent_init_bias_mode={self.data_dependent_init_bias_mode}"
        )

    def _create_optimizer(self, optimizer_class):
        if not self.realmlp_mode:
            return optimizer_class(
                params=self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        param_groups = self._create_realmlp_param_groups()
        kwargs = {
            "params": param_groups,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, self.beta2),
            "eps": self.eps,
        }
        try:
            return optimizer_class(**kwargs)
        except TypeError:
            kwargs.pop("betas")
            kwargs.pop("eps")
            return optimizer_class(**kwargs)

    def _create_realmlp_param_groups(self):
        scale_params = []
        embedding_params = []
        act_params = []
        bias_params = []
        other_params = []

        for name, param in self.model.named_parameters():
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
                {
                    "params": other_params,
                    "name": "other",
                    "lr_factor": 1.0,
                    "wd_factor": 1.0,
                }
            )
        if scale_params:
            groups.append(
                {
                    "params": scale_params,
                    "name": "scale",
                    "lr_factor": self.scaling_lr_mult,
                    "wd_factor": 1.0,
                }
            )
        if embedding_params:
            groups.append(
                {
                    "params": embedding_params,
                    "name": "embed",
                    "lr_factor": self.embedding_lr_mult,
                    "wd_factor": 1.0,
                }
            )
        if bias_params:
            groups.append(
                {
                    "params": bias_params,
                    "name": "bias",
                    "lr_factor": self.bias_lr_mult,
                    "wd_factor": self.bias_wd_factor,
                }
            )
        if act_params:
            groups.append(
                {
                    "params": act_params,
                    "name": "act",
                    "lr_factor": self.act_lr_mult,
                    "wd_factor": 1.0,
                }
            )
        return groups

    def _prepare_targets_for_loss(self, targets):
        targets = targets.float()
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return targets

    def _apply_realmlp_step_hparams(self):
        if not self.realmlp_mode or not self.apply_realmlp_dynamic_schedule:
            return
        t = self.global_step / max(self.total_steps - 1, 1)
        lr_scale = self.lr_schedule(t)
        wd_scale = self.weight_decay_schedule(t)
        drop_scale = self.dropout_schedule(t)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_learning_rate * group.get("lr_factor", 1.0) * lr_scale
            group["weight_decay"] = (
                self.base_weight_decay * group.get("wd_factor", 1.0) * wd_scale
            )
        if hasattr(self.model, "set_dropout"):
            self.model.set_dropout(self.base_dropout * drop_scale)

    def _maybe_run_realmlp_data_dependent_init(self, dataloader):
        if (
            not self.realmlp_mode
            or not self.use_data_dependent_init
            or self.data_dependent_init_done
            or not hasattr(self.model, "data_dependent_init")
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
        with torch.no_grad():
            self.model.data_dependent_init(
                sampled_batches,
                bias_mode=self.data_dependent_init_bias_mode,
                bias_scale=self.data_dependent_init_bias_scale,
            )
        self.data_dependent_init_done = True

    def fit(self, dataset, test_dataset):
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=True if len(dataset) > self.batch_size else False,
            ),
            pin_memory=True,
        )
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
        self._maybe_run_realmlp_data_dependent_init(train_dataloader)
        self.total_steps = max(1, self.epochs * len(train_dataloader))
        self.global_step = 0
        self.apply_realmlp_dynamic_schedule = self.realmlp_mode

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
            self._apply_realmlp_step_hparams()
            split_batch = self.split_batch(batch)
            accumulated_loss = 0
            for sub_batch in split_batch:
                sub_batch = batch_to_device(sub_batch, device=self.device)
                out = self.model(sub_batch[0])
                loss = self.criterion(
                    out.squeeze(), self._prepare_targets_for_loss(sub_batch[1])
                )
                loss.backward()
                accumulated_loss += loss.detach()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.realmlp_mode:
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
                    predictions.append(pred)
                    targets.append(sub_batch[1])
                    accumulated_loss += self.criterion(
                        pred.squeeze(), self._prepare_targets_for_loss(sub_batch[1])
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
        print(f"Loaded best model (based on AUC) from epoch {self.best_epoch}")
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
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=True,
            ),
        )
        self._maybe_run_realmlp_data_dependent_init(dataloader)
        if self.realmlp_mode:
            if isinstance(learning_rates, list):
                self.best_epoch = len(learning_rates)
            elif learning_rates is not None:
                self.best_epoch = 1
            else:
                self.best_epoch = self.epochs
            self.total_steps = max(1, self.best_epoch * len(dataloader))
            self.global_step = 0
            self.apply_realmlp_dynamic_schedule = True
            for _ in range(self.best_epoch):
                self.fit_epoch(dataloader)
            return

        if isinstance(learning_rates, list):
            self.best_epoch = len(learning_rates)
            self.apply_realmlp_dynamic_schedule = False
        elif learning_rates is not None:
            learning_rates = [learning_rates]
            self.best_epoch = len(learning_rates)
            self.apply_realmlp_dynamic_schedule = False
        else:
            self.best_epoch = self.epochs
            self.apply_realmlp_dynamic_schedule = self.realmlp_mode
            learning_rates = [self.base_learning_rate] * self.best_epoch

        for epoch in range(self.best_epoch):
            self.optimizer.param_groups[0]["lr"] = learning_rates[epoch]
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
        )
        with torch.no_grad():
            predictions = list()
            self.model.eval()
            for batch in tqdm(dataloader):
                split_batch = self.split_batch(batch)
                for sub_batch in split_batch:
                    sub_batch = batch_to_device(sub_batch, device=self.device)
                    pred = self.model(sub_batch[0])
                    predictions.append(torch.sigmoid(pred))
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
