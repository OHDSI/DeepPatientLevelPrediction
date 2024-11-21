import time
import pathlib

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from gpu_memory_cleanup import memory_cleanup
from InitStrategy import InitStrategy, DefaultInitStrategy

class Estimator:
    """
    A class that wraps around pytorch models.
    """

    def __init__(self, model, model_parameters, estimator_settings):
        self.seed = estimator_settings["seed"]
        if callable(estimator_settings["device"]):
            self.device = estimator_settings["device"]()
        else:
            self.device = estimator_settings["device"]
        torch.manual_seed(seed=self.seed)

        if "init_strategy" in estimator_settings:
            self.model = estimator_settings["init_strategy"].initialize(model, model_parameters, estimator_settings)
        else:
            self.model = DefaultInitStrategy().initialize(model, model_parameters, estimator_settings)
            
        self.model_parameters = model_parameters
        self.estimator_settings = estimator_settings

        self.epochs = int(estimator_settings.get("epochs", 5))
        if estimator_settings["find_l_r"]:
            self.learning_rate = 3e-4
        else:
            self.learning_rate = estimator_settings.get("learning_rate", 3e-4)
        self.weight_decay = estimator_settings.get("weight_decay", 1e-5)
        self.batch_size = int(estimator_settings.get("batch_size", 1024))
        self.prefix = estimator_settings.get("prefix", self.model.name)
        
        if "accumulation_steps" in estimator_settings.keys() and estimator_settings["accumulation_steps"]:
            self.accumulation_steps = int(estimator_settings["accumulation_steps"])
            self.sub_batch_size = self.batch_size // self.accumulation_steps
        else:
            self.accumulation_steps = 1
            self.sub_batch_size = self.batch_size
        
        self.previous_epochs = int(estimator_settings.get("previous_epochs", 0))
        self.model.to(device=self.device)

        self.optimizer = estimator_settings["optimizer"](
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.criterion = estimator_settings["criterion"](reduction="sum")

        if (
            "metric" in estimator_settings.keys()
            and estimator_settings["metric"] is not None
        ):
            self.metric = estimator_settings["metric"]
            if isinstance(self.metric, str):
                if self.metric == "auc":
                    self.metric = {"name": "auc", "mode": "max"}
                elif self.metric == "loss":
                    self.metric = {"name": "loss", "mode": "min"}
            if (
                "scheduler" in estimator_settings.keys()
                and estimator_settings["scheduler"] is not None
            ):
                estimator_settings["scheduler"]["params"]["mode"] = self.metric["mode"]
            if (
                "early_stopping" in estimator_settings.keys()
                and estimator_settings["early_stopping"] is not None
            ):
                estimator_settings["early_stopping"]["params"]["mode"] = self.metric[
                    "mode"
                ]

        if (
            "scheduler" in estimator_settings.keys()
            and estimator_settings["scheduler"] is not None
        ):
            self.scheduler = estimator_settings["scheduler"]["fun"](
                self.optimizer, **estimator_settings["scheduler"]["params"]
            )

        if (
            "early_stopping" in estimator_settings.keys()
            and estimator_settings["early_stopping"] is not None
        ):
            self.early_stopper = EarlyStopping(
                **estimator_settings["early_stopping"]["params"]
            )
        else:
            self.early_stopper = None

        self.best_score = None
        self.best_epoch = None
        self.learn_rate_schedule = None
        if estimator_settings["compile"]:
            self.model = torch.compile(self.model, dynamic=False)

    def fit(self, dataset, test_dataset):
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=True,
            ),
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler=SequentialSampler(test_dataset),
                batch_size=self.batch_size,
                drop_last=False,
            ),
        )

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
            split_batch = self.split_batch(batch)
            accumulated_loss = 0
            all_out = []
            for sub_batch in split_batch:
                sub_batch = batch_to_device(sub_batch, device=self.device)
                out = self.model(sub_batch[0])
                all_out.append(out.detach())
                loss = self.criterion(out.squeeze(), sub_batch[1])
                loss.backward()
                accumulated_loss += loss.detach()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
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
                    accumulated_loss += self.criterion(pred.squeeze(), sub_batch[1]).detach()
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
        if self.metric["mode"] == "max":
            best_epoch_index = torch.argmax(
                torch.as_tensor([x["metric"] for x in scores])
            ).item()
        elif self.metric["mode"] == "min":
            best_epoch_index = torch.argmin(
                torch.as_tensor([x["metric"] for x in scores])
            ).item()

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
        if self.accumulation_steps > 1 and len(batch[0]["cat"]) > self.sub_batch_size:
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
        if isinstance(learning_rates, list):
            self.best_epoch = len(learning_rates)
        elif ~isinstance(learning_rates, list):
            learning_rates = [learning_rates]
            self.best_epoch = len(learning_rates)
        else:
            self.best_epoch = self.epochs

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
        batch = batch.to(device=device)
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
                b_out = b.to(device=device)
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


def fit_estimator(estimator, train, test):
    try:
        estimator.fit(train, test)
    except torch.cuda.OutOfMemoryError as e:
        memory_cleanup()
        raise e
    return estimator
