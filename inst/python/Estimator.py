import time
import pathlib

import torch


class Estimator:
    """
    A class that wraps around pytorch models.
    """

    def __init__(self,
                 model,
                 model_parameters,
                 estimator_settings):
        self.seed = estimator_settings.seed

        if not type(estimator_settings.device) == str:
            device = estimator_settings.device()
        else:
            device = estimator_settings.device
        self.device = device

        torch.manual_seed(seed=self.seed)
        self.model = model(**model_parameters)
        self.model_parameters = model_parameters
        self.estimator_settings = estimator_settings

        self.epochs = estimator_settings.get("epochs", 5)
        self.learning_rate = estimator_settings.get("lr", 3e-4)
        self.weight_decay = estimator_settings.get("weight_decay", 1e-5)
        self.batch_size = estimator_settings.get("batch_size", 1024)
        self.prefix = estimator_settings.get("prefix", self.model.name)

        self.previous_epochs = estimator_settings.get("previous_epochs", 0)
        self.model.to(device=self.device)

        self.optimizer = estimator_settings.optimizer(params=self.model.parameters(),
                                                      lr=self.learning_rate,
                                                      weight_decay=self.weight_decay)
        self.criterion = estimator_settings.criterion()

        if estimator_settings.metric:
            self.metric = estimator_settings.metric
            if type(self.metric) == str:
                if self.metric == "auc":
                    self.metric = {"name": "auc",
                                   "mode": "max"}
                elif self.metric == "loss":
                    self.metric = {"name": "loss",
                                   "mode": "min"}
            if estimator_settings.scheduler:
                estimator_settings.scheduler.params.mode = self.metric.mode
            if estimator_settings.early_stopping:
                estimator_settings.early_stopping.params.mode = self.metric.mode

        if estimator_settings.scheduler:
            self.scheduler = estimator_settings.scheduler.fun(self.optimizer,
                                                              **estimator_settings.scheduler.params)

        if estimator_settings.early_stopping:
            self.early_stopper = EarlyStopping(**estimator_settings.early_stopping.params)
        else:
            self.early_stopper = None

        self.best_score = None
        self.best_epoch = None
        self.learn_rate_schedule = None

    def fit(self, dataset, test_dataset):
        all_scores = list()

        trained_epochs = list()
        times = list()
        learning_rates = list()
        for epoch in range(self.epochs):
            start_time = time.time()
            training_loss = self.fit_epoch(dataset)
            scores = self.score(test_dataset)
            end_time = time.time()
            delta_time = end_time - start_time
            current_epoch = epoch + self.previous_epochs
            lr = self.optimizer.param_groups[0].lr
            self.print_progress(scores, training_loss, delta_time, current_epoch)
            self.scheduler.step(scores.metric)
            all_scores[epoch] = scores
            learning_rates.append(lr)
            times.append(round(delta_time, 3))

            if self.early_stopper:
                self.early_stopper(scores.metric)
                if self.early_stopper.improved:
                    model_state_dict = None
                    pass
                if self.early_stopper.early_stop:
                    print("Early stopping, validation metric stopped improving")
                    print(f'Average time per epoch was: {torch.mean(torch.as_tensor(times).item())} seconds')
                    self.finish_fit(all_scores, model_state_dict, epoch, learning_rates)
                    return
        print(f'Average time per epoch was: {torch.mean(torch.as_tensor(times).item())} seconds')
        self.finish_fit(all_scores, model_state_dict, epoch, )
        return

    def fit_epoch(self, dataset):
        training_losses = torch.empty(len(dataset))  # dataloader
        self.model.train()
        index = 0
        for batch in dataset:  # dataloader
            self.optimizer.zero_grad()
            batch = batch_to_device(batch, device=self.device)
            out = self.model(batch[0])
            loss = self.criterion(out, batch[1])
            loss.backward()

            self.optimizer.step()
            training_losses[index] = loss.detach()
            index += 1
        return training_losses.mean().item()

    def score(self, dataset):
        with torch.no_grad:
            loss = torch.empty(len(dataset))  # dataloader
            predictions = list()
            targets = list()
            self.model.eval()
            index = 0
            for batch in dataset:
                batch = batch_to_device(batch, device=self.device)
                pred = self.model(batch[0])
                predictions.append(pred)
                targets.append(batch[1])
                loss[index] = self.criterion(pred, batch[1])
                index += 1
            mean_loss = loss.mean().item()
            auc = compute_auc(predictions, targets)
            scores = dict()
            if self.metric:
                if self.metric["name"] == "auc":
                    scores["metric"] = auc
                elif self.metric["name"] == "loss":
                    scores["metric"] = mean_loss
                else:
                    metric = self.metric.fun(predictions, targets)
                    scores["metric"] = metric
            scores["auc"] = auc
            scores["loss"] = mean_loss
            return scores

    def finish_fit(self, scores, model_state_dict, epoch, learning_rates):
        if self.metric["mode"] == "max":
            best_epoch_index = torch.argmax(torch.as_tensor([x["metric"] for x in scores]))[0]
        elif self.metric["mode"] == "min":
            best_epoch_index = torch.argmin(torch.as_tensor([x["metric"] for x in scores]))[0]

        best_model_state_dict = [x.to(device=self.device) for x in model_state_dict[best_epoch_index]]
        self.model.load_state_dict(best_model_state_dict)

        self.best_epoch = epoch(best_epoch_index)
        self.best_score = {"loss": scores[best_epoch_index]["loss"],
                           "auc": scores[best_epoch_index]["auc"]}
        self.learn_rate_schedule = learning_rates[:best_epoch_index]
        print(f"Loaded best model (based on AUC) from epoch {self.best_epoch}")
        print(f"ValLoss: {self.best_score['loss']}")
        print(f"valAUC: {self.best_score['auc']}")
        if self.metric and self.metric["name"] != "auc" and self.metric["name"] != "loss":
            self.best_score[self.metric["name"]] = scores[best_epoch_index]["metric"]
            print(f"{self.metric['name']}: {self.best_score[self.metric['name']]}")
        return

    def print_progress(self, scores, training_loss, delta_time, current_epoch):
        if self.metric and self.metric["name"] != "auc" and self.metric["name"] != "loss":
            print(f"Epochs: {current_epoch} | Val {self.metric['name']}: {scores['metric']:.2f)} "
                  f"| Val AUC: {scores['auc']:.2f} | Val Loss: {scores['loss']:.2f} "
                  f"| Train Loss: {training_loss:.2f} | Time: {delta_time:.2f} seconds "
                  f"| LR: {self.optimizer.param_groups[0].lr}")
        else:
            print(f"Epochs: {current_epoch} "
                  f"| Val AUC: {scores['auc']:.2f} "
                  f"| Val Loss: {scores['loss']:.2f} "
                  f"| Train Loss: {training_loss:.2f} "
                  f"| Time: {delta_time:.2f} seconds "
                  f"| LR: {self.optimizer.param_groups[0].lr}")
        return

    def fit_whole_training_set(self, dataset, learning_rates=None):
        if len(learning_rates) > 1:
            self.best_epoch = len(learning_rates)
        elif self.best_epoch is None:
            self.best_epoch = self.epochs

        for epoch in self.epochs:
            self.optimizer.param_groups[0].lr = learning_rates[epoch]
            self.fit_epoch(dataset)
        return

    def save(self, path, name):
        save_path = pathlib.Path(path).joinpath(name)
        out = dict(
            model_state_dict=self.model.state_dict(),
            model_parameters=self.model_parameters,
            estimator_settings=self.estimator_settings,
            epoch=self.epochs)
        torch.save(out,
                   f=save_path)
        return save_path

    def predict_proba(self, dataset):
        with torch.no_grad:
            predictions = torch.empty(len(dataset), device=self.device)
            self.model.eval()
            for batch in dataset:  # dataloader
                batch = batch_to_device(batch, device=self.device)
                pred = self.model[batch["batch"]]
                predictions[batch] = torch.sigmoid(pred)  # TODO fix index
        return predictions

    def predict(self, dataset, threshold=None):
        predictions = self.predict_proba(dataset)

        if threshold is None:
            # use outcome rate
            threshold = dataset.target.sum().item() / len(dataset)
        predicted_class = predictions > threshold


class EarlyStopping:

    def __init__(self,
                 patience=3,
                 delta=0,
                 verbose=True,
                 mode='max'):
        self.patience = patience
        self.counter = 0
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.improved = False
        self.delta = delta
        self.previous_score = 0
        self.mode = mode

    def __call__(self,
                 metric):
        if self.mode == 'max':
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
                print(f"Early stopping counter: {self.counter}"
                      f" out of {self.patience}")
            if self.counter >= self.patience:
                self.best_score = score
                self.counter = 0
                self.improved = True
        self.previous_score = score


def batch_to_device(batch, device='cpu'):
    if torch.is_tensor(batch):
        batch = batch.to(device=device)
    else:
        ix = 1
        for b in batch:
            if torch.is_tensor(b):
                b = b.to(device=device)
            else:
                b = batch_to_device(b, device)
            if b is not None:
                batch[ix] = b
            ix += 1
    return batch




def compute_auc(predictions, targets):
    pass
