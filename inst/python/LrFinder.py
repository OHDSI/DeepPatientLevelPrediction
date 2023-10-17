import random

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm

from Estimator import batch_to_device


class ExponentialSchedulerPerBatch(_LRScheduler):

    def __init__(self, optimizer,
                 end_lr,
                 num_iter):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialSchedulerPerBatch, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class LrFinder:

    def __init__(self,
                 model,
                 model_parameters,
                 estimator_settings,
                 lr_settings):
        if lr_settings is None:
          lr_settings = {}
        min_lr = lr_settings.get("min_lr", 1e-7)
        max_lr = lr_settings.get("max_lr", 1)
        num_lr = lr_settings.get("num_lr", 100)
        smooth = lr_settings.get("smooth", 0.05)
        divergence_threshold = lr_settings.get("divergence_threshold", 4)
        torch.manual_seed(seed=estimator_settings["seed"])
        self.seed = estimator_settings["seed"]
        self.model = model(**model_parameters)
        if callable(estimator_settings["device"]):
            self.device = estimator_settings["device"]()
        else:
            self.device = estimator_settings["device"]
        self.model.to(device=self.device)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_lr = num_lr
        self.smooth = smooth
        self.divergence_threshold = divergence_threshold

        self.optimizer = estimator_settings['optimizer'](params=self.model.parameters(),
                                                         lr=self.min_lr)

        self.scheduler = ExponentialSchedulerPerBatch(self.optimizer, self.max_lr, self.num_lr)

        self.criterion = estimator_settings["criterion"]()
        self.batch_size = int(estimator_settings['batch_size'])
        self.losses = None
        self.loss_index = None

    def get_lr(self, dataset):
        batch_index = torch.arange(0, len(dataset), 1).tolist()
        random.seed(self.seed)
        losses = torch.empty(size=(self.num_lr,), dtype=torch.float)
        lrs = torch.empty(size=(self.num_lr,), dtype=torch.float)
        for i in tqdm(range(self.num_lr)):
            self.optimizer.zero_grad()
            random_batch = random.sample(batch_index, self.batch_size)
            batch = dataset[random_batch]
            batch = batch_to_device(batch, self.device)

            out = self.model(batch[0])
            loss = self.criterion(out, batch[1])
            if self.smooth is not None and i != 0:
                losses[i] = self.smooth * loss.item() + (1 - self.smooth) * losses[i - 1]
            else:
                losses[i] = loss.item()
            lrs[i] = self.optimizer.param_groups[0]["lr"]

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if i == 0:
                best_loss = losses[i]
            else:
                if losses[i] < best_loss:
                    best_loss = losses[i]

            if losses[i] > (self.divergence_threshold * best_loss):
                print(f"Loss diverged - stopped early - iteration {i} out of {self.num_lr}")
                break

        # find LR where gradient is highest but before global minimum is reached
        # I added -5 to make sure it is not still in the minimum
        global_minimum = torch.argmin(losses)
        gradient = torch.diff(losses[:(global_minimum - 5)+1])
        smallest_gradient = torch.argmin(gradient)

        suggested_lr = lrs[smallest_gradient]
        self.losses = losses
        self.loss_index = smallest_gradient
        self.lrs = lrs
        return suggested_lr.item()
