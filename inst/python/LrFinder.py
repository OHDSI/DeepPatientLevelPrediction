from os import walk
import random

import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from Estimator import batch_to_device
from gpu_memory_cleanup import memory_cleanup


class ExponentialSchedulerPerBatch(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialSchedulerPerBatch, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class LrFinder:
    def __init__(self, estimator, lr_settings=None):
        self.first_batch = None
        if lr_settings is None:
            lr_settings = {}
        self.min_lr = lr_settings.get("min_lr", 1e-7)
        self.max_lr = lr_settings.get("max_lr", 1)
        self.num_lr = lr_settings.get("num_lr", 100)
        self.smooth = lr_settings.get("smooth", 0.05)
        self.divergence_threshold = lr_settings.get("divergence_threshold", 4)
        torch.manual_seed(seed=estimator.seed)
        self.seed = estimator.seed

        for group in estimator.optimizer.param_groups:
            group['lr'] = self.min_lr

        estimator.scheduler = ExponentialSchedulerPerBatch(
            estimator.optimizer, self.max_lr, self.num_lr
        )
        if estimator.accumulation_steps > 1:
            self.accumulation_steps = estimator.accumulation_steps
        else:
            self.accumulation_steps = 1
        self.estimator = estimator
        self.losses = None
        self.loss_index = None

    def get_lr(self, dataset):
        if len(dataset) < self.estimator.batch_size:
            self.estimator.batch_size = len(dataset)
        batch_index = torch.arange(0, len(dataset), 1).tolist()
        random.seed(self.seed)
        losses = torch.empty(size=(self.num_lr,), dtype=torch.float)
        lrs = torch.empty(size=(self.num_lr,), dtype=torch.float)
        self.estimator.optimizer.zero_grad()
        best_loss = float("inf")
        for i in tqdm(range(self.num_lr)):
            loss_value = 0
            random_batch = random.sample(batch_index, self.estimator.batch_size)
            for j in range(self.accumulation_steps):
                batch = dataset[random_batch[j * self.estimator.sub_batch_size : (j + 1) * self.estimator.sub_batch_size]]
                batch = batch_to_device(batch, self.estimator.device)

                out = self.estimator.model(batch[0])
                loss = self.estimator.criterion(out, batch[1])
                loss.backward()
                loss_value += loss.item()
            loss_value = loss_value / self.accumulation_steps
            if self.smooth is not None and i != 0:
                losses[i] = (
                    self.smooth * loss_value + (1 - self.smooth) * losses[i - 1]
                )
            else:
                losses[i] = loss_value
            lrs[i] = self.estimator.optimizer.param_groups[0]["lr"]

            self.estimator.optimizer.step()
            self.estimator.scheduler.step()

            if losses[i] < best_loss:
                best_loss = losses[i]

            if losses[i] > (self.divergence_threshold * best_loss):
                print(
                    f"Loss diverged - stopped early - iteration {i} out of {self.num_lr}"
                )
                break

        # find LR where gradient is highest but before global minimum is reached
        # I added -5 to make sure it is not still in the minimum
        global_minimum = torch.argmin(losses[:i])
        gradient = torch.diff(losses[: (global_minimum - 5) + 1])
        biggest_gradient = torch.argmax(gradient)

        suggested_lr = lrs[biggest_gradient]
        self.losses = losses[:i]
        self.loss_index = biggest_gradient
        self.lrs = lrs
        return suggested_lr.item()


def get_lr(estimator,
           dataset,
           lr_settings):
    try:
        lr_finder = LrFinder(estimator, lr_settings=lr_settings)
        lr = lr_finder.get_lr(dataset)
    except torch.cuda.OutOfMemoryError as e:
        memory_cleanup()
        raise e
    return lr


