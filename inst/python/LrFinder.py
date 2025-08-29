import random
import math

from typing import Any, Dict, Tuple, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
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
        if self.num_lr < 20:
            self.min_factor = 0
        else:
            self.min_factor = 5

        for group in estimator.optimizer.param_groups:
            group["lr"] = self.min_lr

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
                batch = dataset[
                    random_batch[
                        j * self.estimator.sub_batch_size : (j + 1)
                        * self.estimator.sub_batch_size
                    ]
                ]
                batch = batch_to_device(batch, self.estimator.device)

                out = self.estimator.model(batch[0])
                loss = self.estimator.criterion(out, batch[1])
                loss.backward()
                loss_value += loss.item()
            loss_value = loss_value / self.accumulation_steps
            if self.smooth is not None and i != 0:
                losses[i] = self.smooth * loss_value + (1 - self.smooth) * losses[i - 1]
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

        used = i + 1
        lrs_used = lrs[:used]
        losses_used = losses[:used]
        suggested, details = suggest_lr(
            lrs_used,
            losses_used,
            min_factor=self.min_factor,
            burn_in_frac=0.05,
            burn_in_min_pts=3,
            smooth_window=3,
            return_details=True,
        )
        self.losses = losses_used
        self.loss_index = details["pick_idx"]
        self.lrs = lrs_used
        return suggested


def get_lr(estimator, dataset, lr_settings):
    try:
        lr_finder = LrFinder(estimator, lr_settings=lr_settings)
        lr = lr_finder.get_lr(dataset)
    except torch.cuda.OutOfMemoryError as e:
        memory_cleanup()
        raise e
    return lr


def suggest_lr(
    lrs: Union[torch.Tensor, list, tuple],
    losses: Union[torch.Tensor, list, tuple],
    *,
    min_factor: int = 5,
    burn_in_frac: float = 0.05,
    burn_in_min_pts: int = 3,
    smooth_window: int = 3,
    prefer_last_descent: bool = True,
    fallback_ratio: float = 10.0,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Suggest a learning rate from (lrs, losses) sampled over an exponential LR schedule.
    Numerically-stable version: converts inputs to 1D torch.float64 tensors for computation.
    """

    if isinstance(lrs, torch.Tensor):
        t_lrs = lrs.reshape(-1).to(torch.float64)
    else:
        t_lrs = torch.as_tensor(lrs, dtype=torch.float64).reshape(-1)

    if isinstance(losses, torch.Tensor):
        t_losses = losses.reshape(-1).to(torch.float64)
    else:
        t_losses = torch.as_tensor(losses, dtype=torch.float64).reshape(-1)

    # Put both on the same device
    device = t_losses.device
    if t_lrs.device != device:
        t_lrs = t_lrs.to(device)

    # Promote to float64 for stability (keeps existing values, just higher precision)
    if t_lrs.dtype != torch.float64:
        t_lrs = t_lrs.to(torch.float64)
    if t_losses.dtype != torch.float64:
        t_losses = t_losses.to(torch.float64)

    # --- Basic validation / short inputs ---
    assert t_lrs.shape == t_losses.shape and t_lrs.ndim == 1, (
        "lrs/losses must be 1D and same length"
    )
    n = t_lrs.numel()
    if n < 2:
        out_lr = t_lrs.reshape(-1)[0].item() if n == 1 else 1e-3
        return (out_lr, {"reason": "too_short"}) if return_details else out_lr

    # --- Filter invalid and non-positive LRs ---
    mask = torch.isfinite(t_lrs) & torch.isfinite(t_losses) & (t_lrs > 0)
    t_lrs = t_lrs[mask]
    t_losses = t_losses[mask]
    n = t_lrs.numel()
    if n < 2:
        out_lr = t_lrs.reshape(-1)[0].item() if n == 1 else 1e-3
        return (out_lr, {"reason": "filtered_too_short"}) if return_details else out_lr

    # --- Ensure increasing LR (sort if needed) ---
    order = torch.argsort(t_lrs)
    if not torch.all(order == torch.arange(n, device=device)).item():
        t_lrs = t_lrs[order]
        t_losses = t_losses[order]

    # --- Optional smoothing of losses (moving average with edge padding) ---
    if smooth_window and smooth_window > 1 and n >= smooth_window:
        k = int(smooth_window)
        kernel = torch.ones(1, 1, k, dtype=t_losses.dtype, device=device) / k
        x = t_losses.view(1, 1, -1)
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        xpad = F.pad(x, (pad_left, pad_right), mode="replicate")
        losses_s = F.conv1d(xpad, kernel).view(-1)
    else:
        losses_s = t_losses.clone()

    # --- Global minimum and search window end ---
    gmin = int(torch.argmin(losses_s).item())
    end = max(
        1, gmin - max(0, int(min_factor))
    )  # exclusive upper bound for grad segments
    if end <= 1:
        sug = (t_lrs[gmin] / max(1.0, fallback_ratio)).item()
        details = dict(reason="too_close_to_min", gmin=gmin, end=end)
        return (sug, details) if return_details else sug

    # --- Log-LR axis (clamp away from zero to avoid -inf) ---
    tiny = torch.finfo(t_lrs.dtype).tiny
    log_lr = torch.log(torch.clamp(t_lrs, min=tiny))

    # Finite-difference gradient d(loss)/d(log(lr)) up to 'end'
    dlog = torch.diff(log_lr[:end])
    dL = torch.diff(losses_s[:end])

    # Avoid division by zero if dlog ~ 0 (shouldn't happen for strictly increasing lrs)
    eps = torch.finfo(dlog.dtype).eps
    dlog = torch.where(dlog == 0, torch.full_like(dlog, eps), dlog)
    grad = dL / dlog  # length end-1

    # --- Burn-in skip ---
    start_skip = max(int(burn_in_min_pts), int(math.floor(burn_in_frac * n)))
    start = min(max(0, start_skip), max(0, end - 2))  # ensure start <= end-2

    # --- Prefer last contiguous negative run leading into end-1 ---
    search_lo, search_hi = start, end - 1
    if prefer_last_descent and (search_hi >= search_lo):
        run_end = search_hi
        run_start = run_end
        while run_start > search_lo and (grad[run_start - 1] < 0):
            run_start -= 1
        if run_start < run_end:
            search_lo, search_hi = run_start, run_end

    if search_hi < search_lo:
        sug = (t_lrs[gmin] / max(1.0, fallback_ratio)).item()
        details = dict(reason="empty_search", gmin=gmin, end=end, start=start)
        return (sug, details) if return_details else sug

    window = grad[search_lo : search_hi + 1]
    window_min = torch.min(window)
    window_max = torch.max(window)

    rel_tol = 1e-6
    abs_tol = 1e-12

    range_val = (window_max - window_min).item()
    scale = max(abs(window_min.item()), 1.0)
    if range_val <= abs_tol + rel_tol * scale:
        best_grad_idx = search_hi
    else:
        eps_tie = abs_tol + rel_tol * abs(window_min.item())
        cands = torch.nonzero(window <= window_min + eps_tie, as_tuple=False).flatten()
        best_rel = int(cands[-1].item())
        best_grad_idx = search_lo + best_rel

    pick_idx = min(best_grad_idx + 1, n - 1)
    suggested_lr = float(t_lrs[pick_idx].item())

    if return_details:
        return suggested_lr, dict(
            lrs=t_lrs,
            losses=t_losses,
            losses_s=losses_s,
            log_lr=log_lr,
            grad=grad,
            gmin=gmin,
            end=end,
            start=start,
            search_lo=search_lo,
            search_hi=search_hi,
            best_grad_idx=best_grad_idx,
            pick_idx=pick_idx,
            reason="ok",
        )
    return suggested_lr
