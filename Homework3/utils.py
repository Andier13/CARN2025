
import torch
from torch import optim


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    return torch.device(device_name)

class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.num_bad_epochs = 0

        if mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

    def step(self, value: float) -> bool:
        improved = (value < self.best) if self.mode == "min" else (value > self.best)

        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs > self.patience


class BatchSizeScheduler:
    def __init__(self, base_batch_size: int, cfg: dict):
        self.base_batch_size = base_batch_size
        self.policy = cfg.get("policy", "linear_increase")
        self.max_batch_size = cfg.get("max_batch_size", base_batch_size)
        self.start = cfg.get("start_epoch", 0)
        self.end = cfg.get("end_epoch", 10)
        self.step_size = cfg.get("step", 10)
        self.factor = cfg.get("factor", 2)

    def get_batch_size(self, epoch: int) -> int:
        if self.policy == "linear_increase":
            if epoch <= self.start:
                return self.base_batch_size
            if epoch >= self.end:
                return self.max_batch_size
            t = (epoch - self.start) // self.step_size * self.step_size / (self.end - self.start)
            return int(self.base_batch_size + t * (self.max_batch_size - self.base_batch_size))

        elif self.policy == "step":
            return int(self.base_batch_size * (self.factor ** (epoch // self.step_size)))

        else:
            return self.base_batch_size


def build_lr_scheduler(lr_scheduler_cfg: dict, optimizer: torch.optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    name = lr_scheduler_cfg.get('name')
    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_cfg.get('step_size'),
                                         gamma=lr_scheduler_cfg.get('factor', 0.1))
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=lr_scheduler_cfg.get('factor', 0.1),
                                                    patience=lr_scheduler_cfg.get('patience', 10))
    else:
        raise ValueError(f"Unknown learning rate scheduler: {name}")

