import numpy as np
import torch
import yaml
from pathlib import Path

from torch import GradScaler, nn, cuda, autocast
import torch.jit as jit
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

import wandb

from datasets import build_dataset, build_dataloaders, build_train_dataloader
from models import build_model
from optimizers import build_optimizer, SAM
from utils import (
    EarlyStopper,
    BatchSizeScheduler,
    get_device, build_lr_scheduler,
)


def main():
    cfg_path = Path("configs/current_config.yaml")
    cfg: dict = yaml.safe_load(open(cfg_path))

    if cfg.get('wandb'):
        wandb_cfg = cfg.get('wandb', {})
        wandb.init(project=wandb_cfg.get('project', 'pytorch-pipeline'),
                   entity=wandb_cfg.get('entity', None),
                   config=cfg)
        cfg = wandb.config


    train_cfg = cfg.get('training', {})
    torch.backends.cudnn.benchmark = True
    device = get_device(train_cfg.get('device', 'auto'))

    model_cfg = cfg.get('model')
    dataset_cfg = cfg.get('dataset')
    image_size = dataset_cfg.get('image_size', 224)
    train_dataset, val_dataset = build_dataset(dataset_cfg, model_cfg.get("pretrained"))
    num_workers = dataset_cfg.get('num_workers', 0)

    train_cfg = cfg.get('training')
    initial_batch_size = train_cfg.get('batch_size')
    val_batch_size = train_cfg.get('val_batch_size', 500)
    batch_size_scheduler_cfg = train_cfg.get('batch_size_scheduler')
    batch_size_scheduler = None
    if batch_size_scheduler_cfg:
        batch_size_scheduler = BatchSizeScheduler(initial_batch_size, batch_size_scheduler_cfg)

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, initial_batch_size, val_batch_size,
                                                 num_workers)

    num_classes = model_cfg.get("num_classes")
    model = build_model(model_cfg)
    model.to(device)
    model = jit.script(model)

    optimizer_cfg = cfg.get('optimizer', {})
    optimizer = build_optimizer(optimizer_cfg, model)

    lr_scheduler_cfg = cfg.get('scheduler')
    lr_scheduler = None
    if lr_scheduler_cfg:
        lr_scheduler = build_lr_scheduler(lr_scheduler_cfg, optimizer)

    enable_half = device.type != "cpu"
    scaler = GradScaler(device.type, enabled=enable_half)

    early_stopping_cfg = train_cfg.get('early_stopping')
    early_stopper = None
    early_stopper_monitor = 'loss'
    if early_stopping_cfg:
        early_stopper_monitor = early_stopping_cfg.get('monitor', 'loss')
        mode = 'min' if early_stopper_monitor == 'loss' else 'max'
        patience = early_stopping_cfg.get('patience', 10)
        early_stopper = EarlyStopper(patience=patience, mode=mode)

    current_batch_size = initial_batch_size
    epochs = train_cfg.get('epochs')
    for epoch in range(epochs):

        if batch_size_scheduler:
            new_batch_size = batch_size_scheduler.get_batch_size(epoch)
            if new_batch_size != current_batch_size:
                train_loader = build_train_dataloader(train_dataset, new_batch_size, num_workers)
                current_batch_size = new_batch_size

        train_log = train_epoch(model, train_loader, optimizer, scaler, enable_half, device, epoch, num_classes)
        val_log = validate_epoch(model, val_loader, enable_half, device, epoch, image_size)

        if lr_scheduler:
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_log.get('loss'))
            else:
                lr_scheduler.step()

        if early_stopper:
            monitored_value = val_log.get('loss') if early_stopper_monitor == 'loss' else val_log.get('accuracy')
            if early_stopper.step(monitored_value):
                break

        if cfg.get('wandb'):
            wandb.log({
                **{f"Train {k}": v for k, v in train_log.items()},
                **{f"Validation {k}": v for k, v in val_log.items()}
            })

    print('Training complete')

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler,
                enable_half: bool, device: torch.device, epoch: int, num_classes: int):
    model.train()
    loss_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    correct = 0
    total = 0
    losses = []

    is_sam = isinstance(optimizer, SAM)
    cutmix_or_mixup = v2.RandomChoice([
        v2.CutMix(num_classes=num_classes),
        v2.MixUp(num_classes=num_classes),
        v2.Identity(),
    ])

    progress_bar = tqdm(loader, desc=f"Train {epoch}", leave=False)

    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        mixed_x, mixed_y = cutmix_or_mixup(x, y)

        MB = 1024 ** 2
        current_vram = cuda.memory_allocated(device) / MB

        if is_sam:

            optimizer.zero_grad()
            with autocast(device.type, enabled=enable_half):
                out = model(mixed_x)
                loss = loss_criterion(out, mixed_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            optimizer.first_step(zero_grad=True)

            with autocast(device.type, enabled=enable_half):
                out2 = model(input)
                loss2 = loss_criterion(out2, mixed_y)

            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            optimizer.second_step(zero_grad=True)

            scaler.step(optimizer)
            scaler.update()

        else:

            optimizer.zero_grad()
            with autocast(device.type, enabled=enable_half):
                out = model(mixed_x)
                loss = loss_criterion(out, mixed_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())

        predictions = out.argmax(1)
        targets = mixed_y.argmax(1) if mixed_y.ndim > 1 else y

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "VRAM (MB)": f"{current_vram:.0f}",
        })


    avg_loss = np.mean(losses).item()
    acc = correct / total

    avg_vram = float(cuda.max_memory_allocated(device) / MB)
    cuda.reset_peak_memory_stats(device)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "VRAM (MB)": avg_vram,
    }

def validate_epoch(model: nn.Module, loader: DataLoader, enable_half: bool, device: torch.device, epoch: int, image_size: int):
    model.eval()
    loss_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    correct = 0
    total = 0
    losses = []

    pbar = tqdm(loader, desc=f"Val {epoch}", leave=False)

    with torch.no_grad():
        for x1, x2, y in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            aux = x1[:, :, 0:0 + image_size, 0:0 + image_size]
            flipped_aux = x2[:, :, 0:0 + image_size, 0:0 + image_size]

            with autocast(device.type, enabled=enable_half):
                out = model(aux)
                loss = loss_criterion(out, y)
                out += model(flipped_aux)

            padding_size = image_size // 16
            for i in [-padding_size, 0, padding_size]:
                for j in [-padding_size, 0, padding_size]:
                    if i == 0 and j == 0:
                        continue
                    ii = padding_size + i
                    jj = padding_size + j
                    aux = x1[:, :, ii:ii + image_size, jj:jj + image_size]
                    flipped_aux = x2[:, :, ii:ii + image_size, jj:jj + image_size]
                    with autocast(device.type, enabled=enable_half):
                        out += model(aux)
                        out += model(flipped_aux)

            losses.append(loss.item())
            targets = y
            predictions = out.argmax(1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()


    avg_loss = np.mean(losses).item()
    acc = correct / total

    MB = 1024 ** 2
    avg_vram = float(cuda.max_memory_allocated(device) / MB)
    cuda.reset_peak_memory_stats(device)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "VRAM (MB)": avg_vram,
    }


if __name__ == '__main__':
    main()
