from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset import CustomDataset
from tqdm import tqdm

from display import display_images, create_images
from loss import IntervalMSELoss
from model import SimpleCNN2

def save_model(model: nn.Module):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"models/history/model_{timestamp}.params"
    torch.save(model.state_dict(), model_path)

def load_model() -> nn.Module:
    model = SimpleCNN2()
    model_path = "models/load/best.params"
    model.load_state_dict(torch.load(model_path))
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset(train=True, cache=True)
    val_dataset = CustomDataset(train=False, cache=True)
    num_workers = 0

    batch_size = 64
    val_batch_size = 500

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)

    model = SimpleCNN2()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

    use_lr_scheduler = False
    lr_scheduler = None
    if use_lr_scheduler:
        lr_scheduler = StepLR(optimizer, 30, 0.3)


    epochs = 100
    for epoch in range(epochs):

        train_log = train_epoch(model, train_loader, optimizer, device, epoch)
        val_log = validate_epoch(model, val_loader, device, epoch)

        print(train_log, val_log)

        if lr_scheduler:
            # if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            #     lr_scheduler.step(val_log.get('loss'))
            # else:
            lr_scheduler.step()

        target_val_error = 0.5 / 255 * 28 * 28
        if val_log["loss"] < target_val_error: break

    print('Training complete')

    save_model(model)
    images = create_images(device, model, val_dataset, 10)
    display_images(images)



def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
                epoch: int):
    model.train()
    # loss_criterion = nn.MSELoss(reduction="sum")
    loss_criterion = IntervalMSELoss(0, 1, 2, reduction="sum")

    losses = []
    loss_count = 0

    progress_bar = tqdm(loader, desc=f"Train {epoch}", leave=False)

    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_criterion(out, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loss_count += x.shape[0]

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    avg_loss = np.sum(losses).item() / loss_count

    return {
        "loss": avg_loss,
    }


def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int):
    model.eval()
    # loss_criterion = nn.MSELoss(reduction="sum")
    loss_criterion = nn.L1Loss(reduction="sum")

    losses = []
    loss_count = 0

    pbar = tqdm(loader, desc=f"Val {epoch}", leave=False)

    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = loss_criterion(out, y)
            losses.append(loss.item())
            loss_count += x.shape[0]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

    avg_loss = np.sum(losses).item() / loss_count

    return {
        "loss": avg_loss,
    }

if __name__ == '__main__':
    train()