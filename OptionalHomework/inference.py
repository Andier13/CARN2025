from multiprocessing import freeze_support
from timed_decorator.simple_timed import timed

import torch
import wandb
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset

from dataset import CustomDataset
from train import load_model


@timed(return_time=True, use_seconds=True, stdout=False)
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset.tensors[0]:
        transforms(image)


@timed(return_time=True, use_seconds=True, stdout=False)
@torch.no_grad()
def transform_dataset_with_model(dataset: TensorDataset, model: nn.Module, device: torch.device, batch_size: int, num_workers: int, pin_memory: bool):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)  # TODO: Complete the other parameters
    for images, in dataloader:
        images = images.to(device, non_blocking=pin_memory)
        model(images)
        pass


def test_inference_time(model: nn.Module, device: torch.device, batch_size: int, num_workers: int, pin_memory: bool) -> tuple[float, float]:
    test_dataset = CustomDataset(train=False, cache=False)
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    # batch_size = 1  # TODO: add the other parameters (device, ...)

    model.to(device)
    _, t1 = transform_dataset_with_transforms(test_dataset)
    _, t2 = transform_dataset_with_model(test_dataset, model, device, batch_size, num_workers, pin_memory)
    print(f"Sequential transforming each image took: {t1}s on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2}s on {device.type}. \n")
    return t1, t2

sweep_config = {
    "method": "bayes",
    "metric": {"name": "inference_time", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [1, 4, 8, 16, 32, 64, 128]},
        "num_workers": {"values": [0, 2, 4]},
        "pin_memory": {"values": [True, False]},
    },
}

def run_inference_sweep():
    wandb.init()
    cfg = wandb.config

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = load_model()

    sequential_time, inference_time = test_inference_time(
        model=model,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    wandb.log({
        "sequential_time": sequential_time,
        "inference_time": inference_time,
    })


if __name__ == "__main__":
    freeze_support()
    sweep_id = wandb.sweep(sweep_config, project="inference-speed-test")
    wandb.agent(sweep_id, function=run_inference_sweep, count=20)
