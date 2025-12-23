from multiprocessing import freeze_support
from timed_decorator.simple_timed import timed

import torch
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


def main():
    model = load_model()
    test_inference_time(model, torch.device("cpu"), batch_size=64, num_workers=0, pin_memory=True)
    test_inference_time(model, torch.device("cuda"), batch_size=64, num_workers=0, pin_memory=True)



if __name__ == '__main__':
    freeze_support()
    main()