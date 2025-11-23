import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


def build_transforms(cfg: dict, mean: list[float], std: list[float], image_size: int, pretrained: bool = False):
    aug = cfg.get('augmentations', {})

    cached_train_transforms = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    cached_val_transforms = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std, inplace=True),
    ]

    padding = image_size // 8
    train_transforms = []

    if aug.get('random_horizontal_flip', False):
        train_transforms.append(v2.RandomHorizontalFlip())
    if aug.get('random_crop', False):
        train_transforms.append(v2.RandomCrop(image_size, padding=padding, fill=mean))
    if aug.get('color_jitter', False):
        train_transforms.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    train_transforms.append(v2.Normalize(mean, std, inplace=True))

    if aug.get('random_erasing', False):
        train_transforms.append(v2.RandomErasing(p=0.5, scale=(0.01, 0.15), ratio=(0.3, 3.3), inplace=True))

    return v2.Compose(cached_train_transforms), v2.Compose(train_transforms), v2.Compose(cached_val_transforms)


def get_dataset_mean_and_std(name: str) -> tuple[list[float], list[float]]:
    if name == "mnist":
        mean = [0.1307]
        std = [0.3081]
    elif name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif name == "cifar100":
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    elif name in ["oxfordiiitpet", "oxford_iiit_pet"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return mean, std


class CachedTrainDataset(Dataset):
    def __init__(self, data: Dataset, transform ):
        self.data = [(x, y) for x, y in data]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.transform(self.data[i])

class CachedValDataset(Dataset):
    def __init__(self, data: Dataset, image_size: int):
        padding_size = image_size // 16
        self.data = [
            (v2.functional.pad(x, [padding_size], fill=0.0),
            v2.functional.hflip(v2.functional.pad(x, [padding_size], fill=0.0)),
            y)
            for x, y in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def build_dataset(cfg: dict, pretrained: bool = False) -> tuple[Dataset, Dataset]:
    name = cfg.get('name').lower()
    data_dir = cfg.get('data_dir', './data')
    mean, std = get_dataset_mean_and_std(name)
    image_size = cfg.get('image_size', 224) if not pretrained else 224
    cached_train_transform, train_transform, cached_val_transform = build_transforms(cfg, mean, std, image_size, pretrained=pretrained)

    if name == "mnist":
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=cached_train_transform
        )
        val_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=cached_val_transform
        )

    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=cached_train_transform
        )
        val_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=cached_val_transform
        )

    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=cached_train_transform
        )
        val_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=cached_val_transform
        )

    elif name in ["oxfordiiitpet", "oxford_iiit_pet"]:
        train_dataset = datasets.OxfordIIITPet(
            data_dir,
            split="trainval",
            target_types="category",
            download=True,
            transform=cached_train_transform,
        )
        val_dataset = datasets.OxfordIIITPet(
            data_dir,
            split="test",
            target_types="category",
            download=True,
            transform=cached_val_transform,
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return CachedTrainDataset(train_dataset, train_transform), CachedValDataset(val_dataset, image_size)


def build_dataloaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, val_batch_size: int,
                     num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    train_loader = build_train_dataloader(train_dataset, batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader

def build_train_dataloader(train_dataset: Dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
               num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True,
               drop_last=True)



