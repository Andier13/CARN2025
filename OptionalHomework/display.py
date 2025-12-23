import torch
import cv2
import numpy as np
from torch import nn
from torch.utils.data import Dataset

def create_images(device: torch.device, model: nn.Module, val_dataset: Dataset, count: int):
    model.to(device)
    model.eval()

    all_images = []
    for i in range(count):
        x, y = val_dataset[i]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            prediction = model(x.unsqueeze(0))

        y = y.squeeze().unsqueeze(0)
        prediction = prediction.squeeze().unsqueeze(0)

        def to_image(x):
            image = np.round(x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image = cv2.resize(image, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
            return image

        x = to_image(x)
        y = to_image(y)
        prediction = to_image(prediction)

        def pad(x):
            padding = (32 - 28) // 2 * 12
            return cv2.copyMakeBorder(x, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

        y = pad(y)
        prediction = pad(prediction)

        diff = np.abs(prediction.astype(np.int16) - y.astype(np.int16))
        diff = (diff / diff.max() * 255).astype(np.uint8)

        compare_images = cv2.hconcat([x, y, prediction, diff])

        all_images.append(compare_images)

    return all_images

def save_images(images: list[np.ndarray], filename: str):
    path = f"images/{filename}"
    image = cv2.vconcat(images)
    cv2.imwrite(path, image)

def display_images(images: list[np.ndarray]):
    for i, image in enumerate(images):
        cv2.imshow(f"Image {i}", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
