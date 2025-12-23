import torch

from dataset import CustomDataset
from display import create_images, save_images, display_images
from train import load_model

if __name__ == '__main__':
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(train=False, cache=True)
    images = create_images(device, model, test_dataset, 6)
    save_images(images, "compare.png")
    display_images(images)