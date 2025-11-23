import torch.nn as nn
import timm


class SimpleMLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get('name')
    num_classes = cfg.get('num_classes')
    image_size = cfg.get('image_size')
    pretrained = cfg.get('pretrained', False)

    if name in ['resnet18', 'resnet50', 'resnest14d', 'resnest26d']:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    elif name == 'MLP':
        in_features = cfg.get('channels', 3) * (image_size ** 2)
        model = SimpleMLP(in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')

    return model
