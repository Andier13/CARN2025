from torch import nn, Tensor


# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3),
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
#             nn.Flatten(),
#             nn.Linear(in_features=28*28, out_features=28*28),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = self.layers(x)
#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, 28, 28)
#         if batch_size == 1:
#             x = x.squeeze(0)
#         return x


# class SimpleLinear(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=3*32*32, out_features=28*28),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = self.layers(x)
#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, 28, 28)
#         if batch_size == 1:
#             x = x.squeeze(0)
#         return x

class SimpleCNN2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=True),
            nn.Flatten(-2, -1),
            nn.Linear(in_features=32*32, out_features=28*28, bias=True),
            # nn.Sigmoid(),
            nn.Unflatten(-1, (28, 28))
        )

    def forward(self, x: Tensor):
        return self.layers(x)

# class SimpleCNN3(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
#             nn.Flatten(),
#             nn.Linear(in_features=28*28, out_features=28*28),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = self.layers(x)
#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, 28, 28)
#         if batch_size == 1:
#             x = x.squeeze(0)
#         return x

# class SimpleCNN4(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=True),
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, bias=True),
#             nn.Flatten(),
#             nn.Linear(in_features=28*28, out_features=28*28, bias=True),
#             # nn.Sigmoid(),
#         )
#
#     def forward(self, x: Tensor):
#         x = self.layers(x)
#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, 28, 28)
#         if batch_size == 1:
#             x = x.squeeze(0)
#         return x