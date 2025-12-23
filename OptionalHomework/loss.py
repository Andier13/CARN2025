from torch import nn, Tensor


class IntervalMSELoss(nn.Module):

    def __init__(self, lower_bound: float, upper_bound: float, weight: float, reduction: str = "mean"):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.weight = weight
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor):
        out_of_bounds = (prediction < self.lower_bound) | (prediction > self.upper_bound)
        weights = out_of_bounds * (self.weight - 1) + 1
        return nn.functional.mse_loss(prediction, target, reduction=self.reduction, weight=weights)
