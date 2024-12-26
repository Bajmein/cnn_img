import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean') -> None: #fixme probar luego con gamma en 3
        super(FocalLoss, self).__init__()
        self.alpha: int = alpha
        self.gamma: int = gamma
        self.reduction: str = reduction

    def forward(self, inputs, targets) -> torch.Tensor:
        BCE_loss: torch.Tensor = F.cross_entropy(inputs, targets, reduction='none')
        pt: torch.Tensor = torch.exp(-BCE_loss)
        F_loss: torch.Tensor = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
