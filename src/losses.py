# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            raise ValueError("El parámetro alpha debe ser un escalar, lista, tupla o tensor.")

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        if len(self.alpha) > 1:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha[0]

        F_loss = alpha_t * (1 - pt).pow(self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class Losses:

    def __init__(self):
        self.losses = {
            "focal_loss": FocalLoss,
            "cross_entropy": nn.CrossEntropyLoss
        }

    def get_loss(self, loss_name, **kwargs):
        if loss_name not in self.losses:
            raise ValueError(f"La pérdida '{loss_name}' no está registrada. "
                             f"Pérdidas disponibles: {list(self.losses.keys())}")
        return self.losses[loss_name](**kwargs)


if __name__ == "__main__":
    criterion = Losses().get_loss("focal_loss", alpha=[1.0, 2.0], gamma=2, reduction="mean")
    logits = torch.tensor([[2.0, 0.5], [0.3, 3.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = criterion(logits, targets)
    print("Pérdida focal calculada:", loss.item())
