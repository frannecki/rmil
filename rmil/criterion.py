import torch
import torch.nn as nn


class BinBCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        targ_bin = torch.zeros_like(inputs)
        for i in range(targets.shape[0]):
            targ_bin[i, :targets[i]] = 1
        targ_bin = targ_bin.to(self.device)
        loss = self.criterion(inputs, targ_bin)
        return loss


class GradeLoss(nn.Module):
    def __init__(self):
        super(GradeLoss, self).__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).sum(1)
        return torch.mean((probs - targets) ** 2)


class GradedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GradedCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).sum(1) * 3 / 4
        cross_entropy_loss = self.ce(logits, targets)
        graded_loss = torch.mean((probs - targets) ** 2)
        return 0.1 * cross_entropy_loss + graded_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: float):
        super(CrossEntropyLoss, self).__init__()
        self.w = weight # weight

    def forward(self, logits, targets):
        return self.w * super().forward(logits, targets)
