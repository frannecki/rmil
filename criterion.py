import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, num_classes, device, alpha=None,
                 gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.ones(num_classes).to(device)
        if alpha is not None:
            self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, inputs, targets):
        N, C = tuple(inputs.size()[:2])
        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1).long()
        class_mask.data.scatter_(1, ids.data, 1.)

        probs = F.softmax(inputs, dim=1)
        probs = (probs * class_mask).sum(1).view(-1, 1)
        batch_loss = -((1-probs) ** self.gamma) * torch.log(probs + 1e-5)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
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
        probs = torch.sigmoid(logits).sum(1)
        cross_entropy_loss = self.ce(logits, targets)
        graded_loss = torch.mean((probs - targets) ** 2)
        return 0.1 * cross_entropy_loss + graded_loss
