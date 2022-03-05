import torch.nn as nn
import torchvision.models as models


class SSClassifier(nn.Module):
    def __init__(self, out_features=2):
        super(SSClassifier, self).__init__()
        model = models.resnet18(True)
        blocks = list(model.children())[:-1]
        self.fc = nn.Linear(in_features=512, out_features=out_features)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        x = self.fc(x.flatten(1))
        return x
