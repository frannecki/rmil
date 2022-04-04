import torch
from ..model import Classifier


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)  # Make sure backward would be called

    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DomainClassifier(Classifier):
    def forward(self, x):
        x = self.feature_extractor(x)
        x = GradientReverse.apply(x)
        x = self.mlp(x)
        return x


def build_da_model(args, backbone, backbone_out_features):
    r"""Domain adversarial training model builder"""
    return DomainClassifier(backbone,
                            backbone_out_features,
                            args.avgpool_size)
