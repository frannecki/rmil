r"""Definition of models"""
import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50
from .resnet import _resnet, BasicBlock
from torchvision.models.vgg import vgg11


def resnet_mini(pretrained):
    return _resnet("resnet_mini", BasicBlock, [1, 1, 1, 1],
                   pretrained, progress=False)


RESNET_BLOCKS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet_mini': resnet_mini
}


class Classifier(nn.Module):
    r"""Vanilla DNN classifier"""
    def __init__(self,
                 backbone,
                 backbone_out_features,
                 avgpool_size,
                 *out_features):
        super(Classifier, self).__init__()
        self.feature_extractor = backbone
        self.mlp = MLP(avgpool_size, backbone_out_features, *out_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mlp(x)
        return x


class ResnetBackbone(nn.Module):
    r"""Resnet backbone"""
    def __init__(self, variant, in_channels, pretrained=True):
        super(ResnetBackbone, self).__init__()
        assert variant in RESNET_BLOCKS.keys(), \
            f"ResnetBackbone for {variant} not implemented"
        block = RESNET_BLOCKS[variant]
        resnet = block(pretrained=pretrained)
        if in_channels != 3:
            resnet._modules['conv1'] = nn.Conv2d(
                in_channels=in_channels, out_channels=64,
                kernel_size=7, stride=2, padding=3)
        feature_extractor_layers_names = list(resnet._modules)[:-2]
        self.feature_extractor = nn.Sequential(
            *[getattr(resnet, layer_name) for
              layer_name in feature_extractor_layers_names])

    def forward(self, x):
        return self.feature_extractor(x)


class VGGBackbone(nn.Module):
    r"VGG backbone"
    def __init__(self, pretrained=True):
        super(VGGBackbone, self).__init__()
        vgg = vgg11(pretrained)
        self.feature_extractor = vgg.features[:-5]

    def forward(self, x):
        return self.feature_extractor(x)


class MLP(nn.Module):
    r"""Multiple-layer perceptron (MLP) Classifier
    that accepts feature maps and returns probs"""
    def __init__(self, avgpool_size, in_feature: int, *out_features):
        r"""MLP classifier constructor

        Args:
            avgpool_size: output size of average pooling size
            in_feature: dimension of the feature vector accepted
                by fully connected layer
            out_features: dimensions of the feature vector outputs
        """
        super(MLP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        layers = []
        for out_feature in out_features[:-1]:
            layers.append(nn.Linear(in_feature, out_feature))
            layers.append(nn.ReLU())
            in_feature = out_feature
        layers.append(nn.Linear(in_feature, out_features[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class GatedAttention(nn.Module):
    r"""Gated Attention Module

    This classifier performs self-attention on a group of
    image instance feature maps (patch)

    for k-th instance, its weight is calculated as follows
    $$
    \begin{equation*}
        \alpha_k = \frac{\exp{\{ \pmb{w}^T [\tanh{(\pmb{V}\pmb{h}_k^T)}
        \odot \sigma{(\pmb{U}\pmb{h}_k^T)} ] \}}} {\sum_i^K \exp{\{ \pmb{w}^T
        [\tanh{(\pmb{V}\pmb{h}_i^T)} \odot \sigma{(\pmb{U}\pmb{h}_i^T)}] \}}}
    \end{equation*}
    $$
    where $h_k$ is the corresponding feature vector of size $M$, $\pmb{U}$ and
    $\pmb{V}$ are fully connected layer weights of size $L \times M$, and
    $\pmb{w}$ is a one-dimensional vector of size $L$
    """
    def __init__(self, attn_features: int,
                 backbone_out_features: int, avgpool_size: int):
        r"""Gated Attention Module constructor

        Args:
            attn_features: query and key dimension for feature vectors $L$
            backbone_out_features: dimension of output feature vectors
                from the feature extractor
            avgpool_size: output size of feature map average pooling before
                feeding into self-attention module
        """
        super(GatedAttention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        attn_in_features = backbone_out_features * avgpool_size * avgpool_size
        self.query = nn.Sequential(
            nn.Linear(in_features=attn_in_features,
                      out_features=attn_features), nn.Tanh())
        self.key = nn.Sequential(
            nn.Linear(in_features=attn_in_features,
                      out_features=attn_features), nn.Sigmoid())
        self.attn = nn.Linear(in_features=attn_features, out_features=1)
        self.attn_weight = nn.Softmax(dim=1)

    def forward(self, x):
        h = x.view(-1, *x.shape[2:])
        h = self.avgpool(h)
        h = h.view(x.size(0), x.size(1), -1)
        query = self.query(h)
        key = self.key(h)
        weights = self.attn(query * key).unsqueeze(-1).unsqueeze(-1)
        alpha = self.attn_weight(weights)
        return torch.sum(alpha * x, dim=1)


class AttentionMIL(nn.Module):
    r"""Multi instance learning based on vanilla resnet18 classifier
    and self-attention module
    """
    def __init__(self, backbone, attn_block, avgpool_size,
                 backbone_out_features: int, out_features: int):
        r"""Constructor for multiple-instance learning model
        with gated attention

        Args:
            backbone: backbone feature extractor block
            attn_block: attention module
            avgpool_size: output size for average pooling
            backbone_out_features: dimension of output feature vectors
                from the feature extractor
            out_features: dimension of output feature vectors
        """
        super(AttentionMIL, self).__init__()
        self.feature_extractor = backbone
        self.attn = attn_block
        self.mlp = MLP(avgpool_size, backbone_out_features, out_features)
        self.with_attn = attn_block is not None

    def forward(self, X):
        x = X.view(-1, *X.shape[2:])
        x = self.feature_extractor(x)
        x = x.view(*X.shape[:2], *x.shape[1:])
        if self.with_attn:
            x = self.attn(x)
        else:
            x = torch.mean(x, dim=1)
        x = self.mlp(x)
        return x

    def fine_tune(self, tune=True):
        for param in self.feature_extractor.parameters():
            param.requires_grad = not tune


def build_feature_backbone(args):
    r"""Feature extractor backbone builder"""
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    backbone = ResnetBackbone(args.backbone, 3, args.pretrained)
    if args.backbone == 'resnet50':
        backbone_out_features = 2048
    else:
        backbone_out_features = 512
    return backbone, backbone_out_features


def build_attn_mil(args, backbone, backbone_out_features):
    r"""Attention multiple instance learning model builder"""
    attn_block = None
    if args.attn:
        attn_block = GatedAttention(
            attn_features=64,
            backbone_out_features=backbone_out_features,
            avgpool_size=args.avgpool_size_attn)
    return AttentionMIL(backbone, attn_block, args.avgpool_size,
                        backbone_out_features, args.out_features)


def build_naive_model(args, backbone, backbone_out_features, out_features):
    r"""Naive DNN model builder for image classification"""
    return Classifier(backbone, backbone_out_features,
                      args.avgpool_size, out_features)
