from . import config
from ..model import Classifier


class SslClassifier(Classifier):
    pass


def build_ssl_model(args, backbone, backbone_out_features):
    r"""Self supervised learning model builder"""
    return SslClassifier(backbone,
                         backbone_out_features,
                         args.avgpool_size,
                         len(config.IMAGE_SIZES))
