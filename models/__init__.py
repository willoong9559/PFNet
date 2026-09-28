from .base import Base
from .pfnet_polar import PFNet

__all__ = {
    'Base': Base,
    'PFNet': PFNet
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
