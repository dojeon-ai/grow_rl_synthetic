from .base import BaseHead
from .gap import GAPHead
from .spatial_pool import SpatialPoolHead
from .mlp import MLPHead
from .dynamic_moe import DynamicMoE
from .dynamic_moe_big import BigDynamicMoE
from .plasticity_injection import PlasticityInjection
from .expanding_mlp import ExpandingMLP
from .progressive_net import ProgressiveNet

__all__ = [
    'BaseHead', 
    'SpatialPoolHead', 
    'GAPHead',
    'MLPHead',
    'DynamicMoE',
    'BigDynamicMoE',
    'PlasticityInjection',
    'ExpandingMLP',
    'ProgressiveNet',
]