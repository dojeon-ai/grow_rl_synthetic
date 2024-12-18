import torch.nn as nn
import torch
from einops import rearrange
from src.models.backbones.base import BaseBackbone


class Nature(BaseBackbone):
    name = 'nature'
    def __init__(self, in_shape):
        super().__init__(in_shape)
        c, h, w = in_shape
        in_channels = c

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
        )
            
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.layers(x)
            
        return x