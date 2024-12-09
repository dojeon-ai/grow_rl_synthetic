from abc import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation

NEG_INF = -1e8

class ExpandingLinear(nn.Module):
    def __init__(
        self,
        total_expansions: int,
        h_in: int,
        h_out: int,
        fixed_h_in: bool,
        fixed_h_out: bool,
    ):
        super().__init__()
        self.expand_ratio = 1.0 / (total_expansions + 1)
        self.current_idx = 1
        self.max_idx = total_expansions + 1
        self.h_in = h_in
        self.h_out = h_out
        self.fixed_h_in = fixed_h_in
        self.fixed_h_out = fixed_h_out

        # init weights by torch default
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        self.weight = nn.Parameter(torch.empty(h_in, h_out))
        self.bias = nn.Parameter(torch.empty(h_out,))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def get_in_out(self):
        if self.fixed_h_in or self.current_idx == self.max_idx:
            h_in = self.h_in
        else:
            h_in = int(self.h_in * self.expand_ratio * self.current_idx)
        if self.fixed_h_out or self.current_idx == self.max_idx:
            h_out = self.h_out
        else:
            h_out = int(self.h_out * self.expand_ratio * self.current_idx)
        return h_in, h_out

    def forward(self, x):
        h_in, h_out = self.get_in_out()
        w = self.weight[:h_in,:h_out]
        b = self.bias[:h_out]    
        x = x@w+b
        return x

    def expand(self):
        prev_h = self.get_in_out()
        self.current_idx = min(self.max_idx, self.current_idx+1)
        new_h = self.get_in_out()
        print(prev_h, "->", new_h)

    def count_active_params(self):
        # manually compute params
        h_in, h_out = self.get_in_out()
        return h_in*h_out + h_out

class ExpandingMLP(BaseHead):
    name = 'expanding_mlp'
    def __init__(
        self,
        total_expansions,
        in_shape,
        output_size,
        norm_type,
        activ_type,
        drop_prob,
        hidden_dims
    ):
        
        super().__init__(in_shape, output_size)
        c, h, w = self.in_shape
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.activ = init_activation(activ_type=activ_type)
        in_channel = c * h * w
        
        layers = []
        hidden_dims = [in_channel] + hidden_dims + [output_size]
        for idx in range(len(hidden_dims)-1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx+1]
            layers.append(
                ExpandingLinear(
                    total_expansions=total_expansions,
                    h_in=h_in,
                    h_out=h_out,
                    fixed_h_in=(idx==0),
                    fixed_h_out=(idx==len(hidden_dims)-2),
                )
            )
            layers.append(self.activ)
            layers.append(nn.Dropout(p=drop_prob))
        layers = layers[:-2]
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc(x)
        info = {}
        return x, info

    def intervene(self):
        for layer in self.fc:
            if isinstance(layer, ExpandingLinear):
                layer.expand()

    def count_active_params(self):
        cnt = 0
        for layer in self.fc:
            if isinstance(layer, ExpandingLinear):
                cnt += layer.count_active_params()
        return cnt