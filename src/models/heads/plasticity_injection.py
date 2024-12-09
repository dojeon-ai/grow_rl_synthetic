from abc import *
import copy
import torch
import torch.nn as nn
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation

class PairedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: int,
        out_dim: int,
        activ_type: int,
        drop_prob: int,
        disable_pair: bool = False, 
    ):
        
        super().__init__()
        self.disable_pair = disable_pair
        self.activ = init_activation(activ_type=activ_type)
        in_dims = [in_dim, *hidden_dims]
        out_dims = [*hidden_dims, out_dim]
        layers = []
        for h_in, h_out in zip(in_dims, out_dims):
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.activ)
            layers.append(nn.Dropout(p=drop_prob))
        layers = layers[:-2] # cut last activation & dropout
        self.fc = nn.Sequential(*layers)

        if not self.disable_pair:
            # Create frozen pair
            self.pair_fc = copy.deepcopy(self.fc)
            self.pair_fc.load_state_dict(self.fc.state_dict())
            for param in self.pair_fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.disable_pair:
            return self.fc(x)
        else:
            return self.fc(x) - self.pair_fc(x)

    def freeze(self):
        # Freeze main MLP
        for param in self.fc.parameters():
            param.requires_grad = False

    def get_param_count(self):
        return sum(p.numel() for p in self.fc.parameters())
        

class PlasticityInjection(BaseHead):
    name = 'plasticity_injection'
    def __init__(
        self, 
        max_injections,
        in_shape, 
        output_size,
        norm_type,
        activ_type,
        drop_prob,
        hidden_dims
    ):

        super().__init__(in_shape, output_size)
        self.max_idx = max_injections
        c, h, w = self.in_shape
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        in_channel = c * h * w

        self.paired_MLPs = nn.ModuleList([
            PairedMLP(
                in_dim=in_channel,
                hidden_dims=hidden_dims,
                out_dim=output_size,
                activ_type=activ_type,
                drop_prob=drop_prob,
                disable_pair=pair_idx==0, 
            )
            for pair_idx in range(max_injections+1)
        ])
        self.active_idx = 0

    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)

        xs = [self.paired_MLPs[i](x) for i in range(self.active_idx+1)]
        x = torch.stack(xs, dim=-1).sum(dim=-1)
        info = {}

        return x, info

    def intervene(self):
        prev_idx = self.active_idx
        self.active_idx = min(self.max_idx, self.active_idx+1)
        if self.active_idx > prev_idx:
            self.paired_MLPs[prev_idx].freeze()

    def count_active_params(self):
        return self.paired_MLPs[0].get_param_count() * (2*self.active_idx+1)
