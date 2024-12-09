from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: int,
        out_dim: int,
        activ_type: int,
        drop_prob: int, 
    ):
        super().__init__()
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

    def forward(self, x):
        return self.fc(x)

class BigDynamicMoE(BaseHead):
    name = 'big_dynamic_moe'
    def __init__(
        self,
        initial_experts,
        max_experts,
        grow_increment,
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
        
        self.curr_experts = initial_experts
        self.max_experts = max_experts
        self.grow_increment = grow_increment
        self.experts = nn.ModuleList([
            MLP(        
                in_dim=in_channel,
                hidden_dims=hidden_dims,
                out_dim=output_size,
                activ_type=activ_type,
                drop_prob=drop_prob, 
            )
            for _ in range(max_experts)
        ])
        self.routers = nn.ModuleList([nn.Linear(in_channel, 1) for _ in range(max_experts)])
        
    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        
        logits = torch.stack(
            [self.routers[i](x) for i in range(self.curr_experts)], dim=1
        )
        expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.curr_experts)], dim=1
        )
        probs = F.softmax(logits, dim=1)
        out = torch.sum(expert_outputs*probs, axis=1)
        info = {}
        return out, info

    def intervene(self):
        print(self.curr_experts, 'to', min(self.curr_experts+self.grow_increment, self.max_experts))
        self.curr_experts = min(self.curr_experts+self.grow_increment, self.max_experts)

    def count_active_params(self):
        param_per_expert = (
            sum(p.numel() for p in self.experts[0].parameters())
            + sum(p.numel() for p in self.routers[0].parameters())
        )
        return param_per_expert * self.curr_experts