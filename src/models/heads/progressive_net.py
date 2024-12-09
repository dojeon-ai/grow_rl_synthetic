from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation

NEG_INF = -1e8

class MLPColumn(nn.Module):
    def __init__(
        self,
        num_prev_columns: int,
        scaler_init_range: float,
        in_dim: int,
        hidden_dims: int,
        out_dim: int,
        activ_type: int,
        drop_prob: int,
    ):
        
        super().__init__()
        self.activated = True
        self.num_prev_columns = num_prev_columns
        self.activ = init_activation(activ_type=activ_type)
        in_dims = [in_dim, *hidden_dims]
        out_dims = [*hidden_dims, out_dim]
        first_in_dim = in_dims.pop(0)
        first_out_dim = out_dims.pop(0)
        self.embed = nn.Linear(first_in_dim, first_out_dim)
        self.main_stems = nn.ModuleList([
            nn.Linear(h_in, h_out)
            for h_in, h_out in zip(in_dims, out_dims)
        ])
        if num_prev_columns > 0:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(h_in*num_prev_columns, h_in),
                    self.activ,
                    nn.Linear(h_in, h_out)
                )
                for h_in, h_out in zip(in_dims, out_dims)
            ])
            scalers = []
            for h_in in in_dims:
                scaler = nn.Parameter(torch.empty(h_in*num_prev_columns,))
                init.uniform_(scaler, -scaler_init_range, scaler_init_range)
                scalers.append(scaler)
            self.scalers = nn.ParameterList(scalers)

    def forward(self, x, prev_features):
        assert self.num_prev_columns==0 or (len(prev_features) == len(self.main_stems))

        features = []
        x = self.embed(x)
        x = self.activ(x)
        features.append(x)
        for i in range(len(self.main_stems)-1):
            x = self.main_stems[i](x)
            if self.num_prev_columns > 0:
                x = x + self.adapters[i](self.scalers[i]*prev_features[i])
            x = self.activ(x)
            features.append(x)

        if self.activated:
            i = -1
            x = self.main_stems[i](x)
            if self.num_prev_columns > 0:
                x = x + self.adapters[i](self.scalers[i]*prev_features[i])

        if self.num_prev_columns > 0:
            next_features = [
                torch.cat([pf, f], dim=-1) for pf, f in zip(prev_features, features)
            ]
        else:
            next_features = features

        return x, next_features
        
    def deactivate(self):
        print(f"Deactivating column #{self.num_prev_columns+1}")
        # Freeze column
        for param in self.parameters():
            param.requires_grad = False
        self.activated = False

    def get_param_count(self):
        all_params = sum(p.numel() for p in self.parameters())
        if self.activated or self.num_prev_columns == 0:
            return all_params
        else:
            return (
                all_params
                - sum(p.numel() for p in self.main_stems[-1].parameters())
                - sum(p.numel() for p in self.adapters[-1].parameters())
                - self.scalers[-1].numel()
            )
            

class ProgressiveNet(BaseHead):
    name = 'progressive_net'
    def __init__(
        self,
        max_columns,
        scaler_init_range,
        in_shape,
        output_size,
        norm_type,
        activ_type,
        drop_prob,
        hidden_dims
    ):
        
        super().__init__(in_shape, output_size)

        self.max_columns = max_columns
        self.curr_columns = 1

        c, h, w = self.in_shape
        in_channel = c * h * w
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.columns = nn.ModuleList([
            MLPColumn(
                num_prev_columns=column_idx,
                scaler_init_range=scaler_init_range,
                in_dim=in_channel,
                hidden_dims=hidden_dims,
                out_dim=output_size,
                activ_type=activ_type,
                drop_prob=drop_prob,
            )
            for column_idx in range(max_columns)
        ])
        
    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)

        prev_features = []
        for i in range(self.curr_columns):
            out, prev_features = self.columns[i](x, prev_features)
        
        info = {}
        return out, info

    def intervene(self):
        prev_columns = self.curr_columns
        self.curr_columns = min(self.max_columns, self.curr_columns+1)
        if prev_columns != self.curr_columns:
            self.columns[prev_columns-1].deactivate()
        print(f'{prev_columns} -> {self.curr_columns}')
        
    def count_active_params(self):
        cnt = 0
        for i in range(self.curr_columns):
            cnt += self.columns[i].get_param_count()
        return cnt