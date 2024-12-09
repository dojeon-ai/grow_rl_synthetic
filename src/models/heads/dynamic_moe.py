from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation

NEG_INF = -1e8

class BottleneckExpert(nn.Module):
    def __init__(
        self,
        activ_type: str,
        h_in: int,
        h_bottleneck: int,
        h_out: int,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_in, h_bottleneck),
            init_activation(activ_type=activ_type),
            nn.Linear(h_bottleneck, h_out),
        )
    
    def forward(self, x):
        return self.fc(x)

class MoELayer(nn.Module):
    def __init__(
        self,
        expert_type: str,
        initial_experts: int,
        max_experts: int,
        grow_increment: int,
        output_max_k: int,
        grad_max_k: int,
        bottleneck_match_active_param: bool,
        h_in: int,
        h_out: int,
    ):
        super().__init__()
        self.curr_experts = initial_experts
        self.max_experts = max_experts
        self.grow_increment = grow_increment
        if expert_type == 'linear':
            self.experts = nn.ModuleList([nn.Linear(h_in, h_out) for _ in range(max_experts)])
        elif expert_type == 'bottleneck_relu':
            if bottleneck_match_active_param:
                N = min(output_max_k, max_experts)
            else:
                N = max_experts 
            h_b = int(h_in*(h_out-N)/(h_in+h_out)/N)
            self.experts = nn.ModuleList([BottleneckExpert('relu', h_in, h_b, h_out) for _ in range(max_experts)])
        self.routers = nn.ModuleList([nn.Linear(h_in, 1) for _ in range(max_experts)])
        self.output_max_k = output_max_k
        self.grad_max_k = grad_max_k
        
        self.expert_pcount = sum(p.numel() for p in self.experts[0].parameters())
        self.router_pcount = sum(p.numel() for p in self.routers[0].parameters())

    def forward(self, x):
        logits = torch.stack(
            [self.routers[i](x) for i in range(self.curr_experts)], dim=1
        )
        expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.curr_experts)], dim=1
        )

        # topK
        if self.curr_experts > self.output_max_k:
            values, indices = torch.topk(
                logits, self.output_max_k, dim=1,
            )
            new_logits = torch.ones_like(logits)*NEG_INF
            new_logits = new_logits.scatter_(1, indices, values)
            logits = new_logits

        # gradient topK (on expert outputs)
        if self.curr_experts > self.grad_max_k:
            values, indices = torch.topk(
                logits, self.grad_max_k, dim=1,
            )
            mask = torch.zeros_like(logits).scatter_(1, indices, 1)
            top_outputs = mask * expert_outputs
            bottom_outputs = (1-mask) * expert_outputs
            expert_outputs = top_outputs + bottom_outputs.detach()

        probs = F.softmax(logits, dim=1)
        out = torch.sum(expert_outputs*probs, dim=1)
        
        # computing load balancing loss here for convenience
        values, indices = torch.topk(
            logits, k=1, dim=1,
        )
        frac_selected = torch.zeros_like(logits).scatter_(1, indices, 1).mean(dim=0)
        frac_prob = probs.mean(dim=0)
        load_balance_loss = self.curr_experts*(frac_selected*frac_prob).sum()

        return out, frac_prob, load_balance_loss

    def dynamic_grow(self):
        print(self.curr_experts, 'to', min(self.curr_experts+self.grow_increment, self.max_experts))
        self.curr_experts = min(self.curr_experts+self.grow_increment, self.max_experts)

    def count_active_params(self):
        # Note: this counts parameters that 'contribute to final output'.
        #       this does NOT count the actual activated parameters
        #       NOR the actual computational efficiency of current implementation.
        if self.curr_experts > self.output_max_k:
            return self.expert_pcount*self.output_max_k + self.router_pcount*self.curr_experts
        else:
            return (self.expert_pcount+self.router_pcount)*self.curr_experts
            

class DynamicMoE(BaseHead):
    name = 'dynamic_moe'
    def __init__(
        self,
        expert_type: str,
        initial_experts: int,
        max_experts: int,
        grow_increment: int,
        output_max_k: int,
        grad_max_k: int,
        bottleneck_match_active_param: bool,
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
        in_channel = c * h * w
        
        layers = []
        hidden_dims = [in_channel] + hidden_dims
        for idx in range(len(hidden_dims)-1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx+1]
            layers.append(
                MoELayer(
                    expert_type=expert_type,
                    initial_experts=initial_experts,
                    max_experts=max_experts,
                    grow_increment=grow_increment,
                    output_max_k=output_max_k,
                    grad_max_k=grad_max_k,
                    bottleneck_match_active_param=bottleneck_match_active_param,
                    h_in=h_in,
                    h_out=h_out,
                )
            )
            layers.append(init_activation(activ_type=activ_type))
            layers.append(nn.Dropout(p=drop_prob))
        
        layers.append(
            MoELayer(
                expert_type=expert_type,
                initial_experts=initial_experts,
                max_experts=max_experts,
                grow_increment=grow_increment,
                output_max_k=output_max_k,
                grad_max_k=grad_max_k,
                bottleneck_match_active_param=bottleneck_match_active_param,
                h_in=hidden_dims[-1],
                h_out=output_size,
            )
        )
        self.fc = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        balance_losses = []
        info = {}
        for idx, layer in enumerate(self.fc):
            if isinstance(layer, MoELayer):
                x, avg_prob, balance_loss = layer(x)
                if balance_loss is not None:
                    balance_losses.append(balance_loss)
                for e_i, e_p in enumerate(avg_prob):
                    info[f'router_prob/layer{idx}_expert{e_i:02d}'] = e_p
            else:
                x = layer(x)
        
        if len(balance_losses) > 0:
            info['aux_loss'] = torch.stack(balance_losses).sum()
        return x, info

    def intervene(self):
        for layer in self.fc:
            if isinstance(layer, MoELayer):
                layer.dynamic_grow()

    def count_active_params(self):
        cnt = 0
        for layer in self.fc:
            if isinstance(layer, MoELayer):
                cnt += layer.count_active_params()
        return cnt