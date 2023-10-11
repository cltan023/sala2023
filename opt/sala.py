""" 
Our implementation is adapted from LookAhead optimizer with minimal modification.
https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lookahead.py
"""

import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from random import random

class SALA(Optimizer):
    def __init__(self, base_optimizer, sala_alpha=0.8, sala_k=5, sala_rho=0.05):
        # NOTE super().__init__() not called on purpose
        if not 0.0 <= sala_alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {sala_alpha}')
        if not 1 <= sala_k:
            raise ValueError(f'Invalid lookahead steps: {sala_k}')
        defaults = dict(sala_alpha=sala_alpha, sala_k=sala_k, sala_rho=sala_rho)
        self._base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self._base_optimizer.param_groups:
                group.setdefault(name, default)
        
        self.sala_step = 0
        self.sala_k = sala_k
        self.sala_rho = sala_rho
        self.param_dist = 0.0
        
    @torch.no_grad()
    def update_slow(self, group):
        group['dist'] = 0.0
        for fast_p in group["params"]:
            param_state = self._base_optimizer.state[fast_p]
            if 'sala_slow_buff' not in param_state:
                param_state['sala_slow_buff'] = torch.empty_like(fast_p)
                param_state['sala_slow_buff'].copy_(fast_p)
            slow = param_state['sala_slow_buff']
            diff = fast_p - slow
            group['dist'] += diff.pow(2.0).sum().item()
            slow.add_(diff, alpha=group['sala_alpha'])
            # fast_p.copy_(slow)
        group['dist'] = group['dist'] ** 0.5 * (1.0 - group['sala_alpha'])
                    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SALA requires closure, but it was not provided"
        
        self.sala_step += 1
        if self.sala_step % self.sala_k == 0:
            if self.param_dist < self.sala_rho:
                for group in self._base_optimizer.param_groups:
                    group['grad_norm'] = self.grad_norm(group)
                    scale = group["sala_rho"] / (group['grad_norm'] + 1.0e-12)
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.data.add_(p.grad, alpha=scale)
                        
                closure = torch.enable_grad()(closure)
                self.zero_grad()
                closure()
        
            for group in self._base_optimizer.param_groups:  
                for p in group["params"]:
                    if p.grad is None: continue
                    state = self._base_optimizer.state[p]
                    p.data.copy_(state['sala_slow_buff'])
                
        self._base_optimizer.step()
        
        if self.sala_step % self.sala_k == self.sala_k - 1:
            self.param_dist = 0.0
            for group in self._base_optimizer.param_groups:
                self.update_slow(group)
                self.param_dist += group['dist'] ** 2
            self.param_dist = self.param_dist ** 0.5
            if self.param_dist < self.sala_rho:
                for group in self._base_optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        state = self._base_optimizer.state[p]
                        p.data.copy_(state['sala_slow_buff'])
                  
    def grad_norm(self, group):
        shared_device = self._base_optimizer.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm.item()

    def state_dict(self):
        return self._base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._base_optimizer.load_state_dict(state_dict)
        self.param_groups = self._base_optimizer.param_groups
        
    def backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self._base_optimizer.param_groups:
            for p in group['params']:
                param_state = self._base_optimizer.state[p]
                tmp = torch.zeros_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(param_state['sala_slow_buff'])
                param_state['sala_slow_buff'].copy_(tmp.data)

    def clear_and_load_backup(self):
        for group in self._base_optimizer.param_groups:
            for p in group['params']:
                param_state = self._base_optimizer.state[p]
                tmp = torch.zeros_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(param_state['sala_slow_buff'])
                param_state['sala_slow_buff'].copy_(tmp.data)