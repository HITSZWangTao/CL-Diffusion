import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class GaussianDiffusion(nn.Module):
    '''
    Gaussion diffusion
    '''
    def __init__(self,max_step,noise_schedule,device):
        super().__init__()
        self.max_step = max_step # maximum diffusion steps
        beta = np.array(noise_schedule) # \beta, [T]
        alpha = torch.tensor((1-beta).astype(np.float32)).to(device) # \alpha_t [T]
        self.alpha_bar = torch.cumprod(alpha, dim=0) # \bar{\alpha_t}, [T]
        self.noise_weights = torch.sqrt(1 - self.alpha_bar) # \sqrt{1 - \bar{\alpha_t}}, [T]
        self.info_weights = torch.sqrt(self.alpha_bar) # \sqrt{\bar{\alpha_t}}, [T]

    def degrade_fn(self, x_0, t):
        device = x_0.device
        noise_weight = self.noise_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, 1, 1, 1]
        info_weight = self.info_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, 1, 1, 1] 
        noise = noise_weight * torch.randn_like(x_0, dtype=torch.float32, device=device) 
        x_t = info_weight * x_0 + noise 
        return x_t

    def native_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max).to(device) 
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat