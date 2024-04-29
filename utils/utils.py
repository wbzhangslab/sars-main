import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.optim import Optimizer

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.cuda()
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class Attention(nn.Module):
	
	def __init__(self, p):
		super(Attention, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
     
class Attention_kl(nn.Module):
	
	def __init__(self, p):
		super(Attention_kl, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.kl_div(self.attention_map(fm_s), self.attention_map(fm_t), reduction='mean')

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
