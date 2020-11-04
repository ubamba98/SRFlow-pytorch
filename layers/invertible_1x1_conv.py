import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class Invertible_1x1_conv2D(nn.Module):
    '''
    Applies invertible 1x1 2D convolution, as described in the paper 
    `Glow: Generative Flow with Invertible 1×1 Convolutions <https://arxiv.org/abs/1807.03039>`
    '''
    def __init__(self, in_channels):
        super(Invertible_1x1_conv, self).__init__()
        self.in_channels = in_channels
        w_init = np.linalg.qr(np.random.randn((self.in_channels, self.in_channels)))[0].astype('float32')
        self.register_parameter("W", nn.Parameter(torch.from_numpy(w_init)))

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
    
    def forward(self, x, logdet=None, reverse=False):
        _check_input_dim(x)

        _, C, H, W = x.shape
        dlogdet = torch.log(torch.abs(torch.det(self.W)))*H*W
        
        if not reverse:
            z = F.conv2d(x, self.W.view(C, C, 1, 1))
            logdet = logdet + dlogdet

        else:
            _W = torch.inverse(self.W)
            z = F.conv2d(x, _W.view(C, C, 1, 1))
            logdet = logdet - dlogdet

        return z, logdet