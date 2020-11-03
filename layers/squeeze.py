import torch
import torch.nn as nn
from torch.nn import functional as F

class Squeeze2D(nn.Module):
    '''
    Reshapes the activation map to half the width and height. In order to preserve the locality, neighboring pixels are stacked.
    This operation is done over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    '''
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
    
    def forward(self, x, reverse=False):
        _check_input_dim(x)

        N, C, H, W = x.shape
        if not reverse:
            assert H%2 == 0 and W%2 == 0
            x = F.unfold(x, kernel_size=2, stride=2).view(N, -1, H//2, W//2)
        else:
            x = x.view(N, C, -1)
            x = F.fold(x, output_size=(H*2, W*2), kernel_size=2, stride=2)
        return x