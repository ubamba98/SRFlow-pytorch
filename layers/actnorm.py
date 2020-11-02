import torch
import torch.nn as nn

class ActNorm2D(nn.Module):
    '''
    Applies Activation Normalizaton over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper `Glow: Generative Flow with Invertible 1×1 Convolutions <https://arxiv.org/abs/1807.03039>`.
    '''
    def __init__(self, num_features, scale=1.):
        size = (1, self.num_features, 1, 1)
        self.initialized = False
        self.num_features = num_features
        self.scale = scale
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
    
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
    
    def init_params(self, x):
        x_mean = torch.mean(x, (0, 2, 3), keepdim=True)
        x_std = torch.std(x, (0, 2, 3), keepdim=True)
        logs = self.scale/(x_std+1e-6)
        self.bias.data.copy_(x_mean)
        self.logs.data.copy_(logs)
        self.initialized = True
    
    def forward(self, x, logdet=None, reverse=False):
        _check_input_dim(x)

        if not self.initialized:
            init_params(x)
        
        _, C, H, W = x.shape
        logdet_factor = H * W

        assert C == self.num_features

        if not reverse:
            x = x + self.bias
            x = x * torch.exp(self.logs)
        else:
            x = x * torch.exp(-self.logs)
            x = x - self.bias
            logdet_factor *= -1
        
        if logdet!=None:
            dlogdet = torch.sum(self.logs)*logdet_factor
            return x, logdet + dlogdet

        return x

        
    
