from einops import rearrange
from torch import nn
from torch.nn.functional import softmax

from ..ours.modules import *


class ST(nn.Module):

    def __init__(self, input_channels, seq_len):
        super(ST, self).__init__()
        self.conv = nn.Sequential(
            DownsamBlock(input_channels, 512, [2, 1]),
            IdentiBlock(512, 512, 1)
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_k = nn.Conv2d(seq_len, seq_len, 3, padding=1)
        self.conv_v = nn.Conv2d(seq_len, seq_len, 3, padding=1)

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        b = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        x = self.spatial_pool(x)
        x = rearrange(x, "(b t) c h w -> (b c) t h w", b=b)
        k = self.conv_k(x)
        v = self.conv_v(x)
        k = rearrange(k, "(b c) t h w -> b t (c h w)", b=b) # (b, t, 512)
        v = rearrange(v, "(b c) t h w -> b t (c h w)", b=b) # (b, t, 512)
        k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        cos = torch.matmul(k_norm[:, -1][:, None, :], k_norm.permute(0, 2, 1)) # (b, 1, t)
        cos_ = softmax(cos, dim=-1).permute(0, 2, 1) # (b, t, 1)
        v = (cos_ * v).sum(dim=1)
        return v


class TS(nn.Module):

    def __init__(self, input_channels, seq_len):
        super(TS, self).__init__()
        self.conv = nn.Sequential(
            DownsamBlock(input_channels, 512, [2, 1]),
            IdentiBlock(512, 512, 1)
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_k = nn.Conv2d(seq_len, seq_len, 3, padding=1)
        self.conv_v = nn.Conv2d(seq_len, seq_len, 3, padding=1)

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b c) t h w")
        k = self.conv_k(x)
        v = self.conv_v(x)
        k = rearrange(k, "(b c) t h w -> b t (c h w)", b=b)
        v = rearrange(v, "(b c) t h w -> b t (c h w)", b=b)
        k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        cos = torch.matmul(k_norm[:, -1][:, None, :], k_norm.permute(0, 2, 1)) # (b, 1, t)
        cos_ = softmax(cos, dim=-1).permute(0, 2, 1) # (b, t, 1)
        v = (cos_ * v).sum(dim=1)
        v = rearrange(v, "b (c h w) -> b c h w", h=h, w=w)        
        v = self.conv(v)
        v = self.spatial_pool(v)
        v = v.reshape((b, -1))
        return v