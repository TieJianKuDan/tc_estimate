from torch import nn
import torch


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):

    def __init__(self, dim, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            Residual(
                nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.block(x)


class DenseConvMixerBlock(nn.Module):

    def __init__(self, depth, dim, kernel_size):
        super(DenseConvMixerBlock, self).__init__()
        self.mixer_blocks = [None] * depth
        self.bottlenecks = [None] * depth
        for i in range(depth):
            self.mixer_blocks[i] = ConvMixerBlock(dim, kernel_size)
            self.bottlenecks[i] = nn.Sequential(
                nn.Conv2d(dim*(i + 2), dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )
        self.mixer_blocks = nn.Sequential(*self.mixer_blocks)
        self.bottlenecks = nn.Sequential(*self.bottlenecks)
        self.depth = depth

    def forward(self, x):
        mem = []
        last = x
        mem.append(x)
        for i in range(self.depth):
            last = self.mixer_blocks[i](last)
            mem.append(last)
            last = self.bottlenecks[i](torch.cat(mem, dim=1))
        return last


class DenseConvMixerLayer(nn.Module):

    def __init__(self, layer_depth, block_depth, dim, kernel_size):
        super(DenseConvMixerLayer, self).__init__()
        self.mixer_blocks = [None] * layer_depth
        self.bottlenecks = [None] * layer_depth
        for i in range(layer_depth):
            self.mixer_blocks[i] = DenseConvMixerBlock(block_depth, dim, kernel_size)
            self.bottlenecks[i] = nn.Sequential(
                nn.Conv2d(dim*(i + 2), dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )
        self.mixer_blocks = nn.Sequential(*self.mixer_blocks)
        self.bottlenecks = nn.Sequential(*self.bottlenecks)
        self.depth = layer_depth

    def forward(self, x):
        mem = []
        last = x
        mem.append(x)
        for i in range(self.depth):
            last = self.mixer_blocks[i](last)
            mem.append(last)
            last = self.bottlenecks[i](torch.cat(mem, dim=1))
        return last