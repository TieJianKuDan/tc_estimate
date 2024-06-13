import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torch.nn import functional as F

from .functions import *


class RIConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, batch_size, h, w, stride, padding):
        super(RIConv2d, self).__init__()
        self.coordi = gen_coord_3(batch_size, h, w).cuda()
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=padding)

    def forward(self, x):
        return deform_conv2d(
            input=x, 
            offset=self.coordi, 
            weight=self.conv.weight,
            padding=(self.padding, self.padding),
            stride=(self.stride, self.stride)
        )


class IdentiBlock(nn.Module):
    def __init__(self, in_channel, out_channel, batch_size, h, w):
        super(IdentiBlock, self).__init__()
        self.conv1 = RIConv2d(
            in_channel,
            out_channel, 
            batch_size,
            h, w,
            stride=1, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = RIConv2d(
            out_channel,
            out_channel, 
            batch_size,
            h, w,
            stride=1, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class DownsamBlock(nn.Module):
    def __init__(self, in_channel, out_channel, batch_size, h, w):
        super(DownsamBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                out_channel, 
                kernel_size=1, 
                stride=2, 
                padding=0, 
                bias=False
            ),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = RIConv2d(
            in_channel, 
            out_channel, 
            batch_size,
            h, w,
            stride=2, 
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = RIConv2d(
            out_channel, 
            out_channel, 
            batch_size,
            h, w,
            stride=1, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, batch_size, h, w, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = RIConv2d(
            2, 
            1, 
            batch_size,
            h, w,
            padding=padding, 
            stride=1
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale

class CBAM(nn.Module):
    def __init__(
            self, 
            input_channels, 
            batch_size, 
            h, w, 
            reduction_ratio=16, 
            kernel_size=3
    ):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(
            input_channels, reduction_ratio=reduction_ratio
        )
        self.spatial_att = SpatialAttention(
            batch_size=batch_size, h=h, w=w, kernel_size=kernel_size
        )

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class DSDownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)