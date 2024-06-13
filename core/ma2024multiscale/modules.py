import torch
from torch import nn


class SCB(nn.Module):
    '''
    Separable Convolutional Block
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernels_per_layer=1,
        stride=1
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer,
            out_channels,
            kernel_size=1
        )
        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.out(x)
        return x


class MSCB(nn.Module):
    '''
    Multiscale Separable Convolutional Block
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride=1
    ) -> None:
        super().__init__()
        self.depthwise1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            groups=in_channels,
        )
        self.depthwise3 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
        )
        self.depthwise5 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            stride=stride,
            groups=in_channels,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            padding=1,
            stride=stride
        )
        self.pointwise = nn.Conv2d(
            in_channels * 4,
            out_channels,
            kernel_size=1
        )
        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        x1 = self.depthwise1(x)
        x2 = self.depthwise3(x)
        x3 = self.depthwise5(x)
        x4 = self.max_pool(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.pointwise(x)
        x = self.out(x)
        return x


class SCPB(nn.Module):
    '''
    SeparableConv Pool Block
    '''
    def __init__(
            self, 
            in_channels, 
            out_channels
    ) -> None:
        super().__init__()
        self.sc1 = SCB(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.drop1 = nn.Dropout2d(p=0.2)
        self.sc2 = SCB(
            in_channels=out_channels,
            out_channels=out_channels
        )
        self.drop2 = nn.Dropout2d(p=0.2)
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            padding=1,
            stride=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        y = self.sc1(x)
        y = self.drop1(y)
        y = self.sc2(y)
        y = self.drop2(y)
        y = self.max_pool(y)
        x = self.conv1(x)
        y = x + y
        return y


class CAB(nn.Module):
    '''
    Channel Attention Block
    '''
    def __init__(
        self, 
        in_channels
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_a1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels//4,
            kernel_size=1
        )
        self.relu_a = nn.ReLU()
        self.conv_a2 = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels,
            kernel_size=1
        )
        self.conv_m1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels//4,
            kernel_size=1
        )
        self.relu_m = nn.ReLU()
        self.conv_m2 = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels,
            kernel_size=1
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        a = self.avg_pool(x)
        a = self.conv_a1(a)
        a = self.relu_a(a)
        a = self.conv_a2(a)
        m = self.max_pool(x)
        m = self.conv_m1(m)
        m = self.relu_m(m)
        m = self.conv_m2(m)
        g = self.sig(a + m)
        x = g * x
        return x


class SAB(nn.Module):
    '''
    Spatial Attention Block
    '''
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )

    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        g = torch.cat([a, m], dim=1)
        g = self.conv(g)
        x = x * torch.sigmoid(g)
        return x


class ASB(nn.Module):
    '''
    Att-separableconv Block 
    '''
    def __init__(
        self, 
        in_channels
    ) -> None:
        super().__init__()
        self.sc1 = SCB(
            in_channels=in_channels,
            out_channels=in_channels
        )
        self.drop1 = nn.Dropout2d(p=0.2)
        self.sc2 = SCB(
            in_channels=in_channels,
            out_channels=in_channels
        )
        self.drop2 = nn.Dropout2d(p=0.2)
        self.att = nn.Sequential(
            CAB(in_channels),
            SAB()
        )
    
    def forward(self, x):
        '''
        shape: (b, c, h, w)
        '''
        y = self.sc1(x)
        y = self.drop1(y)
        y = self.sc2(y)
        y = self.drop2(y)
        y = self.att(y)
        y = x + y
        return y