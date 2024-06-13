from einops import rearrange
import torch
from torch import nn

from ..utils.modules import RegPL
from .modules import *


class TCIntensityNet(nn.Module):
    '''
    doi: 10.1109/TGRS.2024.3349416.
    '''
    def __init__(
        self, 
        in_channels
    ) -> None:
        super().__init__()
        # scale feature
        self.mscb1 = MSCB(in_channels, 32)
        self.mscb2 = MSCB(32, 64, stride=2)
        # feature condense
        self.scpb1 = SCPB(64, 128)
        self.scpb2 = SCPB(128, 256)
        self.scpb3 = SCPB(256, 1024)
        # deep semantic
        self.asb = nn.Sequential(
            *[ASB(1024) for _ in range(6)]
        )
        self.scpb4 = SCPB(1024, 4096)
        # layer feature
        self.conv1 = nn.Conv2d(64, 64, 1, 2)
        self.conv2 = nn.Conv2d(192, 192, 1, 2)
        self.conv3 = nn.Conv2d(448, 448, 1, 2)
        self.conv4 = nn.Conv2d(1472, 1472, 1, 1)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # decision
        self.fc = nn.Sequential(
            nn.Linear(5568, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        '''
        shape: (b, t, c, h, w)
        '''
        x = rearrange(x, "b t c h w -> b (t c) h w")
        # scale feature
        f1 = self.mscb1(x)
        f1 = self.mscb2(f1)
        # feature condense
        f2 = self.scpb1(f1)
        f3 = self.scpb2(f2)
        f4 = self.scpb3(f3)
        # deep semantic
        f5 = self.asb(f4)
        f5 = self.scpb4(f5)
        f5 = self.gap(f5)
        # layer feature
        f6 = self.conv1(f1)
        f6 = torch.cat((f2, f6), dim=1)
        f6 = self.conv2(f6)
        f6 = torch.cat((f3, f6), dim=1)
        f6 = self.conv3(f6)
        f6 = torch.cat((f4, f6), dim=1)
        f6 = self.conv4(f6)
        f6 = self.gap(f6)
        # decision
        f7 = torch.cat((f5, f6), dim=1)
        f7 = self.fc(f7)
        return f7


class TCIntensityNetPL(RegPL):

    def __init__(self, model_config, optim_config) -> None:
        super().__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.in_channels,
        )
        self.save_hyperparameters()