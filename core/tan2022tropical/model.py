from einops import rearrange
from torch import nn

from .modules import BasicConv
from ..utils.modules import RegPL
from ..ours.modules import CBAM


class TCIntensityNet(nn.Module):
    '''
    doi: 10.3390/rs14040812
    '''
    def __init__(self, in_channels, out_channels):
        super(TCIntensityNet, self).__init__()
        self.atten = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # loss too much info
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=2),
            CBAM(in_channels*2, reduction_ratio=1),
            nn.BatchNorm2d(in_channels*2)
        )

        self.block1 = BasicConv(in_channels*2, 32)
        self.adapt1 = nn.Conv2d(in_channels*2, 32, 1)
        self.drop = nn.Dropout2d()
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.block2 = BasicConv(32, 64)
        self.adapt2_1 = nn.Conv2d(32, 64, 1)
        self.adapt2_2 = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.block3 = BasicConv(64, 128)
        self.adapt3 = nn.Conv2d(64, 128, 1)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.block4 = BasicConv(128, 256)
        self.adapt4_1 = nn.Conv2d(128, 256, 1)
        self.adapt4_2 = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1d = nn.Sequential(
            nn.Flatten(2),
            nn.Conv1d(256, 128, 1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, 1),
            nn.LeakyReLU(),
        )

        self.reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 256, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_channels),
        )

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.atten(x)

        x1 = self.block1(x)
        x1 = self.adapt1(x) + x1

        x2 = self.drop(x1)
        x2 = self.pool1(x2)

        x3 = self.block2(x2)
        x3 += self.adapt2_1(x2) + self.adapt2_2(x)

        x4 = self.pool2(x3)

        x5 = self.block3(x4)
        x5 += self.adapt3(x4)

        x6 = self.pool3(x5)

        x7 = self.block4(x6)
        x7 += self.adapt4_1(x6) + self.adapt4_2(x4)

        x7 = self.conv1d(x7)
        x7 = self.reg(x7)
        return x7
    
class TCIntensityNetPL(RegPL):
    
    def __init__(self, model_config, optim_config) -> None:
        super(TCIntensityNetPL, self).__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.in_channels,
            model_config.out_channels,
        )
        self.save_hyperparameters()
        