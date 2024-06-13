from einops import rearrange
from torch import nn

from .modules import *
from ..utils.modules import RegPL


class TCIntensityNet(nn.Module):
    '''
    doi: 10.1109/TGRS.2024.3352704
    '''
    def __init__(self, input_channels, output_channels, seq_len):
        super(TCIntensityNet, self).__init__()
        # extract spatial feature
        self.prepare = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            IdentiBlock(64, 64, 1),
            IdentiBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            DownsamBlock(64, 128, [2, 1]),
            IdentiBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            DownsamBlock(128, 256, [2, 1]),
            IdentiBlock(256, 256, 1)
        )
        self.st = ST(256, seq_len)
        self.ts = TS(256, seq_len)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        b = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b)
        f1 = self.st(x)
        f2 = self.ts(x)
        f = torch.cat((f1, f2), dim=1)
        return self.fc(f)


class TCIntensityNetPL(RegPL):
    
    def __init__(self, model_config, optim_config) -> None:
        super(TCIntensityNetPL, self).__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.in_channels,
            model_config.out_channels,
            model_config.seq_len
        )
        self.save_hyperparameters()