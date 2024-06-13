from einops import rearrange
from torch import nn

from ..utils.modules import RegPL


class TCIE(nn.Module):
    '''
    doi: 10.1109/JSTARS.2021.3050767
    '''
    def __init__(self, in_channels, out_channels):
        super(TCIE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        )
        self.reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10816, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, out_channels)
        )

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.features(x)
        x = self.reg(x)
        return x
    
class TCIEPL(RegPL):
    
    def __init__(self, model_config, optim_config) -> None:
        super(TCIEPL, self).__init__(model_config, optim_config)
        self.model = TCIE(
            model_config.in_channels,
            model_config.out_channels,
        )
        self.save_hyperparameters()
        