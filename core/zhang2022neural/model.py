import math
from einops import rearrange
from torch import nn
import torch

from .modules import *
from ..utils.modules import RegPL


class TCIntensityNet(nn.Module):
    '''
    doi: 10.1016/j.knosys.2022.110005
    '''
    def __init__(
            self, 
            in_channels=2, 
            hidden_channels=512, 
            out_channels=1,
            patch_size=8):
        super(TCIntensityNet, self).__init__()
        self.front = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                hidden_channels, 
                kernel_size=patch_size, 
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(hidden_channels)
        )
        self.mid = DenseConvMixerLayer(4, 4, hidden_channels, 5)
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, t):
        '''
        x: (b, t, c, h, w)
        '''
        t = t.to(x.dtype)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        xse = torch.sum(x, dim=1, keepdim=True)
        xte = torch.sin((math.pi / 2) * (t / 256))
        x = xse * xte[:, None, None, None] + x
        x = self.front(x)
        x = self.mid(x)
        x = self.end(x)
        return x
    
class TCIntensityNetPL(RegPL):
    
    def __init__(self, model_config, optim_config) -> None:
        super(TCIntensityNetPL, self).__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.in_channels,
            model_config.hidden_channels,
            model_config.out_channels,
            model_config.patch_size
        )
        self.save_hyperparameters()

    def forward(self, batch, t):
        return self.model(batch, t)
        
    def training_step(self, batch, batch_idx):
        rad, t, ws = batch
        ws_hat = self(rad, t)
        ws = ws[:, None]
        l = self.loss_fun(ws_hat, ws)
        self.log_dict(
            {
                "train/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        rad, t, ws = batch
        ws_hat = self(rad, t)
        ws = ws[:, None]
        l = self.loss_fun(ws_hat, ws)
        self.log_dict(
            {
                "val/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def test_step(self, batch, batch_idx):
        rad, t, ws = batch
        if not hasattr(self, "truth"):
            self.truth = []
            self.pred = []
        ws_hat = self(rad, t)
        ws = ws[:, None]
        l = self.loss_fun(ws_hat, ws)
        self.truth.append(ws.detach().cpu())
        self.pred.append(ws_hat.detach().cpu())
        self.log_dict(
            {
                "test/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        