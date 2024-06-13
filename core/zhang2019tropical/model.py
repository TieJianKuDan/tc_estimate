import torch
from torch import nn
from torch.nn.functional import l1_loss, mse_loss

from ..utils.modules import RegPL


def tcie_loss(y_hat, y):
    '''
    y_t1: time t
    y_t2: time t-1
    y: actual value
    '''
    y_t1, y_t2 = y_hat
    lc = torch.sqrt(mse_loss(y_t1, y)) \
        + l1_loss(torch.ones_like(y), y_t1 / y)
    lcp = torch.abs(l1_loss(y_t1, y_t2) - 0.5 / 0.5144)
    return lc + 0.5 * lcp

class TCIntensityNet(nn.Module):
    '''
    doi: 10.1109/TGRS.2019.2938204
    '''
    def __init__(self, h, w) -> None:
        super().__init__()
        # Features Extraction
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        # WV Attention
        self.att1 = nn.Sequential(
            nn.Conv2d(1, 32, 10, 2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.att2 = nn.Sequential(
            nn.Conv2d(1, 32, 10, 2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Intensity Regression
        self.reg1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (h // 2**3) * (w // 2**3), 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 166),
            # nn.Softmax(dim=1)
            nn.Linear(166, 1),
        )
        self.reg2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (h // 2**3) * (w // 2**3), 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 166),
            # nn.Softmax(dim=1)
            nn.Linear(166, 1),
        )
        # I don't know why ClsR + l1-loss cannot converge
        # vw1 = torch.arange(13, 17.3, 0.3) / 0.5144
        # vw2 = torch.arange(17.3, 24.6, 0.6) / 0.5144
        # vw3 = torch.arange(24.6, 51.1, 0.3) / 0.5144
        # vw4 = torch.arange(51.1, 79.9, 0.6) / 0.5144
        # self.vw = torch.cat((vw1, vw2, vw3, vw4), dim=0)[:, None].cuda()
    
    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        ir1 = x[:, 1, 1][:, None, :, :]
        ir2 = x[:, 0, 1][:, None, :, :]
        wv1 = x[:, 1, 0][:, None, :, :]
        wv2 = x[:, 0, 0][:, None, :, :]
        ir1 = self.layer1_1(ir1)
        ir2 = self.layer2_1(ir2)
        wv1 = self.att1(wv1)
        wv2 = self.att2(wv2)
        x1 = ir1 * wv1
        x2 = ir2 * wv2
        x1 = self.layer1_2(x1)
        x2 = self.layer2_2(x2)
        x1 = self.reg1(x1)
        x2 = self.reg2(x2)
        # x1 = torch.matmul(x1, self.vw)
        # x2 = torch.matmul(x2, self.vw)
        return x1, x2
    

class TCIntensityNetPL(RegPL):

    def __init__(self, model_config, optim_config) -> None:
        super().__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.h,
            model_config.w,
        )
        self.loss_fun = tcie_loss
        self.save_hyperparameters()
        