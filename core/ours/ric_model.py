from einops import rearrange

from ..utils.functions import build_cosin_similarity
from ..utils.modules import RegPL
from .ric_modules import *


class TCIntensityNet(nn.Module):
    def __init__(self, input_channels, output_channels, batch_size, h, w, seq_len):
        super(TCIntensityNet, self).__init__()
        # extract spatial feature
        self.prepare = nn.Sequential(
            RIConv2d(input_channels, 64, batch_size*seq_len, h//2**1, w//2**1, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            IdentiBlock(64, 64, batch_size*seq_len, h//2**2, w//2**2),
            IdentiBlock(64, 64, batch_size*seq_len, h//2**2, w//2**2)
        )
        self.layer2 = nn.Sequential(
            DownsamBlock(64, 128, batch_size*seq_len, h//2**3, w//2**3),
            IdentiBlock(128, 128, batch_size*seq_len, h//2**3, w//2**3)
        )
        self.layer3 = nn.Sequential(
            DownsamBlock(128, 256, batch_size*seq_len, h//2**4, w//2**4),
            IdentiBlock(256, 256, batch_size*seq_len, h//2**4, w//2**4)
        )
        self.layer4 = nn.Sequential(
            DownsamBlock(256, 512, batch_size*seq_len, h//2**5, w//2**5),
            IdentiBlock(512, 512, batch_size*seq_len, h//2**5, w//2**5)
        )
        # extract spatial-temporal feature
        self.branch1 = nn.Sequential(
            CBAM(512 * seq_len, batch_size, h//2**5, w//2**5),
            DSDownBlock(512 * seq_len, 256 * seq_len),
            CBAM(256 * seq_len, batch_size, h//2**6, w//2**6),
            DSDownBlock(256 * seq_len, 128 * seq_len),
            CBAM(128 * seq_len, batch_size, h//2**7, w//2**7),
            nn.Conv2d(128 * seq_len, 512, 1)
        )
        # extract motion feature
        self.branch2 = nn.Sequential(
            DownsamBlock(
                (seq_len-1)*1, (seq_len-1)*2,
                batch_size, h*w//2**11, h*w//2**11
            ),
            DownsamBlock(
                (seq_len-1)*2, (seq_len-1)*4, 
                batch_size, h*w//2**12, h*w//2**12
            ),
            DownsamBlock(
                (seq_len-1)*4, (seq_len-1)*8, 
                batch_size, h*w//2**13, h*w//2**13
            ),
            DownsamBlock(
                (seq_len-1)*8, (seq_len-1)*16, 
                batch_size, h*w//2**14, h*w//2**14
            ),
            DownsamBlock(
                (seq_len-1)*16, (seq_len-1)*32, 
                batch_size, h*w//2**15, h*w//2**15
            )
        )
        # regress
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear((seq_len-1)*32 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        m = build_cosin_similarity(
            rearrange(x, "(b t) c h w -> b t c h w", b=b)
        ).to(x.device)
        m = self.branch2(m)

        x = rearrange(x, "(b t) c h w -> b (t c) h w", b=b)
        x = self.branch1(x)
        x = torch.cat((x, m), dim=1)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    

class TCIntensityNetPL(RegPL):
    
    def __init__(self, model_config, optim_config) -> None:
        super(TCIntensityNetPL, self).__init__(model_config, optim_config)
        self.model = TCIntensityNet(
            model_config.in_channels,
            model_config.out_channels,
            optim_config.batch_size,
            model_config.h,
            model_config.w,
            model_config.seq_len,
        )
        self.save_hyperparameters()