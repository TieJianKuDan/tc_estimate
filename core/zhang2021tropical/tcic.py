from einops import rearrange
from .modules import *
from ..utils.modules import ClassiPL


class TCIC(nn.Module):
    '''
    doi: 10.1109/JSTARS.2021.3050767
    '''
    def __init__(
            self, 
            input_channels,
            A=5, B=10, C=5, 
            k=256, l=256, m=384, n=384, 
            class_nums=3):
        super(TCIC, self).__init__()
        self.stem = Inception_Stem(input_channels)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(output_channels, 1152, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1152)
        self.inception_resnet_c = self._generate_inception_module(2144, 2144, C, InceptionResNetC)

        #6x6 featuresize
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(2144, class_nums)

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        '''
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2144)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):

        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers
    

class TCICPL(ClassiPL):

    def __init__(self, model_config, optim_config) -> None:
        super().__init__(model_config, optim_config)
        self.model = TCIC(model_config.input_channels)
        self.save_hyperparameters()
