import torch
import torch.nn as nn
from models.SimpleMobileNetV2 import InvertedResidual


'''
DCnBottlenecks is an (n+1) layer network with n conv 
layers and a fully connected layer. The convolution layers uses
the Inverted Residual block architecture from MobileNetV2
'''


class DCnBottlenecks(nn.Module):
    def __init__(self, bottleneck_cfg, num_classes=10, drop_probs=0.3, in_channels=1, use_avgpool=False):
        """
        bottleneck_cfg: array-like, shape [[output_dimension_1, stride_1, expand_ratio_1], ....]
        num_classes: int, number of classes for classification
        drop_probs: drop connect drop rate
        in_channels: int, input dimension
        use_avgpool: bool, whether to do average pooling after convolution / before fully connected layer
        """
        super(DCnBottlenecks, self).__init__()

        self.bottleneck_cfg = bottleneck_cfg
        self.num_classes = num_classes
        self.drop_probs = drop_probs
        self.in_channels = in_channels
        # output dimension of the last bottleneck
        self.out_channels = bottleneck_cfg[-1][0]
        # self.out_channels = 8
        self.use_avgpool = use_avgpool
        self.name = f'DC{len(self.bottleneck_cfg)}Bottlenecks'

        # Linear bottlenecks
        self.bottlenecks = self._make_bottlenecks()

        if self.use_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer for classification
        self.fc = nn.Linear(self.out_channels * 28 * 28, num_classes)

    def _make_bottlenecks(self):
        layers = []
        in_planes = self.in_channels
        # for x in [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],]:
        for x in self.bottleneck_cfg:
            # x = [output_dimension, stride, expand_ratio]
            layers.append(InvertedResidual(in_channels=in_planes, out_channels=x[0],
                                           stride=x[1], expand_ratio=x[2],
                                           drop_probs=self.drop_probs))
            '''
            Input dimension of the next bottleneck  = output dimension of current bottleneck
            '''
            in_planes = x[0]

        return nn.Sequential(*layers)

    def forward(self, x):
        # inverted residual convolution layers
        x = self.bottlenecks(x)

        if self.use_avgpool:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc(x)
        return x