import torch
import torch.nn as nn
from models.SimpleMobileNetV2 import InvertedResidual


'''
DCMobileNetV2_DoubleIR is a three layer network with two conv 
layers and a fully connected layer. The convolution layers uses
the Inverted Residual block structure from MobileNetV2
'''


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()


class DCtwoBottlenecks(MobileNetV2):
    def __init__(self, num_classes=10, drop_probs=0.3, out_1=16, er_1=12, out_2=32, er_2=12,
                 use_avgpool=False, in_channels=1):
        super(DCtwoBottlenecks, self).__init__()
        # Input channels for MNIST (grayscale images)
        self.in_channels = in_channels

        # Inverted Residual block layer
        self.features = nn.Sequential(
            InvertedResidual(in_channels=1, out_channels=out_1, stride=1, expand_ratio=er_1, drop_probs=drop_probs),
            InvertedResidual(in_channels=out_1, out_channels=out_2, stride=1, expand_ratio=er_2, drop_probs=drop_probs)
        )

        self.use_avgpool = use_avgpool
        if self.use_avgpool:
            '''
            Global average pooling
            Output dimension should be made adjustable, input size into
            the fully connected layer would need to be adjusted accordingly
            '''
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer for classification
        self.fc = nn.Linear(out_2 * 28 * 28, num_classes)

    def forward(self, x):
        # inverted residual convolution layer
        x = self.features(x)
        # global average pooling
        if self.use_avgpool:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc(x)
        return x
