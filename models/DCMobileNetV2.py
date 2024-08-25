import torch
import torch.nn as nn
from models.SimpleMobileNetV2 import InvertedResidual


'''
SimpleMobileNet is a two layer network with one conv 
layer and a fully connected layer. The convolution layer uses
the Inverted Residual block structure from MobileNetV2
'''


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2,self).__init__()


class DCMobileNetV2(MobileNetV2):
    def __init__(self, num_classes=10, use_avgpool=False):
        super(DCMobileNetV2, self).__init__()
        # Input channels for MNIST (grayscale images)
        self.in_channels = 1

        # Inverted Residual block layer
        self.features = nn.Sequential(
            InvertedResidual(in_channels=1, out_channels=8, stride=1, expand_ratio=18),
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
        self.fc = nn.Linear(8 * 28 * 28, num_classes)

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
