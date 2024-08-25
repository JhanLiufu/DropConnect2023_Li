import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.ResNet import LambdaLayer
from models.dcconv2d import DCConv2d


'''
This is an implementation of the Inverted Residual block
used in MobileNetv2 architecture plus an option to drop
weights in the depth-wise convolution
'''


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, option='B', drop_probs=0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.drop_probs = drop_probs
        # make the input 'thicker' i.e. more input channels
        hidden_dim = in_channels * expand_ratio

        layers = []
        # expand the input image
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=stride, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depth-wise conv
        if drop_probs == 0:
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                          stride=stride,padding=1, groups=hidden_dim, bias=False))
        else:
            layers.append(DCConv2d(self.drop_probs, hidden_dim, hidden_dim, 3,
                          stride, 1, False, groups=hidden_dim))

        # Point-wise conv
        layers.extend([
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

        # Shortcut connection (inverted residual)
        self.shortcut = nn.Sequential()
        # print(in_channels)
        # print(out_channels)
        # if stride != 1 or in_channels != out_channels:
        #     if option == 'A':
        #         self.shortcut = LambdaLayer(lambda x:
        #                                     F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4),
        #                                           "constant", 0))
        #     elif option == 'B':
        #         self.shortcut = nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(out_channels)
        #         )

    def forward(self, x):
        # print(x.shape)
        out = self.conv(x)
        # print(out.shape)
        shortcut_out = self.shortcut(x)
        # print(shortcut_out.shape)
        out += shortcut_out

        return out


'''
SimpleMobileNet is a two layer network with one conv 
layer and a fully connected layer. The convolution layer uses
the Inverted Residual block structure from MobileNetV2
'''


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2,self).__init__()


class SimpleMobileNetV2(MobileNetV2):
    def __init__(self, num_classes=10, use_avgpool=False):
        super(MobileNetV2, self).__init__()
        # Input channels for MNIST (grayscale images)
        self.in_channels = 1

        # Inverted Residual block layer
        self.features = nn.Sequential(
            InvertedResidual(in_channels=1, out_channels=32, stride=1, expand_ratio=6),
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
        self.fc = nn.Linear(32 * 28 * 28, num_classes)

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
