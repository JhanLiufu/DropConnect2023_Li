# reference: https://github.com/akamaster/pytorch_resnet_cifar10/tree/master

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.SimpleMobileNetV2 import InvertedResidual
from models.dcconv2d import DCConv2d


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',
                 replace_conv=[False, False], drop_probs=[0,0], expand_ratios=[6,6]):
        '''
        in_planes: int, input dimension
        out_planes: int, output dimensions
        replace_conv: array-like, [boolean, boolean]. whether to drop the first or second convolution
                      with bottleneck
        drop_probs: array-like, [x_1, x_2], drop rate for each bottleneck. Not used if a normal 
                    convolution is used
        expand_ratios: expansion ratio for bottleneck. Not used if a normal 
                    convolution is used
        '''
        super(BasicBlock, self).__init__()
        if replace_conv[0]:
            self.conv1 = InvertedResidual(in_planes, planes, stride=stride, option=option, 
                                          drop_probs=drop_probs[0], expand_ratio=expand_ratios[0])
        else:
            # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv1 = DCConv2d(drop_probs[0], in_planes, planes, 3, stride, 1, False)
        self.bn1 = nn.BatchNorm2d(planes)

        if replace_conv[1]:
            self.conv2 = InvertedResidual(planes, planes, stride=1, option=option, 
                                          drop_probs=drop_probs[1], expand_ratio=expand_ratios[1])
        else:
            # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = DCConv2d(drop_probs[1], planes, planes, 3, 1, 1, False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, block_setting=None, num_classes=10):
        # block_setting=[3,3,3] is default for ResNet20
        super(ResNet, self).__init__()
        if block_setting is None:
            # first value is stride, second value is the BasicBlock to modify
            block_setting = [
                             [3, [{'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                  {'replace_conv': [1, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [6, 0]},
                                  {'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                 ]
                              ], 
                              [3, [{'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                  {'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                  {'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                 ]
                              ],
                              [3, [{'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                  {'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                  {'replace_conv': [0, 0],
                                   'drop_probs': [0, 0],
                                   'expand_ratios': [0, 0]},
                                 ]
                              ],
                            ]
        
        self.name = self._make_name_from_block_setting(block_setting)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, block_setting[0], stride=1)
        self.layer2 = self._make_layer(block, 32, block_setting[1], stride=2)
        self.layer3 = self._make_layer(block, 64, block_setting[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)


    def _make_name_from_block_setting(self, block_setting):
        model_name = 'ResNet'
        for layer_setting in block_setting:
            layer_name = ''
            for b in layer_setting[1]:
                for r, d, e in zip(b.get('replace_conv'), b.get('drop_probs'), b.get('expand_ratios')):
                    # if replace_conv is 0, the other two should be 0 as well.
                    layer_name += f'|{r}{d}{e}'
            model_name += f'_&{layer_name}'
        return model_name


    def _make_layer(self, block, planes, layer_block_setting, stride):
        strides = [stride] + [1]*(layer_block_setting[0]-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, 
                                replace_conv = layer_block_setting[1][i].get('replace_conv'),
                                drop_probs = layer_block_setting[1][i].get('drop_probs'),
                                expand_ratios = layer_block_setting[1][i].get('expand_ratios')))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])


# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])
