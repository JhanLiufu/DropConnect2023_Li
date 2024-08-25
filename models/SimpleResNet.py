from models.ResNet import _weights_init, BasicBlock
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()


'''
SimpleResNet model is a two layer network with one conv 
layer and a fully connected layer. The convolution layer uses
the BasicBlock structure from ResNet
'''


class SimpleResNet(ResNet):
    def __init__(self):
        super(ResNet, self).__init__()
        # Input channels for MNIST (grayscale images)
        self.in_planes = 1

        # Define the BasicBlock layer
        self.block = BasicBlock(self.in_planes, 16)

        # Define the fully connected layer for classification (10 output classes)
        # Adjust input size based on output size of the block
        self.linear = nn.Linear(16 * 28 * 28, 10)

        self.apply(_weights_init)

    def forward(self, x):
        # Forward pass through the BasicBlock layer
        out = self.block(x)
        # Flatten the output of the block for the linear layer
        out = out.view(out.size(0), -1)
        # Fully connected layer for classification
        out = self.linear(out)
        return out

