import torch
import torch.nn as nn
import torchvision


class LeNet(nn.Module):
    """
    The network is listed below:
        + Conv 6 * 1 * 5 * 5
        + AvgPool 6 * 14 * 14
        + Conv 16 * 6 * 5 * 5
        + AvgPool 16 * 5 * 5
        + Conv 120 * 16 * 5 * 5
        + FC 84 * 120
        + FC 10 * 84
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.padding = torchvision.transforms.Pad(2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.padding(x)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x