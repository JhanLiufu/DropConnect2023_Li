import torch.nn as nn
import torch.nn.functional as F
import torch


class DCConv2d(nn.Conv2d):
    def __init__(self, drop_prob, in_channels, out_channels, kernel_size, stride, padding, bias, groups=1):
        super(DCConv2d, self).__init__(in_channels, out_channels,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=bias, groups=groups)
        self.drop_prob = drop_prob
        # self.weight_mask = generate_weight_mask(self.weight.shape, drop_prob=drop_prob).to(self.weight.device)
        self.weight_mask = generate_weight_mask(self.weight.shape, drop_prob=drop_prob).to(self.weight.device)
        # print("During initialization, self.weight.device is " + str(self.weight.device))
        # print("During initialization, self.weight_mask.device is " + str(self.weight_mask.device))

    def forward(self, input):
        if self.training:
            weight = F.dropout(self.weight, p=self.drop_prob)
        else:
            # if cuda is available, move to cuda ...
            if torch.cuda.is_available():
                # print("During inference (before moving), self.weight.device is " + str(self.weight.device))
                # print("During inference (before moving), self.weight_mask.device is " + str(self.weight_mask.device))
                self.weight_mask = self.weight_mask.to(self.weight.device)
                # print("During inference, self.weight.device is " + str(self.weight.device))
                # print("Now self.weight_mask.device is " + str(self.weight_mask.device))
            weight = self.weight * self.weight_mask

        output = F.conv2d(input, weight=weight, stride=self.stride,
                          padding=self.padding, groups=self.groups)
        return output


def generate_weight_mask(shape, drop_prob=0.1):
    '''
    Function to generate a random weight mask with dropout probability drop_prob
    '''
    # Generate a random mask with values in the range [0, 1)
    mask = torch.rand(shape)
    # Apply dropout by setting values <= drop_prob to 0 and scaling the remaining values
    mask = (mask >= drop_prob).int()
    return mask
