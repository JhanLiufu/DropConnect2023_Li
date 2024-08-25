import argparse
import os
import importlib
import inspect
import warnings
import shutil
import time
import json

from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from recorder import AverageMeter, ProgressMeter, Summary, Plot

model_path = 'models'
model_names = sorted([os.path.splitext(f)[0] for f in os.listdir(model_path) if f.endswith('.py')])
models = {}

for m in model_names:
    module = importlib.import_module(model_path + '.' + m)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if obj.__module__ == module.__name__:
                models[m] = obj

nn_datasets = ['mnist', 'cifar10']

parser = argparse.ArgumentParser(description='FLOPs measurement')
parser.add_argument('-d', '--dataset', default='mnist', choices=nn_datasets,
                    help='dataset (default: mnist)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='LeNet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: LeNet)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models[args.arch]()
    print(model)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data loading code
    if args.dataset == 'mnist':
        input = torch.rand([1, 1, 28, 28])
    elif args.dataset == 'cifar10':
        input = torch.rand([1, 3, 32, 32])
    else:
        raise Exception("Dataset not supported")

    input = input.to(device)

    print(f"{args.arch} FLOP count:")
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))

if __name__ == '__main__':
    main()