import argparse
import os
import importlib
import inspect
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from recorder import Plot
from training_utils import train, validate, save_checkpoint

model_path = 'models'
model_names = sorted([os.path.splitext(f)[0] for f in os.listdir(model_path) if f.endswith('.py')])
models = {}

for m in model_names:
    module = importlib.import_module(model_path + '.' + m)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if model_path + '.' + name == module.__name__:
                models[m] = obj

nn_datasets = ['mnist', 'cifar10']

parser = argparse.ArgumentParser(description='PyTorch Training and Testing')
parser.add_argument('--dataset', default='mnist', choices=nn_datasets,
                    help='dataset (default: mnist)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='LeNet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: LeNet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--milestones', nargs='+', default=None, type=int,
                    help='milestones for learning rate decay (default: None)')
parser.add_argument('--lr-step', default=50, type=int, metavar='N',
                    help='learning rate decay steps (default: 50)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='M',
                    help='gamma for learning rate decay (default: 0.1)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['SGD', 'Adam'], help='optimizer')
parser.add_argument('--json', default=None, type=str, metavar='PATH',
                    help='path to the JSON configuration (default: none)')

# for ResNet block setting argument
parser.add_argument('--block_setting', default=None, type=list, metavar='BS',
                    help='DCResNet block configuration (default: None -> see model def)')

best_acc1 = 0


def main():
    global best_acc1
    learning_curve_path = 'learning_curve'
    checkpoints_path = 'checkpoints'

    args = parser.parse_args()

    with open(args.json, 'r') as f:
        args_from_json = json.load(f)

    # Use the dictionary to set the arguments
    for key, value in args_from_json.items():
        if hasattr(args, key):
            setattr(args, key, value)

    print(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("Use GPU: {} for training".format(args.gpu))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.block_setting is not None:
        model = models[args.arch](block_setting=args.block_setting)
    else:
        model = models[args.arch]()
    print(model)

    try:
        learning_curve_path = os.path.join(learning_curve_path, model.name + '_learning_curve.png')
        checkpoints_path = os.path.join(checkpoints_path, model.name)
    except AttributeError:
        # if model doesn't define its own name
        learning_curve_path = os.path.join(learning_curve_path, args.arch + '_learning_curve.png')
        checkpoints_path = os.path.join(checkpoints_path, args.arch)

    if torch.cuda.is_available():
        print("GPU is available")
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("mps is available")
        device = torch.device("mps")
    else:
        print("Use CPU")
        device = torch.device("cpu")

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

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   
    if args.milestones is None:
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.milestones,
                                gamma=args.gamma, last_epoch=args.start_epoch - 1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), download=True)

        val_dataset = datasets.MNIST('./data', train=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), download=True)
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

        val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        raise Exception("Dataset not supported")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    plt = Plot()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        t_acc1, t_loss = train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        v_acc1, v_loss = validate(val_loader, model, criterion, args)

        plt.record(t_acc1.item(), t_loss, v_acc1.item(), v_loss)
        plt.plot(learning_curve_path)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = v_acc1 > best_acc1
        best_acc1 = max(v_acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, checkpoints_path)


if __name__ == '__main__':
    main()