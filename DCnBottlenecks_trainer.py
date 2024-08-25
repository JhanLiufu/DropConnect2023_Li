import sys
repo_path = r'C:\Users\mengz\Box\Li_Lab\DropConnect\METHODS\dropnet'
sys.path.append(repo_path)

# import argparse
import os
import importlib
import inspect
import warnings
import json
import pickle as pkl

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


class Arguments:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


repo_path = r'C:\Users\mengz\Box\Li_Lab\DropConnect\METHODS\dropnet'
param_folder_path = f'{repo_path}\config'
file_name = r'\DCnBottlenecks'
param_file = f'{param_folder_path}{file_name}.json'
with open(param_file, 'r') as f:
    args_from_json = json.load(f)

args = Arguments(args_from_json)
args.gpu = None
args.milestones = None

model_path = 'models'
module = importlib.import_module(f'{model_path}.{args.arch}')
model_class = None

for name, obj in inspect.getmembers(module):
    if inspect.isclass(obj):
        if model_path + '.' + name == module.__name__:
            print(f'Model {module.__name__} successfully imported')
            model_class = obj

# [[output_dimension_1, stride_1, expand_ratio_1], ....]
model_configs = [
                 # {'drop_probs': 0.1,
                 #  'bottleneck_cfg': [[8, 1, 6],]},
                 # {'drop_probs': 0.1,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.1,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.1,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.2,
                 #  'bottleneck_cfg': [[8, 1, 6], ]},
                 # {'drop_probs': 0.2,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.2,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.2,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.2,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6],]},
                 {'drop_probs': 0.2,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6],]},
                 {'drop_probs': 0.2,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.2,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.3,
                 #  'bottleneck_cfg': [[8, 1, 6], ]},
                 # {'drop_probs': 0.3,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.3,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 # {'drop_probs': 0.3,
                 #  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6],]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6],]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6],]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                     [8, 1, 6], [8, 1, 6],]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                    [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                    [8, 1, 6], [8, 1, 6], [8, 1, 6]]},
                 {'drop_probs': 0.3,
                  'bottleneck_cfg': [[8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                    [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6],
                                    [8, 1, 6], [8, 1, 6], [8, 1, 6], [8, 1, 6]]},

                ]


def main():
    for cfg in model_configs:
        best_acc1 = 0

        # -------------------- create model --------------------
        cur_bottleneck_cfg = cfg.get('bottleneck_cfg')
        cur_drop_probs = cfg.get('drop_probs')
        model = model_class(cur_bottleneck_cfg, drop_probs=cur_drop_probs)

        learning_curve_path = os.path.join(f'{repo_path}\learning_curve',
                                           f'{model.name}_dr{cur_drop_probs}_learning_curve.png')
        checkpoints_path = os.path.join(f'{repo_path}\checkpoints', f'{model.name}_dr{cur_drop_probs}')

        print('Model created')

        # -------------------- GPU settings --------------------
        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')
            print("Use GPU: {} for training".format(args.gpu))

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

        print('GPU and model setup done')

        # -------------------- Data loading --------------------
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

        print('Data ready')

        # -------------------- Model training --------------------
        if args.evaluate:
            validate(val_loader, model, criterion, args)
            continue

        plt = Plot()

        print('Begin training')
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
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, checkpoints_path)

        # -------------------- Save model config --------------------
        try:
            with open(f'{checkpoints_path}\model_config.pkl', 'wb') as f_cfg:
                pkl.dump(cfg, f_cfg)
        except Exception:
            print('Model config pickle dump failed')


if __name__ == '__main__':
    main()