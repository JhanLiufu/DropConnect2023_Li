from enum import Enum
import torch
import matplotlib.pyplot as plt


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Plot(object):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def record(self, t_acc, t_loss, v_acc, v_loss):
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)

    def plot(self, learning_curve_path):
        epochs = range(1, len(self.train_acc) + 1)
        dpi = 100  
        width, height = 1200, 800
        legend_fontsize = 10
        marker_size = 3
        figsize = width / float(dpi), height / float(dpi)

        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        lns1 = ax1.plot(epochs, self.train_loss, 'o-', label='Training Loss', color='tab:brown', markersize=marker_size)
        lns2 = ax1.plot(epochs, self.val_loss, 's-', label='Validation Loss', color='tab:purple', markersize=marker_size)
        ax1.tick_params(axis='y', labelcolor=color)

        ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
        lns3 = ax2.plot(epochs, self.train_acc, 'o-', label='Training Accuracy', color='tab:blue', markersize=marker_size)
        lns4 = ax2.plot(epochs, self.val_acc, 's-', label='Validation Accuracy', color='tab:red', markersize=marker_size)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

        # Combine all the legends into one
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center right', fontsize=legend_fontsize)

        fig.tight_layout()  # to make sure there's no overlap
        plt.title('Training and Validation Statistics')

        plt.tight_layout()
        plt.savefig(learning_curve_path)
        plt.close()
        if learning_curve_path is not None:
            fig.savefig(learning_curve_path)
            print ('---- save figure {} into {}'.format('learning_curve_path', learning_curve_path))
        plt.close(fig)
