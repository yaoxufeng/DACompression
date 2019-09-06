import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of TCP")
parser.add_argument('dataset', type=str, choices=['Office31', 'ImageCLEF'])
parser.add_argument('train_path', type=str)
parser.add_argument('test_path', type=str)
parser.add_argument('checkpoint_dir', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--method', type=str, default="train",
                    help='train method, including train with mixup, mixmatch, just for albation experiments')


# ========================= Learning Configs ==========================
parser.add_argument('--num_epoch', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-opt', '--optimizer', default="SGD", type=str,
                    metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default=None, type=str, choices=['warmup', 'cosine_decay', 'step_decay', 'custom'],
                    metavar='LRSchedule', help='learning rate schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--k', '--k_transformation', default=1, type=int,
                    metavar='W', help='time of transformation for data augmentation')
parser.add_argument('--seed', default=8, type=int)


# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--log_file', '-lf', type=str,
                    metavar='N', help='location to store log information')
parser.add_argument('--tensorboard_file', '-tf', type=str,
                    metavar='N', help='location to store tensorboard information')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=None)
