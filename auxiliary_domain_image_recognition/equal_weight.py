import time
import random
import warnings
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import CompleteLogger
from utils.data import ForeverDataIterator
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import accuracy
from utils.classifier import MultiOutputClassifier
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_datasets, train_target_datasets, val_datasets, test_datasets, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    val_loaders = {name: DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
                   for name, dataset in val_datasets.items()}
    test_loaders = {name: DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
                    for name, dataset in test_datasets.items()}

    train_loaders = [ForeverDataIterator(DataLoader(dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.workers, drop_last=True)) for
                     task_name, dataset in train_source_datasets.items()]
    task_names = train_source_datasets.keys()

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    heads = nn.ModuleDict({
        dataset_name: nn.Linear(backbone.out_features, num_classes) for dataset_name in args.source
    })
    classifier = MultiOutputClassifier(backbone, heads, pool_layer=pool_layer, finetune=not args.scratch).to(device)
    if args.pretrained is not None:
        print("Loading from ", args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        classifier.load_state_dict(checkpoint, strict=False)

    print(heads)
    # define optimizer and lr scheduler
    optimizer = utils.get_optimizer(classifier, args.optimizer, args.lr, args.momentum, args.wd)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = utils.validate_all(test_loaders, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = {}
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_loaders, task_names, classifier, optimizer, epoch, args.iters_per_epoch, args, device)
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = utils.validate_all(val_loaders, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if sum(acc1.values()) > sum(best_acc1.values()):
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = acc1

    print("best_acc1 = {}\navg = {}".format(best_acc1, sum(best_acc1.values()) / len(best_acc1)))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate_all(test_loaders, classifier, args, device)
    print("test_acc1 = {}\navg = {}".format(acc1, sum(acc1.values()) / len(acc1)))

    logger.close()


def train(train_loaders, task_names, model, optimizer, epoch, iters_per_epoch, args, device):
    batch_time = AverageMeter('Time', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    accs = AverageMeter('Acc', ':3.1f')
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, losses, accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(iters_per_epoch):
        # clear grad
        optimizer.zero_grad()

        for task_idx, task_name in enumerate(task_names):
            x, labels = next(train_loaders[task_idx])[:2]
            x = x.to(device)
            labels = labels.to(device)

            # compute output
            y = model(x, task_name)
            loss = F.cross_entropy(y, labels) / len(task_names)
            acc = accuracy(y, labels)[0]

            losses.update(loss.item(), x.size(0))
            accs.update(acc.item(), x.size(0))

            # backward
            loss.backward()

        # do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='equal weight algorithm')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: DomainNet)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size of each domain (default: 8)')
    parser.add_argument('--test-batch-size', default=48, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--pretrained', default=None)
    args = parser.parse_args()
    main(args)
