from __future__ import print_function
import os
import json
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from model import Pct
import torch.nn as nn
from util import IOStream
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as metrics
from data import Roof_Primitive_cls
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    writer = SummaryWriter()
    train_dataset = Roof_Primitive_cls(data_fd='data/' + args.exp_name, partition='train')
    test_dataset = Roof_Primitive_cls(data_fd='data/' + args.exp_name, partition='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())

    print(f'Roof primitive types: {roof_types}')
    device = torch.device("cuda" if args.cuda else "cpu")
    model_cls = Pct(args, len(roof_types)).to(device)
    model_cls = nn.DataParallel(model_cls)

    if args.use_sgd:
        print("Use SGD")
        opt_cls = optim.SGD(model_cls.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt_cls = optim.Adam(model_cls.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler_cls = CosineAnnealingLR(opt_cls, args.epochs, eta_min=args.lr)
    criterion_cls = nn.CrossEntropyLoss(reduction='mean').cuda()

    best_test_loss = 999999

    for epoch in range(args.epochs):
        scheduler_cls.step()
        train_loss = 0.0
        count = 0.0
        model_cls.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch}, training'):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt_cls.zero_grad()

            logits_cls = model_cls(data)
            probs = F.softmax(logits_cls)
            loss_cls = criterion_cls(logits_cls, label)
            loss_cls.backward()

            opt_cls.step()

            preds = probs.max(dim=1)[1]
            true = label.max(dim=1)[1]
            train_true.append(true.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            count += batch_size
            train_loss += np.sum(loss_cls.item())

        train_loss = train_loss * 1.0 / count
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model_cls.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader, desc=f'Epoch {epoch}, testing'):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits_cls = model_cls(data)
            probs = F.softmax(logits_cls)
            loss_cls = criterion_cls(logits_cls, label)
            preds = probs.max(dim=1)[1]
            true = label.max(dim=1)[1]
            test_true.append(true.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            count += batch_size
            test_loss += np.sum(loss_cls.item())

        test_loss = test_loss * 1.0 / count
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        writer.add_scalar('Cls_Loss/train', train_loss, epoch)
        writer.add_scalar('Cls_Loss/test', test_loss, epoch)
        writer.add_scalar('ACC/train', train_acc, epoch)
        writer.add_scalar('ACC/test', test_acc, epoch)
        outstr = (f'Epoch {epoch}, train cls_loss: {train_loss:.4f}, test cls_loss: {test_loss:.4f}, '
                  f'train acc:{train_acc:.4f}, test acc:{test_acc:.4f}')
        io.cprint(outstr)
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            torch.save(model_cls.state_dict(), 'checkpoints/%s/models/model_cls.t7' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='roof_primitive', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='roof_primitive', metavar='N')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--clear_log', type=bool, default=True,
                        help='Clear log rile')
    args = parser.parse_args()

    log_fp = 'checkpoints/' + args.exp_name + '/run.log'
    if args.clear_log and os.path.exists(log_fp):
        os.remove(log_fp)

    _init_()

    io = IOStream(log_fp)
    io.cprint(str(args))

    # tensorboard --logdir runs

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
