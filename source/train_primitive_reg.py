from __future__ import print_function
import json
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from data_utils import *
from util import IOStream
from model import Pct_reg
import torch.optim as optim
from data import Roof_Primitive_reg
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
    device = torch.device("cuda" if args.cuda else "cpu")

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())
        dim_num = [primitive_dict[roof_type]['para_num'] for roof_type in roof_types]

    models, opts, schedulers, train_loaders, test_loaders = [], [], [], [], []
    for idx, dim in enumerate(dim_num):
        train_dataset = Roof_Primitive_reg(data_fd='data/' + args.exp_name, partition='train',
                                           primitive=roof_types[idx], primitive_info=primitive_dict[roof_types[idx]])
        test_dataset = Roof_Primitive_reg(data_fd='data/' + args.exp_name, partition='test',
                                          primitive=roof_types[idx], primitive_info=primitive_dict[roof_types[idx]])
        train_loaders.append(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
        test_loaders.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False))

        model_reg = nn.DataParallel(Pct_reg(args, dim).to(device))

        if args.use_sgd:
            print("Use SGD")
            opt_reg = optim.SGD(model_reg.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
        else:
            print("Use Adam")
            opt_reg = optim.AdamW(model_reg.parameters(), lr=args.lr, weight_decay=1e-2)

        models.append(model_reg)
        opts.append(opt_reg)
        schedulers.append(CosineAnnealingLR(opt_reg, args.epochs, eta_min=args.lr*0.01))

    criterion_reg = nn.SmoothL1Loss(reduction='sum').cuda()
    best_test_loss = np.ones(len(models)) * 999999

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = np.zeros(len(models))
        count = np.zeros(len(models))
        para_error_all, para_prop_all = [], []

        for idx, model_reg in enumerate(models):
            schedulers[idx].step()
            if idx == 0:
                print(f'\nEpoch {epoch}, Learning rate: {schedulers[idx].get_lr()}\n')
            model_reg.train()
            para_error, para_prop = [], []

            for data, _, paras, scale, _, paras_true in tqdm(train_loaders[idx], desc=f'Epoch {epoch}, {roof_types[idx]}, training'):
                data, paras, paras_true = data.to(device), paras.to(device).squeeze(), paras_true.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opts[idx].zero_grad()

                logits_reg = model_reg(data)
                para_pred, paras_true = logits_reg.detach().cpu().numpy(), paras_true.detach().cpu().numpy()
                scale = scale.detach().cpu().numpy()
                para_pred, paras_true = para_pred[:, [0, 1, -1]], paras_true[:, [0, 1, -1]]
                para_pred = np.array([para_pred[i] * scale[i] for i in range(len(scale))])

                ds = para_pred - paras_true
                para_error.append(ds)
                para_prop.append(np.abs(ds) / paras_true)
                loss_reg = criterion_reg(logits_reg, paras)
                loss_reg.backward()
                opts[idx].step()

                count[idx] += batch_size
                train_loss[idx] += np.sum(loss_reg.item())

            para_error_all.append(np.concatenate(para_error, axis=0))
            para_prop_all.append(np.concatenate(para_prop, axis=0))
        train_loss = train_loss / count
        print(f'\n')

        for idx in range(len(models)):
            primitive_type = roof_types[idx]
            shape_RMSE = np.sqrt(np.mean(np.square(para_error_all[idx])))
            shape_prop_mean = np.mean(para_prop_all[idx])

            print(f'Training Epoch {epoch}, {primitive_type}, shape RMSE: {shape_RMSE:.4f}, '
                  f'shape prop mean: {shape_prop_mean:.4f}')

        print(f'\n')
        ####################
        # Test
        ####################
        test_loss = np.zeros(len(models))
        count = np.zeros(len(models))
        para_error_all, para_prop_all = [], []

        for idx, model_reg in enumerate(models):
            model_reg.eval()
            para_error, para_prop = [], []

            for data, _, paras, scale, _, paras_true in tqdm(test_loaders[idx], desc=f'Epoch {epoch}, {roof_types[idx]}, testing'):
                data, paras, paras_true = data.to(device), paras.to(device).squeeze(), paras_true.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                logits_reg = model_reg(data)
                para_pred, paras_true = logits_reg.detach().cpu().numpy(), paras_true.detach().cpu().numpy()
                scale = scale.detach().cpu().numpy()
                para_pred, paras_true = para_pred[:, [0, 1, -1]], paras_true[:, [0, 1, -1]]
                para_pred = np.array([para_pred[i] * scale[i] for i in range(len(scale))])

                ds = np.abs(para_pred - paras_true)
                para_error.append(ds)
                para_prop.append(ds / paras_true)
                loss_reg = criterion_reg(logits_reg, paras)

                count[idx] += batch_size
                test_loss[idx] += np.sum(loss_reg.item())

            para_error_all.append(np.concatenate(para_error, axis=0))
            para_prop_all.append(np.concatenate(para_prop, axis=0))
        test_loss = test_loss / count
        print(f'\n')

        for idx in range(len(models)):
            primitive_type = roof_types[idx]
            shape_RMSE = np.sqrt(np.mean(np.square(para_error_all[idx])))
            shape_prop_mean = np.mean(para_prop_all[idx])

            print(f'Testing Epoch {epoch}, {primitive_type}, shape RMSE: {shape_RMSE:.4f}, '
                  f'shape prop mean: {shape_prop_mean:.4f}')

        print(f'\n')
        for idx in range(len(models)):
            primitive_type = roof_types[idx]
            writer.add_scalar(f'Reg_Loss_{primitive_type}/train', train_loss[idx], epoch)
            writer.add_scalar(f'Reg_Loss_{primitive_type}/test', test_loss[idx], epoch)
            outstr = (f'Epoch {epoch}, {primitive_type}, '
                      f'train reg_loss: {train_loss[idx]:.4f}, test reg_loss: {test_loss[idx]:.4f}')
            io.cprint(outstr)
            if train_loss[idx] <= best_test_loss[idx]:
                best_test_loss[idx] = train_loss[idx]
                torch.save(models[idx].state_dict(),
                           f'checkpoints/%s/models/model_reg_{primitive_type}.t7' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='roof_primitive_sigma_5cm', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='roof_primitive_sigma_5cm', metavar='N')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
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
