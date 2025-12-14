from __future__ import print_function
import os
import time
import torch
import argparse
import warnings
import numpy as np
import open3d as o3d
import torch.nn as nn
from model import Pct
from util import IOStream
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ISPRS_Vaihingen_primitive
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

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
    train_loader = DataLoader(ISPRS_Vaihingen_primitive(data_fd='data/' + args.exp_name, partition='train'),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ISPRS_Vaihingen_primitive(data_fd='data/' + args.exp_name, partition='test'),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    out_dim = 32
    model = Pct(args, output_channels=out_dim * 3).to(device)
    print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion_vertex = nn.MSELoss(reduction='sum').cuda()
    # criterion_vertex = vert_loss().cuda()

    best_test_loss = 999999

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        train_vert_RMSE = 0.0
        count = 0.0
        model.train()
        idx = 0
        train_time = 0.0
        for data, label, scale in (train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion_vertex(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            train_time += (end_time - start_time)

            count += batch_size
            train_loss += loss.item() * batch_size
            scale = scale.numpy()
            vert_true = label.cpu().numpy()
            vert_true = np.einsum('ij,i->ij', vert_true, scale).reshape(batch_size, out_dim, 3)
            vert_pred = logits.detach().cpu().numpy()
            vert_pred = np.einsum('ij,i->ij', vert_pred, scale).reshape(batch_size, out_dim, 3)
            for data_idx in range(batch_size):
                vt = vert_true[data_idx]
                vp = vert_pred[data_idx]
                train_vert_RMSE += np.sqrt(np.square(np.linalg.norm(vp - vt, axis=1)).sum() / vp.shape[0])

            if idx == 0:
                pc = data.cpu().numpy()[0].T * scale[0]
                vt, vp = vert_true[0], vert_pred[0]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.paint_uniform_color([0.5, 0.5, 0.5])

                pcd_vt = o3d.geometry.PointCloud()
                pcd_vt.points = o3d.utility.Vector3dVector(vt)
                pcd_vt.paint_uniform_color([0.0, 1.0, 0.0])

                pcd_vp = o3d.geometry.PointCloud()
                pcd_vp.points = o3d.utility.Vector3dVector(vp)
                pcd_vp.paint_uniform_color([1.0, 0.0, 0.0])
                writer.add_3d('Raw PC', to_dict_batch([pcd]), step=epoch)
                writer.add_3d('Train Ground Truth', to_dict_batch([pcd_vt]), step=epoch)
                writer.add_3d('Train Prediction', to_dict_batch([pcd_vp]), step=epoch)

            idx += 1

        train_loss = train_loss * 1.0 / count
        train_vert_RMSE = train_vert_RMSE * 1.0 / count

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_vert_RMSE = 0.0
        count = 0.0
        model.eval()
        test_time = 0.0
        for data, label, scale in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            test_time += (end_time - start_time)
            loss = criterion_vertex(logits, label)

            count += batch_size
            test_loss += loss.item() * batch_size
            scale = scale.numpy()
            vert_true = label.cpu().numpy()
            vert_true = np.einsum('ij,i->ij', vert_true, scale).reshape(batch_size, out_dim, 3)
            vert_pred = logits.detach().cpu().numpy()
            vert_pred = np.einsum('ij,i->ij', vert_pred, scale).reshape(batch_size, out_dim, 3)
            for data_idx in range(batch_size):
                vt = vert_true[data_idx]
                vp = vert_pred[data_idx]
                test_vert_RMSE += np.sqrt(np.square(np.linalg.norm(vp - vt, axis=1)).sum() / vp.shape[0])
        test_loss = test_loss * 1.0 / count
        test_vert_RMSE = test_vert_RMSE * 1.0 / count

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Vert_RMSE/train (m)', train_vert_RMSE, epoch)
        writer.add_scalar('Vert_RMSE/test (m)', test_vert_RMSE, epoch)
        outstr = (f'Epoch {epoch}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, '
                  f'train vert RMSE: {train_vert_RMSE:.4f}, test vert RMSE: {test_vert_RMSE:.4f}, '
                  f'train time: {train_time:.4f}, test time:{test_time:.4f}')
        io.cprint(outstr)
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='ISPRS_Vaihingen', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='ISPRS_Vaihingen', metavar='N')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
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
    parser.add_argument('--num_points', type=int, default=1024,
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

    # tensorboard --logdir checkpoints/ISPRS_Vaihingen

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
