from __future__ import print_function
import numpy as np
import torch
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from primitive import *
from model import Pct_reg
from data import get_rotation
from reconstruct_utils import *
import matplotlib.pyplot as plt
from data import Roof_Primitive_reg
from torch.utils.data import DataLoader
from scipy.optimize import dual_annealing

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='roof_primitive', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--dataset', type=str, default='roof_primitive', metavar='N')
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


if __name__ == "__main__":
    exp_name = 'roof_primitive_perfect_0cm'
    batch_size, coef = 16, 1.000
    device = torch.device("cuda")
    vis_flag, CS_flag, CR_flag = False, False, False
    out_types = []

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())
        dim_num = [primitive_dict[roof_type]['para_num'] for roof_type in roof_types]
    dim_num = [dim_num[0]]

    models, train_loaders, test_loaders = [], [], []
    DTM_all, MTD_all =[], []
    for idx, dim in enumerate(dim_num):
        test_dataset = Roof_Primitive_reg(data_fd='data/' + exp_name, partition='train',
                                          primitive=roof_types[idx], primitive_info=primitive_dict[roof_types[idx]])
        test_loaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False))
        model_reg_fp = f'checkpoints/{exp_name}/models/model_reg_{roof_types[idx]}.t7'
        print(f'Loading model from {model_reg_fp}')

        model_reg = nn.DataParallel(Pct_reg(args, dim).to(device))
        pytorch_total_params = sum(p.numel() for p in model_reg.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {pytorch_total_params}')
        model_reg.load_state_dict(torch.load(model_reg_fp))
        model_reg.eval()
        models.append(model_reg)

    for idx, model_reg in enumerate(models):
        pnum, inum, DTMs, MTDs = 0, 0, [], []
        kappa_err, para_err, fitting_RMSE, para_prop, corner_RMSE, GSD = [], [], [], [], [], []
        para_pred_all, para_GT_all = [], []
        out_types.append(roof_types[idx])
        for item in tqdm(test_loaders[idx], desc=f'Evaluating regression task for {roof_types[idx]} '):
            data, label, paras, scales, translations, paras_true = item
            data = data.to(device)
            data = data.permute(0, 2, 1)
            logits_reg = model_reg(data)

            samples = data.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
            preds = logits_reg.cpu().detach().numpy()
            gts = paras.cpu().detach().numpy()
            scales = scales.cpu().detach().numpy()
            translations = translations.cpu().detach().numpy()
            gts_true = paras_true.cpu().detach().numpy()

            inum += 1
            if inum > 1:
                continue

            for j in range(len(samples)):
                sample, gt = samples[j].T, gts[j]
                pred, gt_scale, gt_t = preds[j], scales[j], translations[j]

                func = locals()[primitive_dict[roof_types[idx]]['func_name']]
                cls = primitive_dict[roof_types[idx]]['code']
                gt_true = gts_true[j]

                if CS_flag or CR_flag:
                    scale = 1
                else:
                    # scale = np.mean(gt_true[[0, 1]] / gt[[1, 2]])
                    scale = np.mean(gt_true[[0, 1]] / gt[[0, 1]])

                r_pred = get_rotation(primitive_dict[roof_types[idx]]['cycle'], pred[0])
                r_gt = get_rotation(primitive_dict[roof_types[idx]]['cycle'], gt[0])

                if primitive_dict[roof_types[idx]]['para_num'] == 3:
                    # para_pred = [pred[1] * scale * coef, pred[2] * scale * coef, 1, 1, pred[3] * scale * coef]
                    # para_gt = [gt[1] * scale, gt[2] * scale, 1, 1, gt[3] * scale]
                    # para_pred_all = np.append(para_pred_all,
                    #                           [pred[1] * scale, pred[2] * scale, pred[3] * scale])
                    # para_GT_all = np.append(para_GT_all,
                    #                         [gt[1] * scale, gt[2] * scale, gt[3] * scale])
                    para_pred = [pred[0] * scale * coef, pred[1] * scale * coef, 1, 1, pred[2] * scale * coef]
                    para_gt = [gt[0] * scale, gt[1] * scale, 1, 1, gt[2] * scale]
                    para_pred_all = np.append(para_pred_all,
                                              [pred[0] * scale, pred[1] * scale, pred[2] * scale])
                    para_GT_all = np.append(para_GT_all,
                                            [gt[0] * scale, gt[1] * scale, gt[2] * scale])
                elif primitive_dict[roof_types[idx]]['para_num'] == 4:
                    # para_pred = [pred[1] * scale * coef, pred[2] * scale * coef, 1, pred[3], pred[4] * scale * coef]
                    # para_gt = [gt[1] * scale, gt[2] * scale, 1, gt[3], gt[4] * scale]
                    # para_pred_all = np.append(para_pred_all,
                    #                           [pred[1] * scale, pred[2] * scale, pred[3], pred[4] * scale])
                    # para_GT_all = np.append(para_GT_all,
                    #                         [gt[1] * scale, gt[2] * scale, gt[3], gt[4] * scale])
                    para_pred = [pred[0] * scale * coef, pred[1] * scale * coef, 1, pred[2], pred[3] * scale * coef]
                    para_gt = [gt[0] * scale, gt[1] * scale, 1, gt[2], gt[3] * scale]
                    para_pred_all = np.append(para_pred_all,
                                              [pred[0] * scale, pred[1] * scale, pred[2], pred[3] * scale])
                    para_GT_all = np.append(para_GT_all,
                                            [gt[0] * scale, gt[1] * scale, gt[2], gt[3] * scale])
                else:
                    # para_pred = [pred[1] * scale * coef, pred[2] * scale * coef, pred[3], pred[4], pred[5] * scale * coef]
                    # para_gt = [gt[1] * scale, gt[2] * scale, gt[3], gt[4], gt[5] * scale]
                    # para_pred_all = np.append(para_pred_all,
                    #                           [pred[1] * scale, pred[2] * scale, pred[3], pred[4], pred[5] * scale])
                    # para_GT_all = np.append(para_GT_all,
                    #                         [gt[1] * scale, gt[2] * scale, gt[3], gt[4], gt[5] * scale])
                    para_pred = [pred[0] * scale * coef, pred[1] * scale * coef, pred[2], pred[3],
                                 pred[4] * scale * coef]
                    para_gt = [gt[0] * scale, gt[1] * scale, gt[2], gt[3], gt[4] * scale]
                    para_pred_all = np.append(para_pred_all,
                                              [pred[0] * scale, pred[1] * scale, pred[2], pred[3], pred[4] * scale])
                    para_GT_all = np.append(para_GT_all,
                                            [gt[0] * scale, gt[1] * scale, gt[2], gt[3], gt[4] * scale])

                verts_pred, faces, _ = func(para_pred, True, False)
                verts_gt, _, _ = func(para_gt, True, False)

                if vis_flag:
                    verts_pred = pc_RT(verts_pred, r_pred)
                    verts_gt = pc_RT(verts_gt, r_gt)
                    sample = (sample + gt_t) * gt_scale
                    if pnum <= 10:
                        faces_pl = [[3, face[0], face[1], face[2]] for face in faces]

                        pl = pv.Plotter(shape=(1, 2))
                        pl.set_background('grey')

                        pl.subplot(0, 0)
                        pl.add_text(f'Prediction, Primitive type: {roof_types[cls]}')
                        pl.add_mesh(pv.PolyData(verts_pred, faces_pl), color='red', opacity=0.5, show_edges=True)
                        pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=5, color='blue')

                        pl.subplot(0, 1)
                        pl.add_text(f'Ground Truth, Primitive type: {roof_types[cls]}')
                        pl.add_mesh(pv.PolyData(verts_gt, faces_pl), color='green', opacity=0.5, show_edges=True)
                        pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=5, color='blue')

                        pl.link_views()
                        pl.show()

                        pnum += 1
                    else:
                        break
                else:
                    if CS_flag:
                        rst = dual_annealing(obj, [(- 2 * np.pi, 2 * np.pi)], x0=np.array([0]),
                                             args=([verts_pred, faces, sample],))
                        verts_pred = pc_RT(verts_pred, rst.x[0])
                        DTM, MTD = get_reconstruction_score(verts_pred, faces, sample)
                        DTMs.append(DTM)
                        MTDs.append(MTD)
                    elif CR_flag:
                        continue
                    else:
                        sample = (sample + gt_t) * gt_scale
                        rst = dual_annealing(obj, [(- 2 * np.pi, 2 * np.pi)], x0=np.array([0]),
                                             args=([verts_pred, faces, sample],))
                        verts_pred = pc_RT(verts_pred, rst.x[0])
                        DTM = get_DTM(verts_pred, faces, sample)
                        MTD = get_MTD(verts_pred, sample)
                        fitting_RMSE.append(DTM)
                        corner_RMSE.append(MTD)
                        GSD.append(np.sqrt(para_gt[0] * para_gt[1] / 512))
                        kappa_err.append(np.rad2deg(rst.x[0] - r_gt))

                        # fitting_RMSE.append(0)
                        # kappa_err.append(0)

                        para_err.append(np.array(para_pred) - np.array(para_gt))
                        para_prop.append(abs(np.array(para_pred) - np.array(para_gt)) / np.array(para_gt))

                    pnum += 1

        if CS_flag:
            DTM_all.append(DTMs)
            MTD_all.append(MTDs)
        elif CR_flag:
            print(f'Primitive type: {roof_types[idx]}')
            print(f'Parameter R Square: {np.corrcoef(para_GT_all, para_pred_all)[0, 1] ** 2:.4f}')
        else:
            para_err = np.array(para_err)
            para_prop = np.array(para_prop)
            corner_RMSE_prop = np.array(corner_RMSE) / np.array(GSD)
            shape_err = np.concatenate((para_err[:, 0], para_err[:, 1], para_err[:, -1]))
            para_prop = np.concatenate((para_prop[:, 0], para_prop[:, 1], para_prop[:, -1]))

            print(f'Primitive type: {roof_types[idx]}')
            print(f'Fitting RMSE Mean: {np.mean(fitting_RMSE):.4f}')
            print(f'Fitting RMSE STD: {np.std(fitting_RMSE):.4f}')
            print(f'Corners RMSE Mean: {np.mean(corner_RMSE):.4f}')
            print(f'Corners RMSE STD: {np.std(corner_RMSE):.4f}')
            print(f'GSD Mean: {np.mean(GSD):.4f}')
            print(f'GSD STD: {np.std(GSD):.4f}')
            print(f'Corners RMSE Prop Mean: {np.mean(corner_RMSE_prop):.4f}')
            print(f'Corners RMSE Prop STD: {np.std(corner_RMSE_prop):.4f}')
            print(f'Kappa Error Mean: {np.mean(kappa_err):.4f}')
            print(f'Kappa Error STD: {np.std(kappa_err):.4f}')
            print(f'Kappa Error RMSE: {np.sqrt(np.mean(np.square(kappa_err))):.4f}')
            print(f'Shape Error Mean: {np.mean(shape_err):.4f}')
            print(f'Shape Error STD: {np.std(shape_err):.4f}')
            print(f'Shape Error RMSE: {np.sqrt(np.mean(np.square(shape_err))):.4f}')
            print(f'Shape Error Prop Mean: {np.mean(para_prop):.4f}')

            # ds = np.array([para_err[:, 0], para_err[:, 1], para_err[:, -1]]).T
            # ds[:, 0] = (ds[:, 0] - ds[:, 0].mean()) / ds[:, 0].std() * 0.0042
            # ds[:, 1] = (ds[:, 1] - ds[:, 1].mean()) / ds[:, 1].std() * 0.0042
            # ds[:, 2] = (ds[:, 2] - ds[:, 2].mean()) / ds[:, 2].std() * 0.0042
            #
            # if idx == 1:
            #     fig, ax = plt.subplots()
            #     ax.hist(ds, bins=16, alpha=0.6, histtype='bar', color=['r', 'g', 'b'], label=['dw', 'dl', 'dh'],
            #             align='mid')
            #     ax.set_xlabel('Error (unit:m)')
            #     ax.set_ylabel('Frequency')
            #     ax.set_title('3-LBR-PCT without RELU Primitive Shape Parameters Error Distribution (sigma=0 cm)')
            #     ax.legend()
            #     ax.set_xlim([-0.025, 0.025])
            #     plt.show()
            #     c = 1

            if primitive_dict[roof_types[idx]]['para_num'] == 4 or primitive_dict[roof_types[idx]]['para_num'] == 5:
                aspect_err = para_err[:, 3] if primitive_dict[roof_types[idx]]['para_num'] == 4 else np.concatenate(
                    (para_err[:, 2], para_err[:, 3]))
                print(f'Aspect Error Mean: {np.mean(aspect_err):.4f}')
                print(f'Aspect Error STD: {np.std(aspect_err):.4f}')
                print(f'Aspect Error RMSE: {np.sqrt(np.mean(np.square(aspect_err))):.4f}')

    if CS_flag:
        DTM_df = pd.DataFrame(data=np.array(DTM_all).T, columns=[item + '_DTM' for item in out_types])
        MTD_df = pd.DataFrame(data=np.array(MTD_all).T, columns=[item + '_MTD' for item in out_types])
        df = pd.concat([DTM_df, MTD_df], axis=1)
        df.to_csv('reconstruction_scores.csv', index=False)