from __future__ import print_function
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from model import Pct
import torch.nn as nn
from primitive import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import Roof_Primitive_cls as Roof_Primitive
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

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
    exp_name = 'roof_primitive'
    model_cls_fp = f'checkpoints/{exp_name}/models/model_cls.t7'

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())

    batch_size, cls_num = 16, len(roof_types)
    test_dataset = Roof_Primitive(data_fd=f'data/{exp_name}', partition='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda")
    model_cls = Pct(args, len(roof_types)).to(device)
    model_cls = nn.DataParallel(model_cls)
    cpt_cls = torch.load(model_cls_fp)
    model_cls.load_state_dict(cpt_cls)
    model_cls.eval()
    y_true, y_pred = [], []
    logit_all, logit_centers = {}, {}

    for i in range(cls_num):
        for j in range(cls_num):
            logit_all[f'{i},{j}'] = []

    for _, item in enumerate(tqdm(test_loader, desc='Testing cls model...')):
        data, label = item
        data, label = data.to(device), label.to(device).squeeze()
        test_pred_logits = model_cls(data.permute(0, 2, 1))
        test_pred_logits_np = test_pred_logits.detach().cpu().numpy()
        test_pred_probs_all = F.softmax(test_pred_logits).detach().cpu().numpy()

        test_pred_cls = test_pred_probs_all.argmax(axis=1)
        test_pred_probs = test_pred_probs_all.max(axis=1)
        test_true_cls = label.max(dim=1)[1].cpu().numpy()

        for i in range(len(test_true_cls)):
            cls_pred = test_pred_cls[i]
            cls_true = test_true_cls[i]
            logit_all[f'{cls_pred},{cls_true}'].append(test_pred_logits_np[i])

        y_true += list(test_true_cls)
        y_pred += list(test_pred_cls)

    # for i in range(cls_num):
    #     for j in range(cls_num):
    #         logit_all[f'{i},{j}'] = np.array(logit_all[f'{i},{j}'])
    #         if len(logit_all[f'{i},{j}']) > 0:
    #             logit_centers[f'{i},{j}'] = np.mean(logit_all[f'{i},{j}'], axis=0)
    #             fig, ax = plt.subplots()
    #             ax.boxplot(logit_all[f'{i},{j}'])
    #             ax.set_xticklabels(roof_types, rotation=-45, fontsize='xx-small')
    #             ax.set_ylabel('Logit')
    #             ax.set_ylim([- 20, 20])
    #             ax.set_title(f'Box plot of Logits, Prediction: {roof_types[i]}, True: {roof_types[j]}')
    #             fig.savefig(f'cls_img/logit_{roof_types[i]}_{roof_types[j]}', dpi=600)

    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=roof_types)
    print(f"Accuracy = {accuracy_score(y_true, y_pred)}")
    print(f"Precision = {precision_score(y_true, y_pred, average=None)}")
    print(f"Recall = {recall_score(y_true, y_pred, average=None)}")
    disp.plot()
    plt.show()
    c = 1

    # y_true, y_pred, case_centers = [], [], {}
    # keys = [item for item in logit_centers.keys()]
    #
    # for i in range(cls_num):
    #     for j in range(cls_num):
    #         logits = logit_centers[f'{j},{j}'].copy()
    #         if f'{i},{j}' in keys:
    #             case_centers[f'{i},{j}'] = logit_centers[f'{i},{j}']
    #         # else:
    #         #     logit = (abs(logits[i]) + abs(logits[j])) / 2
    #         #     logits[i] = logit
    #         #     logits[j] = logit
    #         #     case_centers[f'{i},{j}'] = logits
    #
    # keys = [item for item in case_centers.keys()]
    # centers = np.array([case_centers[item] for item in case_centers.keys()])
    #
    # for _, item in enumerate(tqdm(test_loader, desc='Testing cls model...')):
    #     data, label, paras, _, _ = item
    #     data, label, paras = data.to(device), label.to(device).squeeze(), paras.to(device).squeeze()
    #     test_pred_logits = model_cls(data.permute(0, 2, 1))
    #     test_pred_logits_np = test_pred_logits.detach().cpu().numpy()
    #     test_pred_probs_all = F.softmax(test_pred_logits).detach().cpu().numpy()
    #
    #     test_pred_cls = test_pred_probs_all.argmax(axis=1)
    #     test_pred_probs = test_pred_probs_all.max(axis=1)
    #     test_true_cls = label.max(dim=1)[1].cpu().numpy()
    #
    #     for i in range(len(test_true_cls)):
    #         cossim = cosine_similarity([test_pred_logits_np[i]], centers)
    #         case_label = keys[np.argmax(cossim)]
    #         cls_pred = int(case_label.split(',')[0])
    #         cls_true = int(case_label.split(',')[1])
    #         if cls_true != test_pred_cls[i]:
    #             c = 1
    #         test_pred_cls[i] = cls_pred
    #
    #     y_true += list(test_true_cls)
    #     y_pred += list(test_pred_cls)
    #
    # cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=roof_types)
    # print(f"Accuracy = {accuracy_score(y_true, y_pred)}")
    # print(f"Precision = {precision_score(y_true, y_pred, average=None)}")
    # print(f"Recall = {recall_score(y_true, y_pred, average=None)}")
    # c = 1
