import torch
import pickle
import warnings
import argparse
from tqdm import tqdm
import torch.nn as nn
from primitive import *
from model import Pct_reg
from reconstruct_utils import *
from data_utils import pc_scale
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.optimize import dual_annealing
from primitive_estimate_utils import estimate_primitive

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


class RoofN3D_reg(Dataset):
    def __init__(self):
        fp = 'data/RoofN3D.pkl'
        file = open(fp, 'rb')
        dataset = pickle.load(file)
        self.data_list = []

        for item in dataset.keys():
            data_ = dataset[item]
            points = data_['pc']
            samples = smaple_planes([points])
            scale, t, sample = pc_scale(samples[0])
            cls = data_['roof_type_code']
            width, length = data_['width'], data_['length']
            if cls == 0:
                translation = data_['translation']
                outline = data_['outline']
                self.data_list.append([np.float32(sample),
                                       np.float32(scale),
                                       np.float32(t),
                                       np.float32(cls),
                                       np.float32(width),
                                       np.float32(length),
                                       np.float32(outline)])

        self.data_size = len(self.data_list)
        print(f'dataset size: {self.data_size}')


    def __getitem__(self, item):
        return (self.data_list[item][0], self.data_list[item][1],
                self.data_list[item][4], self.data_list[item][5],
                self.data_list[item][6])

    def __len__(self):
        return self.data_size


if __name__ == "__main__":
    exp_name = 'roof_primitive_sigma_5cm'
    batch_size = 16
    device = torch.device("cuda")

    test_dataset = RoofN3D_reg()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())
        dim_num = [primitive_dict[roof_type]['para_num'] for roof_type in roof_types]
    dim_num = [dim_num[0]]

    models = []
    for idx, dim in enumerate(dim_num):
        model_reg_fp = f'checkpoints/{exp_name}/models/model_reg_{roof_types[idx]}.t7'
        print(f'Loading model from {model_reg_fp}')

        model_reg = nn.DataParallel(Pct_reg(args, dim).to(device))
        model_reg.load_state_dict(torch.load(model_reg_fp))
        model_reg.eval()
        models.append(model_reg)

    for item in tqdm(test_loader, desc=f'Evaluating regression task for {roof_types[0]} '):
        data, scale, width, length, outline = item
        # data = item
        data = data.to(device)
        data = data.permute(0, 2, 1)
        logits_reg = models[0](data)
        preds = logits_reg.cpu().detach().numpy()
        samples = data.cpu().detach().numpy()
        widths, lengths = width.cpu().detach().numpy(), length.cpu().detach().numpy()
        scales = scale.cpu().detach().numpy()

        func = locals()[primitive_dict[roof_types[0]]['func_name']]

        for j in range(len(data)):
            w, l = widths[j], lengths[j]
            scale = scales[j]
            pred, sample = preds[j] * scale, samples[j].T * scale

            # RMSE, MTD, para_est = estimate_primitive(sample, primitive_dict, func,
            #                                          roof_types[0], vis=True, sample_num=128,
            #                                          ambiguity=primitive_dict[roof_types[0]]['ambiguity'])

            para_pred = [pred[0], pred[1], 1, 1, pred[2]]
            verts_pred, faces, _ = func(para_pred, True, False)

            rst = dual_annealing(obj, [(- 2 * np.pi, 2 * np.pi)], x0=np.array([0]),
                                 args=([verts_pred, faces, sample],))
            verts_pred = pc_RT(verts_pred, rst.x[0])

            mesh_tri = trimesh.Trimesh(vertices=verts_pred, faces=faces)
            dists = trimesh.proximity.signed_distance(mesh_tri, sample)
            RMSE = np.sqrt(np.mean(np.square(dists)))

            faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
            pl = pv.Plotter()
            pl.add_mesh(pv.PolyData(verts_pred, faces_pl), color='red', opacity=0.5, show_edges=True)
            pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=5, color='blue')
            pl.show()
            c = 1
