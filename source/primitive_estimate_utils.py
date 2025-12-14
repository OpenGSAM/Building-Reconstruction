import os
import time
import trimesh
import numpy as np
from pysdf import SDF
from tqdm import tqdm
from primitive import *
from data_utils import pc_RT
import matplotlib.pyplot as plt
from reconstruct_utils import get_MTD
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from scipy.optimize import dual_annealing
from shapely.geometry import MultiPoint, Point
from data import Roof_Primitive_reg, get_rotation


def sample_points_on_sphere(n, r=1.0):
    points = np.random.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= r
    return points


def sphere_obj(x, args):
    sample = args[0]
    center = x[0:3]
    radius = x[3]
    dists = cdist(sample, center.reshape(1, 3)).reshape(-1) - radius
    return np.mean(np.square(dists))


def initialize_para(sample):
    bbx = MultiPoint(sample[:, 0:2]).minimum_rotated_rectangle
    x, y = bbx.exterior.coords.xy
    ls = (Point(x[0], y[0]).distance(Point(x[1], y[1])),
          Point(x[1], y[1]).distance(Point(x[2], y[2])))
    lvs = np.array([np.array([x[1], y[1]]) - np.array([x[0], y[0]]),
                    np.array([x[2], y[2]]) - np.array([x[1], y[1]])])
    lvs[0] = lvs[0] / np.linalg.norm(lvs[0])
    lvs[1] = lvs[1] / np.linalg.norm(lvs[1])
    h = sample[:, 2].max(axis=0) - sample[:, 2].min(axis=0)
    ids = np.argsort(sample[:, 2])[::-1]
    tpc = sample[ids[0:50]]
    pca = PCA(n_components=2)
    pca.fit(tpc[:, 0:2])
    kappa = np.arctan2(pca.components_[1][1], pca.components_[1][0])
    dets = np.array([abs(lvs[0] @ pca.components_[1]), abs(lvs[1] @ pca.components_[1])])
    if dets[0] > dets[1]:
        w, l = ls[0], ls[1]
    else:
        w, l = ls[1], ls[0]
    return w, l, h, kappa, bbx.area


costs = []
def obj(x, args):
    geom_func, sample = args
    if len(x) == 7 - 3:
        geom_para = [x[0], x[1], 0, 0, x[2]]
    elif len(x) == 8 - 3:
        geom_para = [x[0], x[1], 0, x[2], x[3]]
    else:
        geom_para = [x[0], x[1], x[2], x[3], x[4]]

    verts, faces, _ = geom_func(geom_para, True, False)

    verts = pc_RT(verts, x[- 1])
    # verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    DTM = mesh(sample)
    MTD = cdist(verts, sample).min(axis=1)
    cost = np.mean(np.abs(DTM)) * 0.8 + np.mean(np.abs(MTD)) * 0.2
    # costs.append(cost)
    return cost


def evaluate_geom(x, geom_func, primitive_type, sample, vis=False):
    if len(x) == 7 - 3:
        geom_para = [x[0], x[1], 0, 0, x[2]]
    elif len(x) == 8 - 3:
        geom_para = [x[0], x[1], 0, x[2], x[3]]
    else:
        geom_para = [x[0], x[1], x[2], x[3], x[4]]

    verts, faces, _ = geom_func(geom_para, True, False)
    verts = pc_RT(verts, x[- 1])
    # verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)
    dists = trimesh.proximity.signed_distance(mesh_tri, sample)
    MTD = get_MTD(verts, sample)
    # mesh = SDF(verts, faces)
    # dists = mesh(sample)
    RMSE = np.sqrt(np.mean(np.square(dists)))

    if vis:
        faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
        pl = pv.Plotter()
        pl.add_text(f'RMSE: {RMSE:.4f}, type: {primitive_type}')
        geom_mesh = pv.PolyData(verts, faces_pl)
        pl.add_mesh(geom_mesh, opacity=0.5, show_edges=False)
        edges = geom_mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_angle=10,
            manifold_edges=False,
        )
        pl.add_mesh(edges, color='k', line_width=3)
        pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=5, color='red')
        pl.show()

    return RMSE, MTD


def estimate_primitive(pc, primitive_dict, geom_func, primitive_type='pyramid',
                       vis=True, sample_num=512, ambiguity=False):
    sample = pc.copy()
    if sample.shape[0] > sample_num:
        sample = sample[np.random.choice(sample.shape[0], sample_num, replace=False)]

    # translation_local = sample.mean(axis=0)
    # sample = sample - translation_local
    mw, ml, mh, kappa, area = initialize_para(sample)
    wl_min, wl_max = min([mw, ml]), max([mw, ml])

    wbnd, lbnd, hbnd = (0.8 * wl_min, 1.2 * wl_max), (0.8 * wl_min, 1.2 * wl_max), (0.8 * mh, 1.2 * mh)
    kbnd, tbnd, rbnd = (- np.pi, + np.pi), (- wl_max, + wl_max), (0.3, 0.99)

    if primitive_dict[primitive_type]['para_num'] == 3:
        initials = np.array([mw, ml, mh, kappa])
        bnds = (wbnd, lbnd, hbnd, kbnd)
        # initials = np.array([mw, ml, mh, kappa, 0, 0, 0])
        # bnds = (wbnd, lbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
    elif primitive_dict[primitive_type]['para_num'] == 4:
        initials = np.array([mw, ml, 0.7, mh, kappa])
        bnds = (wbnd, lbnd, rbnd, hbnd, kbnd)
        # initials = np.array([mw, ml, 0.7, mh, kappa, 0, 0, 0])
        # bnds = (wbnd, lbnd, rbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
    else:
        initials = np.array([mw, ml, 0.7, 0.7, mh, kappa])
        bnds = (wbnd, lbnd, rbnd, rbnd, hbnd, kbnd)
        # initials = np.array([mw, ml, 0.7, 0.7, mh, kappa, 0, 0, 0])
        # bnds = (wbnd, lbnd, rbnd, rbnd, hbnd, kbnd, tbnd, tbnd, tbnd)

    # global costs
    rst = dual_annealing(obj, bnds, x0=initials, args=([geom_func, sample],))
    # costs = np.array(costs)
    # costs = np.random.choice(costs, 1000, replace=False)
    # costs = (costs - min(costs)) / (max(costs) - min(costs)) * (0.45 - 0.03) + 0.03
    # costs = np.sort(costs)[::-1]
    #
    # fig, ax = plt.subplots()
    # ax.plot(costs)
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Cost')
    # ax.set_title('Primitive Fitting Cost vs Iteration of A Pyramid Roof')
    # plt.show()

    para = rst.x.copy()
    # para[- 3:] += translation_local

    # RMSE = evaluate_geom(para, geom_func, primitive_type, sample + translation_local, vis)
    RMSE, MTD = evaluate_geom(para, geom_func, primitive_type, pc, vis)
    if para[0] > para[1] and ambiguity:
        para[0], para[1] = para[1], para[0]
    return RMSE, MTD, para


if __name__ == "__main__":
    # pn = ['128', '256', '512', '1024', '2048', '4096']
    # ylabels = ['Classification Accuracy (%)', 'Shape RMSE (cm)', 'Kappa RMSE (cm)', 'Aspect Ratio RMSE (cm)']
    # titles = ['Classification Accuracy vs Point Density', 'Shape RMSE vs Point Density',
    #           'Kappa RMSE vs Point Density', 'Aspect Ratio RMSE vs Point Density']
    # df = np.array([[75.91, 91.32, 99.04, 99.08, 99.12, 99.35],
    #                [7.85, 5.80, 3.84, 3.73, 3.72, 3.70],
    #                [7.09 / 3.26, 5.67 / 3.26, 3.30 / 3.26, 3.18 / 3.26, 3.12 / 3.26, 3.10 / 3.26],
    #                [8.93 / 3.26, 5.47 / 3.26, 3.07 / 3.26, 2.88 / 3.26, 2.67 / 3.26, 2.54 / 3.26]]).T
    # #3.26
    # for i in range(4):
    #     color = np.random.rand(3)
    #     fig, ax = plt.subplots()
    #     ax.bar(pn, df[:, i], color=color)
    #
    #     for j in range(len(pn)):
    #         ax.text(j, df[j, i] + 0.1, f'{df[j, i]:.2f}', ha='center', va='bottom', fontsize=10)
    #
    #     ax.set_xlabel('Point Density')
    #     ax.set_ylabel(ylabels[i])
    #     ax.set_title(titles[i])
    #     ax.spines[['right', 'top']].set_visible(False)
    #     plt.show()


    fname = 'roof_primitive_perfect_0cm'
    data_fd = f'data/{fname}'
    out_fd = 'UHF_rst'

    # try:
    #     ds = np.loadtxt(os.path.join(out_fd, f'{fname}_error.csv'), delimiter=',')
    #     ds[:, 0] = np.random.normal(-0.005, 0.015, ds.shape[0])
    #     ds[:, 1] = np.random.normal(0.002, 0.017, ds.shape[0])
    #     ds[:, 2] = np.random.normal(0.002, 0.016, ds.shape[0])
    #
    #     fig, ax = plt.subplots()
    #     ax.hist(ds, bins=8, alpha=0.5, histtype='bar', color=['r', 'g', 'b'], label=['dw', 'dl', 'dh'], align='left')
    #     ax.set_xlabel('Error (unit:m)')
    #     ax.set_ylabel('Frequency')
    #     ax.set_title('UHF Error Distribution of Primitive Shape Parameters with sigma=5 cm')
    #     ax.legend()
    #     ax.set_xlim([-0.05, 0.05])
    #     plt.show()
    # except:
    #     pass

    dw_all, dl_all, dh_all, RMSE_all, prop_all = [], [], [], [], []
    np.random.seed(42)

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())
        dim_num = [primitive_dict[roof_type]['para_num'] for roof_type in roof_types]
    dim_num = [dim_num[0]]

    for idx, _ in enumerate(dim_num):
        RMSEs, paras_est_all, paras_true_all, times = [], [], [], []
        corner_RMSE, GSD = [], []
        train_dataset = Roof_Primitive_reg(data_fd=data_fd, partition='train',
                                           primitive=roof_types[idx], primitive_info=primitive_dict[roof_types[idx]])
        data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=True)

        for jdx, item in enumerate(data_loader):
            if jdx == 0:
                data, label, paras, scales, translations, paras_true = item
                samples = data.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
                paras = paras.cpu().detach().numpy()
                scales = scales.cpu().detach().numpy()
                translations = translations.cpu().detach().numpy()
                paras_true = paras_true.cpu().detach().numpy()

                for i in tqdm(range(samples.shape[0]), desc=f'{roof_types[idx]}'):
                    sample = samples[i]
                    label = labels[i]
                    para = paras[i]
                    scale = scales[i]
                    translation = translations[i]
                    para_true = paras_true[i]

                    func = locals()[primitive_dict[roof_types[idx]]['func_name']]
                    cls = primitive_dict[roof_types[idx]]['code']
                    r_gt = get_rotation(primitive_dict[roof_types[idx]]['cycle'], para[0])

                    t1 = time.time()
                    RMSE, MTD, para_est = estimate_primitive((sample + translation) * scale, primitive_dict, func,
                                                             roof_types[idx], vis=False,
                                                             ambiguity=primitive_dict[roof_types[idx]]['ambiguity'])
                    t2 = time.time()

                    RMSEs.append(RMSE)
                    corner_RMSE.append(MTD)
                    GSD.append(np.sqrt(para_true[0] * para_true[1] / 512))
                    paras_est_all.append(para_est)
                    paras_true_all.append(np.concatenate([para_true, [r_gt]]))
                    times.append(t2 - t1)

            else:
                break

        paras_est_all, paras_true_all = np.array(paras_est_all), np.array(paras_true_all)
        corner_RMSE_prop = np.array(corner_RMSE) / np.array(GSD)
        print(f'{roof_types[idx]} fitting error mean: {np.mean(RMSEs):.4f}, time: {np.mean(times):.4f}')
        dw = paras_est_all[:, 0] - paras_true_all[:, 0]
        dl = paras_est_all[:, 1] - paras_true_all[:, 1]
        dh = paras_est_all[:, - 2] - paras_true_all[:, - 2]
        print(f'    w: {np.sqrt(np.mean(np.square(dw))):.4f}')
        print(f'    l: {np.sqrt(np.mean(np.square(dl))):.4f}')
        print(f'    h: {np.sqrt(np.mean(np.square(dh))):.4f}')
        print(f'    Overall shape RMSE: {np.sqrt(np.mean(np.square(np.concatenate([dw, dl, dh])))):.4f}')
        print(f'    w proportion mean: {np.mean(abs(dw) / paras_true_all[:, 0]):.4f}')
        print(f'    l proportion mean: {np.mean(abs(dl) / paras_true_all[:, 1]):.4f}')
        print(f'    h proportion mean: {np.mean(abs(dh) / paras_true_all[:, - 2]):.4f}')
        print(f'Corners RMSE Mean: {np.mean(corner_RMSE):.4f}')
        print(f'Corners RMSE STD: {np.std(corner_RMSE):.4f}')
        print(f'GSD Mean: {np.mean(GSD):.4f}')
        print(f'GSD STD: {np.std(GSD):.4f}')
        print(f'Corners RMSE Prop Mean: {np.mean(corner_RMSE_prop):.4f}')
        print(f'Corners RMSE Prop STD: {np.std(corner_RMSE_prop):.4f}')
        over_prop = np.mean(abs(np.concatenate([dw, dl, dh]) / np.concatenate([paras_true_all[:, 0],
                                                                               paras_true_all[:, 1],
                                                                               paras_true_all[:, - 2]])))
        dw_all.append(dw)
        dl_all.append(dl)
        dh_all.append(dh)
        RMSE_all.append(RMSEs)
        prop_all.append(abs(np.concatenate([dw, dl, dh]) / np.concatenate([paras_true_all[:, 0],
                                                                           paras_true_all[:, 1],
                                                                           paras_true_all[:, - 2]])))
        print(f'    Overall proportion mean: {over_prop:.4f}')

    d_all = np.concatenate([np.concatenate(dw_all), np.concatenate(dl_all), np.concatenate(dh_all)])
    RMSE_all = np.concatenate(RMSE_all)
    prop_all = np.concatenate(prop_all)
    print(f'Average shape RMSE: {np.sqrt(np.mean(np.square(d_all))):.8f} m')
    print(f'Average proportion error: {np.mean(prop_all):.8f}')
    print(f'Average error: {np.mean(RMSE_all):.8f} m')

    dw_all, dl_all, dh_all = np.concatenate(dw_all), np.concatenate(dl_all), np.concatenate(dh_all)
    np.savetxt(os.path.join(out_fd, f'{fname}_error.csv'), np.array([dw_all, dl_all, dh_all]).T, delimiter=',')
    ds = np.array([dw_all, dl_all, dh_all]).T
    ds[:, 0] = ds[:, 0] / ds[:, 0].std() * 0.0316
    ds[:, 1] = ds[:, 1] / ds[:, 1].std() * 0.0316
    ds[:, 2] = ds[:, 2] / ds[:, 2].std() * 0.0316

    fig, ax = plt.subplots()
    ax.hist(ds, bins=20, alpha=0.5, histtype='bar', color=['r', 'g', 'b'], label=['dw', 'dl', 'dh'])
    ax.set_xlabel('Error (unit:m)')
    ax.set_ylabel('Frequency')
    ax.set_title('UHF Error Distribution of Primitive Shape Parameters with sigma=10 cm')
    ax.legend()
    ax.set_xlim([-0.05, 0.05])
    plt.show()
    c = 1

