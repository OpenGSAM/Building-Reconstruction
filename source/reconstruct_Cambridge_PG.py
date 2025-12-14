import copy
import torch
import argparse
import alphashape
import pandas as pd
from model import Pct
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
from primitive import *
from data import Clouds
from osgeo import gdal, osr
from reconstruct_utils import *
import torch.nn.functional as F
from bldg_regularization import *
from sklearn.cluster import DBSCAN
from skspatial.objects import Plane
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
from torch.utils.data import DataLoader
from trimesh.creation import triangulate_polygon
from scipy.optimize import differential_evolution

np.bool = np.bool_


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

# RD_df = pd.read_csv('reconstruction_scores.csv')
# RD_threshold = {}
# for col in RD_df.columns:
#     RD_threshold[col] = RD_df[col].mean() + 5 * RD_df[col].std()

with open('primitive_info.json') as f:
    primitive_dict = json.load(f)
    roof_types = list(primitive_dict.keys())
    dim_num = [primitive_dict[roof_type]['para_num'] + 1 for roof_type in roof_types]

device = torch.device("cuda")
model_cls_fp = f'checkpoints/roof_primitive/models/model_cls.t7'
model_cls = nn.DataParallel(Pct(args, len(dim_num)).to(device))
model_cls.load_state_dict(torch.load(model_cls_fp))
model_cls.eval()

model_reg = []
for idx, dim in enumerate(dim_num):
    model_reg_fp = f'checkpoints/roof_primitive/models/model_reg_{roof_types[idx]}.t7'
    model = nn.DataParallel(Pct(args, dim).to(device))
    model.load_state_dict(torch.load(model_reg_fp))
    model.eval()
    model_reg.append(model)


def estimate_primitives(candidates, model_cls, model_reg, thetas):
    loader = DataLoader(Clouds(candidates), batch_size=16, shuffle=False, drop_last=False)
    samples_all, class_all = [], []
    scales_all, translations_all, paras_all = [], [], []
    verts_all, faces_all, bbx_all = [], [], []
    idx, scores = 0, []
    sample_src_all, rads_all = [], []
    verts_src_all, bbx_src_all = [], []
    coef = 1.05

    for _, data in enumerate(tqdm(loader, desc='    Estimating primitives...')):
        item, scales, translations = data
        item = item.to(device)

        clouds = item.detach().cpu().numpy()
        logits_cls = model_cls(item.permute(0, 2, 1))
        pred_probs = F.softmax(logits_cls, dim=1)
        classes = pred_probs.max(dim=1)[1].detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        translations = translations.detach().cpu().numpy()

        for i in range(len(clouds)):
            cls, scale, translation = classes[i], scales[i], translations[i]
            cls = 1 if cls == 5 else cls
            class_all.append(roof_types[cls])
            scales_all.append(scale)
            translations_all.append(translation)
            logits = model_reg[cls](item[None, i].permute(0, 2, 1)).detach().cpu().numpy()[0]
            # rad = get_rotation(primitive_dict[roof_types[cls]]['cycle'], logits[0])

            if primitive_dict[roof_types[cls]]['para_num'] == 3:
                para = [logits[1] * coef, logits[2] * coef, 1, 1, logits[3] * coef]
                bbx = bounding_box([logits[1] * coef, logits[2] * coef, logits[3] * coef])
            elif primitive_dict[roof_types[cls]]['para_num'] == 4:
                para = [logits[1] * coef, logits[2] * coef, 1, logits[3], logits[4] * coef]
                bbx = bounding_box([logits[1] * coef, logits[2] * coef, logits[4] * coef])
            else:
                para = [logits[1] * coef, logits[2] * coef, logits[3], logits[4], logits[5] * coef]
                bbx = bounding_box([logits[1] * coef, logits[2] * coef, logits[5] * coef])

            func = globals()[primitive_dict[roof_types[cls]]['func_name']]
            verts_src, faces, _ = func(para, True, False)

            scores_temp = []
            for theta in thetas:
                verts = pc_RT(verts_src, theta)
                DTM, MTD = get_reconstruction_score(verts, faces, clouds[i])
                scores_temp.append(DTM)
            rad = thetas[np.argmin(scores_temp)]

            sample_src_all.append(clouds[i])
            verts_src_all.append(verts_src)
            bbx_src_all.append(bbx)
            rads_all.append(rad)
            verts, bbx = pc_RT(verts_src, rad), pc_RT(bbx, rad)

            DTM = get_DTM(verts, faces, clouds[i])
            scores.append(DTM)
            verts_all.append((verts + translation) * scale)
            samples_all.append((clouds[i] + translation) * scale)
            bbx_all.append((bbx + translation) * scale)
            faces_all.append(faces)
            paras_all.append(para)

            idx += 1
    bid = np.argmin(scores)

    # for bid in np.argsort(scores):
    #     rst = differential_evolution(obj, [(- 5 * np.pi, 5 * np.pi)], x0=rads_all[bid],
    #                                  args=([verts_src_all[bid], faces_all[bid], sample_src_all[bid]],))
    #     DTM, MTD = get_reconstruction_score(pc_RT(verts_src_all[bid], rst.x[0]), faces_all[bid], sample_src_all[bid])
    #     verts = (pc_RT(verts_src_all[bid], rst.x[0]) + translations_all[bid]) * scales_all[bid]
    #     vis_primitive(class_all[bid], verts, faces_all[bid], samples_all[bid], DTM)

    rst = differential_evolution(obj, [(- 5 * np.pi, 5 * np.pi)], x0=rads_all[bid],
                                 args=([verts_src_all[bid], faces_all[bid], sample_src_all[bid]],))
    DTM, MTD = get_reconstruction_score(pc_RT(verts_src_all[bid], rst.x[0]), faces_all[bid], sample_src_all[bid])
    verts = (pc_RT(verts_src_all[bid], rst.x[0]) + translations_all[bid]) * scales_all[bid]
    bbx = (pc_RT(bbx_src_all[bid], rst.x[0]) + translations_all[bid]) * scales_all[bid]

    return (bid, DTM, MTD, verts, faces_all[bid], bbx, class_all[bid],
            samples_all[bid], paras_all[bid], scales_all[bid], translations_all[bid])


def reconstruct_planes(planes, planes_para_all, dem_plane, theta, APD=0.6, simple_th=0.5, alpha=1.0):
    DTM, pv_meshes, tri_meshes = [], [], []
    G = get_connectivity(planes, APD * 2)
    G[G == 2] = 1
    G_nx = nx.from_numpy_array(G)
    planes_cluster = [list(G_nx.subgraph(item).nodes()) for item in nx.connected_components(G_nx)]

    for nodes in tqdm(planes_cluster, desc=f'    Regularizing {len(planes)} planes...'):
        cloud = np.concatenate([planes[i] for i in nodes])[:, 0:2]
        boundary = regularize(check_multipolygon(alphashape.alphashape(cloud, alpha).simplify(simple_th)),
                              False, 2, theta)[0]
        ashapes, planes_para, planes_ids = [], [], []

        for idx in nodes:
            DTM.append(np.sqrt(np.mean(np.square((planes[idx] - planes_para_all[idx].point) @
                                                 planes_para_all[idx].normal))))
            shapes = check_multipolygon(alphashape.alphashape(planes[idx][:, 0:2], alpha), type_return='all')
            for shape in shapes:
                ashapes.append(shape.simplify(simple_th).buffer(simple_th / 2, join_style='mitre'))
                planes_para.append(planes_para_all[idx])
                planes_ids.append(idx)
        planes_reg = [check_multipolygon(boundary.intersection(poly), 'max')[0]
                      for poly in regularize(ashapes, False, 2, theta)]
        ids = [item for item in range(len(planes_reg))]

        for i in ids:
            for j in [item for item in ids if item != i]:
                if planes_reg[i].intersects(planes_reg[j]):
                    cloud_i, cloud_j = planes[planes_ids[i]], planes[planes_ids[j]]
                    dists = cdist(cloud_j, cloud_i)
                    height_i = np.mean(cloud_i[dists.min(axis=0).argsort()[0:10], 2])
                    height_j = np.mean(cloud_j[dists.min(axis=1).argsort()[0:10], 2])

                    if height_i > height_j:
                        diff = check_multipolygon(planes_reg[j].difference(planes_reg[i]), type_return='max')[0]
                        planes_reg[j] = diff
                    else:
                        diff = check_multipolygon(planes_reg[i].difference(planes_reg[j]), type_return='max')[0]
                        planes_reg[i] = diff

        planes_para = [planes_para[i] for i in ids if not planes_reg[i].is_empty and
                       planes_reg[i].geom_type != 'LineString']
        planes_reg = [planes_reg[i] for i in ids if not planes_reg[i].is_empty and
                      planes_reg[i].geom_type != 'LineString']

        remaining = boundary.difference(cascaded_union(planes_reg))
        if not remaining.is_empty:
            remaining_temp = check_multipolygon(remaining, type_return='all')
            remaining = []
            for poly in remaining_temp:
                flag, rst = simplify_by_angle(poly)
                if flag and 1 <= rst.area <= 100:
                    remaining.append(rst)

            remaining_para = []
            for poly in remaining:
                dists = [item.exterior.distance(poly.centroid) for item in planes_reg]
                remaining_para.append(planes_para[np.argmin(dists)])

            planes_reg = planes_reg + remaining
            planes_para = planes_para + remaining_para

        verts_all, faces_all = [], []
        for i in range(len(planes_reg)):
            vertices, faces = triangulate_polygon(planes_reg[i])
            vertices = np.array([get_vert_z(p, planes_para[i].point, planes_para[i].normal) for p in vertices])

            if len(faces) == 0:
                faces = [[3, 0, 1, 2]]
            else:
                faces = np.insert(faces, 0, 3, axis=1)

            pids = i * np.ones(len(vertices))
            if len(verts_all) == 0:
                verts_all = np.insert(vertices, 3, pids, axis=1)
            else:
                verts_all = np.concatenate((verts_all, np.insert(vertices, 3, pids, axis=1)))
            faces_all.append(faces)

            # roof_mesh = pv.PolyData(vertices, faces)
            # mesh = roof_mesh.extrude_trim((0, 0, -1.0), dem_plane).clean().triangulate()
            # tri_mesh = trimesh.Trimesh(mesh.points, mesh.faces.reshape((mesh.n_cells, 4))[:, 1:])
            # tri_mesh.fix_normals()
            # if tri_mesh.is_volume:
            #     tri_meshes.append(tri_mesh)
            # else:
            #     pv_meshes.append(mesh)

        clusters = DBSCAN(eps=1, min_samples=2).fit(verts_all[:, 0:3])
        for lbl in np.unique(clusters.labels_):
            if lbl != -1:
                verts_all[lbl == clusters.labels_, 2] = verts_all[lbl == clusters.labels_, 2].max()

        for i in range(len(planes_reg)):
            vertices = verts_all[verts_all[:, 3] == i][:, 0:3]
            faces = faces_all[i]

            roof_mesh = pv.PolyData(vertices, faces)
            mesh = roof_mesh.extrude_trim((0, 0, -1.0), dem_plane).clean().triangulate()
            tri_mesh = trimesh.Trimesh(mesh.points, mesh.faces.reshape((mesh.n_cells, 4))[:, 1:])
            tri_mesh.fix_normals()
            if tri_mesh.is_volume:
                tri_meshes.append(tri_mesh)
            else:
                pv_meshes.append(mesh)

    return np.mean(DTM), pv_meshes, tri_meshes


def get_bldg_xyzs(bldg_ext, dem, ct, gt_inv):
    bldg_ext_xyz = []
    for pt in bldg_ext:
        mapx, mapy, z = ct.TransformPoint(pt[0], pt[1])
        pixel_x, pixel_y = gdal.ApplyGeoTransform(gt_inv, mapx, mapy)
        pixel_x = round(pixel_x)
        pixel_y = round(pixel_y)
        pixel_x = max(min(pixel_x, width - 1), 0)
        pixel_y = max(min(pixel_y, height - 1), 0)
        bldg_ext_xyz.append([pt[0], pt[1], dem[pixel_y, pixel_x]])
    return np.array(bldg_ext_xyz)


def get_domain_direction(clouds, simple_th=0.5, alpha=1.0):
    clouds = np.concatenate(clouds)
    ashape = check_multipolygon(alphashape.alphashape(clouds[:, 0:2], alpha).simplify(simple_th))
    border_reg, direction = regularize(ashape)
    IOU = ashape[0].intersection(border_reg[0]).area / ashape[0].union(border_reg[0]).area
    return border_reg[0], direction, IOU


if __name__ == "__main__":
    np.random.seed(42)
    data_name = 'Cambridge_PG'
    pc_fd = f'data/{data_name}/clouds'
    out_fd = f'data/{data_name}/reconstruction'
    out_plane_fd = f'data/{data_name}/reconstruction/planes'
    out_prim_fd = f'data/{data_name}/reconstruction/primitives'
    out_mesh_fd = f'data/{data_name}/reconstruction/meshes'
    dem_fp = f'data/{data_name}/dem.tif'
    log_fp = os.path.join(out_fd, f'log.txt')

    if not os.path.exists(out_fd):
        os.mkdir(out_fd)
    if not os.path.exists(out_plane_fd):
        os.mkdir(out_plane_fd)
    if not os.path.exists(out_prim_fd):
        os.mkdir(out_prim_fd)
        for prim in roof_types:
            os.mkdir(os.path.join(out_prim_fd, prim))
    if not os.path.exists(out_mesh_fd):
        os.mkdir(out_mesh_fd)
        log_file = open(log_fp, 'w')
    else:
        log_file = open(log_fp, 'a')

    pc_fps = [os.path.join(pc_fd, item) for item in os.listdir(pc_fd) if 'lasx' not in item]
    # Loading DEM
    print('Loading DEM...')
    ds = gdal.Open(dem_fp)
    gt = ds.GetGeoTransform()
    dem = np.array(ds.ReadAsArray())
    width = ds.RasterXSize
    height = ds.RasterYSize
    point_srs = osr.SpatialReference()
    point_srs.ImportFromWkt(ds.GetProjection())
    point_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(ds.GetProjection())
    ct = osr.CoordinateTransformation(point_srs, dem_srs)
    gt_inv = gdal.InvGeoTransform(gt)

    # Point cloud features
    APD, lod = 0.5, 2

    for fp in pc_fps:
        fname = os.path.basename(fp).split('.')[0]
        out_plane_fp = os.path.join(out_plane_fd, f'{fname}.obj')
        out_mesh_fp = os.path.join(out_mesh_fd, f'{fname}.obj')

        if os.path.exists(out_mesh_fp):
            print(f'{fname} exist, skip')
        else:
            print(f'Working on {fname}')

            # Read building bottom height
            pc_raw, t = read_las(fp)
            bldg_ext = np.array(MultiPoint(pc_raw[:, 0:2]).convex_hull.exterior.coords[:]) + t[0:2]
            bldg_ext_xyz = get_bldg_xyzs(bldg_ext, dem, ct, gt_inv) - t
            plane_coef = Plane.best_fit(bldg_ext_xyz)

            smin, smax = pc_raw.min(axis=0), pc_raw.max(axis=0)
            dem_plane = pv.Plane(center=tuple(plane_coef.point), direction=tuple(plane_coef.normal),
                                 i_size=int(smax[0] - smin[0] + 500), j_size=int(smax[1] - smin[1] + 500))

            pc_height = plane_coef.distance_points(pc_raw)
            pc_raw = pc_raw[pc_height >= 2]
            # RANSAC plane detection and facade removal
            planes, paras = facade_remove(ransac_cc(pc_raw, sp=256, mnd=20, eps=0.15), th=15)

            # pl = pv.Plotter()
            # for cloud in planes:
            #     c = np.random.uniform(0, 1, 3)
            #     pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c, opacity=0.6)
            # for para in paras:
            #     arrow = pv.Arrow(start=para.point, direction=para.normal, tip_length=1, tip_radius=0.5)
            #     pl.add_mesh(arrow, color='red')
            # pl.show()
            if len(planes) == 0:
                print('  CloudCompare RANSAC Failed, using pyransac3D RANSAC...')
                planes, paras = facade_remove(ransac_sac(pc_raw, sp=256, eps=0.15), th=15)

            if len(planes) == 0:
                print('  No planes found, use primitive for fitting.')
                try:
                    # Domain direction computation
                    _, theta, IOU = get_domain_direction([pc_raw], simple_th=APD, alpha=0.2)
                    thetas = get_thetas(theta, lod)

                    rst = estimate_primitives(smaple_planes([pc_raw]), model_cls, model_reg, thetas)
                    bid, DTM, MTD, verts, faces, bbx, cls, sample, para, translation, scale = rst

                    out_prim_fp = os.path.join(out_prim_fd, cls, f'{fname}_{0}.obj')
                    geom_DTM = get_DTM(verts, faces, sample)
                    data_poly = MultiPoint(sample[:, 0:2]).convex_hull
                    model_poly = MultiPoint(verts[:, 0:2]).convex_hull
                    geom_IOU = data_poly.intersection(model_poly).area / data_poly.union(model_poly).area
                    log_file.write(f'{fname},{cls}_{0},{geom_DTM},{geom_IOU}\n')
                    print(f'    found a {cls}, DTM: {DTM:.8f}, MTD: {MTD:.8f}, IOU: {geom_IOU:.8f}')

                    primitive_geom = (pv.PolyData(verts, np.insert(faces, 0, 3, axis=1)).
                                      extrude_trim((0, 0, -1.0), dem_plane))
                    primitive_geom.translate(t).save(out_prim_fp)
                    primitive_geom.translate(t).save(out_mesh_fp)
                except:
                    continue
            else:
                # Domain direction computation
                _, theta, IOU = get_domain_direction(planes)
                thetas = get_thetas(theta, lod)
                # Plane resampling
                planes_src = copy.deepcopy(planes)
                planes_sample = smaple_planes(planes)
                primitive_geoms = []

                iter_num, iter_flag = 0, True
                while iter_flag:
                    print(f'  Iteration {iter_num}:')
                    candidates, proposals = create_candidate(planes, planes_sample, 2 * APD)

                    if len(candidates) == 0:
                        iter_flag = False
                    else:
                        rst = estimate_primitives(candidates, model_cls, model_reg, thetas)
                        bid, DTM, MTD, verts, faces, bbx, cls, sample, para, translation, scale = rst

                        # if score < RD_threshold[cls]:
                        if DTM < 0.015 and MTD < 0.08:
                            out_prim_fp = os.path.join(out_prim_fd, cls, f'{fname}_{iter_num}.obj')
                            geom_DTM = get_DTM(verts, faces, sample)
                            data_poly = MultiPoint(sample[:, 0:2]).convex_hull
                            model_poly = MultiPoint(verts[:, 0:2]).convex_hull
                            geom_IOU = data_poly.intersection(model_poly).area / data_poly.union(model_poly).area
                            log_file.write(f'{fname},{cls}_{iter_num},{geom_DTM},{geom_IOU}\n')
                            print(f'    found a {cls}, DTM: {DTM:.8f}, MTD: {MTD:.8f}, IOU: {geom_IOU:.8f}')

                            # Outliers removal
                            npids = get_outliers(planes, verts, faces, proposals[bid], 1.25 * geom_DTM)
                            planes = [planes[i] for i in npids]
                            planes_src = [planes_src[i] for i in npids]
                            planes_sample = [planes_sample[i] for i in npids]
                            paras = [paras[i] for i in npids]

                            primitive_geom = (pv.PolyData(verts, np.insert(faces, 0, 3, axis=1)).
                                              extrude_trim((0, 0, -1.0), dem_plane).clean().triangulate())
                            primitive_geom.translate(t).save(out_prim_fp)
                            primitive_geoms.append(primitive_geom)
                        else:
                            print(f'    poor {cls}, DTM: {DTM:.8f}, MTD: {MTD:.8f}')
                            iter_flag = False
                    iter_num += 1

                    if not iter_flag:
                        if len(planes_src) > 0:
                            plane_DTM, pv_meshes, tri_meshes = reconstruct_planes(planes_src, paras, dem_plane, theta,
                                                                                  APD=APD, simple_th=APD, alpha=1)

                            log_file.write(f'{fname},plane,{plane_DTM},{IOU}\n')
                            pl = pv.Plotter()

                            for prim in primitive_geoms:
                                tri_mesh = trimesh.Trimesh(prim.points,
                                                           prim.faces.reshape((prim.n_cells, 4))[:, 1:])
                                tri_mesh.fix_normals()
                                if tri_mesh.is_volume:
                                    tri_meshes.append(tri_mesh)
                            if len(tri_meshes) > 0:
                                tri_meshes = pv.wrap(trimesh.boolean.union(tri_meshes))
                                pl.add_mesh(tri_meshes.translate(t), show_edges=True)
                            for mesh in pv_meshes:
                                pl.add_mesh(mesh.translate(t), show_edges=True)
                            pl.export_obj(out_plane_fp)
                            pl.export_obj(out_mesh_fp)
                            pl.close()
                        else:
                            if len(primitive_geoms) > 0:
                                all_meshes = []
                                pl = pv.Plotter()
                                for prim in primitive_geoms:
                                    tri_mesh = trimesh.Trimesh(prim.points,
                                                               prim.faces.reshape((prim.n_cells, 4))[:, 1:])
                                    tri_mesh.fix_normals()
                                    if tri_mesh.is_volume:
                                        all_meshes.append(tri_mesh)
                                    else:
                                        pl.add_mesh(prim.translate(t), show_edges=True)
                                try:
                                    tri_meshes = pv.wrap(trimesh.boolean.union(all_meshes))
                                    pl.add_mesh(tri_meshes.translate(t), show_edges=True)
                                except:
                                    continue
                                pl.export_obj(out_mesh_fp)
                                pl.close()
    log_file.close()
