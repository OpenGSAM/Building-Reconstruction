import pickle
import numpy as np
from tqdm import tqdm
from primitive import *
from data_utils import *
from data import get_rotation

np.bool = np.bool_

if __name__ == "__main__":
    sigma, RMSEs = 0.05, []
    data_fd = f'data/roof_primitive_sigma_{int(sigma * 100)}cm'
    print(f'Generating data with sigma: {sigma * 100} cm...')
    train_out_fp = os.path.join(data_fd, 'data_train.pkl')
    test_out_fp = os.path.join(data_fd, 'data_test.pkl')
    np.random.seed(42)

    with open('primitive_info.json') as f:
        primitive_dict = json.load(f)
        roof_types = list(primitive_dict.keys())

    if not os.path.exists(data_fd):
        os.mkdir(data_fd)

    data_num, pt_num, train_id, test_id = 100000, 512, 0, 0
    dataset_train, dataset_test = {}, {}
    cls_num = len(roof_types)

    for roof in roof_types:
        for i in tqdm(range(int(data_num / cls_num)), desc=f'Generating {roof}...'):
            if i >= int(data_num / cls_num) * 0.8:
                train_flag = False
            else:
                train_flag = True

            func, cls = locals()[primitive_dict[roof]['func_name']], primitive_dict[roof]['code']

            # Sample a roof
            w, l = np.random.uniform(5, 25, 2)
            h = np.random.uniform(3, 10)
            wr, lr = np.random.uniform(0.3, 0.9, 2)
            edge_prop = 0.05
            surface_prop = 1 - edge_prop

            noise = np.random.normal(0, sigma, int(pt_num * surface_prop))
            # noise = np.random.normal(0, sigma, pt_num)
            RMSE = np.sqrt(np.mean(noise ** 2))
            RMSEs.append(RMSE)
            one_hot = np.array([0 for j in range(cls_num)])
            one_hot[cls] = 1
            occlusion_prob = np.random.uniform(0, 1)

            if primitive_dict[roof]['ambiguity'] and w > l:
                w, l, wr, lr = l, w, lr, wr

            verts, faces, _, paras = func([w, l, wr, lr, h])
            bbx = bounding_box([w, l, h])

            rad = np.random.uniform(- np.pi / 2, + np.pi / 2) if primitive_dict[roof]['cycle'] == 'p' \
                else np.random.uniform(- np.pi, + np.pi)
            rot = get_rotation(primitive_dict[roof]['cycle'], rad, True)
            face_num = len(faces)

            # Sample points on the roof with occlusion
            if occlusion_prob < 0.1 and primitive_dict[roof]['miss']:
                keep_ids = np.random.choice([item for item in range(face_num)], face_num - 1, replace=False)
                faces = [faces[item] for item in keep_ids]
                occlusion = True
            else:
                occlusion = False
            faces = np.concatenate(faces)

            verts = pc_RT(verts, rad)
            bbx = pc_RT(bbx, rad)
            faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
            geom_mesh = pv.PolyData(pc_RT(verts, 0.0), faces_pl)
            edges = geom_mesh.extract_feature_edges()
            points = np.array(edges.points)
            lines = edges.lines.reshape(-1, 3)[:, 1:]

            # Sample points on edges
            sample_edge = []
            for line in lines:
                p1, p2 = points[line[0]], points[line[1]]
                dl = p2 - p1
                sample = p1 + dl * np.linspace(0, 1, 200)[:, np.newaxis]
                sample_edge.append(sample)
            sample_edge = np.concatenate(sample_edge)
            sample_edge = sample_edge[np.random.choice(sample_edge.shape[0], int(pt_num * edge_prop), replace=False)]

            geom = o3d.geometry.TriangleMesh()
            geom.vertices = o3d.utility.Vector3dVector(verts)
            geom.triangles = o3d.utility.Vector3iVector(faces)

            sample = geom.sample_points_uniformly(int(pt_num * surface_prop), True)
            normals, sample = np.asarray(sample.normals), np.asarray(sample.points)
            sample += np.array([normals[:, 0] * noise, normals[:, 1] * noise, normals[:, 2] * noise]).T
            sample = np.concatenate([sample, sample_edge], axis=0)

            # sample = geom.sample_points_uniformly(pt_num, True)
            # normals, sample = np.asarray(sample.normals), np.asarray(sample.points)
            # sample += np.array([normals[:, 0] * noise, normals[:, 1] * noise, normals[:, 2] * noise]).T

            pc = sample.copy()
            scale, t, sample = pc_scale(sample)
            bbx = bbx / scale - t
            ws = np.linalg.norm(bbx[1] - bbx[0])
            ls = np.linalg.norm(bbx[2] - bbx[1])
            hs = np.linalg.norm(bbx[3] - bbx[2])
            paras_scaled = paras.copy()
            paras_scaled[0] = ws
            paras_scaled[1] = ls
            paras_scaled[-1] = hs

            data_ = {'pc': pc,
                     'sample': sample,
                     'rotation': rot,
                     'scales': scale,
                     'primitives': roof,
                     'cls_one_hot': one_hot,
                     'translation': t,
                     'paras': paras,
                     'paras_scaled': paras_scaled,
                     'occlusion': occlusion,
                     'RMSE': RMSE}

            if train_flag:
                dataset_train[str(train_id)] = data_
                train_id += 1
            else:
                dataset_test[str(test_id)] = data_
                test_id += 1

            # if not train_flag:
            #     print(f'Rooftype: {roof}')
            #     pl = pv.Plotter(shape=(1, 2))
            #     pl.subplot(0, 0)
            #     faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
            #     geom_mesh = pv.PolyData(pc_RT(verts, 0.0), faces_pl)
            #     pl.add_mesh(geom_mesh, opacity=0.5, show_edges=False)
            #     edges = geom_mesh.extract_feature_edges(
            #         boundary_edges=True,
            #         non_manifold_edges=False,
            #         feature_angle=10,
            #         manifold_edges=False,
            #     )
            #     pl.add_mesh(edges, color='k', line_width=3)
            #
            #     pl.subplot(0, 1)
            #     pl.add_points(pv.PolyData(pc), render_points_as_spheres=True, point_size=7, color='green')
            #     pl.link_views()
            #     pl.show()
            #     c = 1

    print(f'Mean RMSE: {np.mean(RMSEs):.4f}, std RMSE: {np.std(RMSEs):.4f}')
    f_train = open(train_out_fp, 'wb')
    f_test = open(test_out_fp, 'wb')
    pickle.dump(dataset_train, f_train)
    pickle.dump(dataset_test, f_test)

