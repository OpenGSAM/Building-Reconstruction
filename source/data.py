import os
import pickle
from primitive import *
from data_utils import pc_scale
from torch.utils.data import Dataset

np.bool = np.bool_


def get_rotation(cycle, value, inv=False):
    if cycle == 'p':
        rad = value * np.pi - np.pi / 2
        rot = (value + np.pi / 2) / np.pi
    else:
        rad = value * 2 * np.pi - np.pi
        rot = (value + np.pi) / (np.pi * 2)

    if inv:
        return rot
    else:
        return rad


class Roof_Primitive_cls(Dataset):
    def __init__(self, data_fd, partition='train'):
        fp = os.path.join(data_fd, f'data_{partition}.pkl')
        file = open(fp, 'rb')
        dataset = pickle.load(file)
        self.data_list = []
        self.data_size = len(dataset.keys())
        print(f'{partition} dataset size: {len(dataset.keys())}')

        for item in dataset.keys():
            data_ = dataset[str(item)]
            self.data_list.append([np.float32(data_['sample']),
                                   np.float32(data_['cls_one_hot'])])

    def __getitem__(self, item):
        return self.data_list[item][0], self.data_list[item][1]

    def __len__(self):
        return self.data_size


class Roof_Primitive_reg(Dataset):
    def __init__(self, data_fd, partition='train', primitive='pyramid', primitive_info='pyramid'):
        fp = os.path.join(data_fd, f'data_{partition}.pkl')
        file = open(fp, 'rb')
        dataset = pickle.load(file)
        self.data_list = []

        for item in dataset.keys():
            data_ = dataset[str(item)]

            if primitive_info['para_num'] == 3:
                paras = [data_['paras_scaled'][0], data_['paras_scaled'][1], data_['paras_scaled'][-1]]
                paras_true = [data_['paras'][0], data_['paras'][1], data_['paras'][-1]]
            elif primitive_info['para_num'] == 4:
                paras = [data_['paras_scaled'][0], data_['paras_scaled'][1],
                         data_['paras_scaled'][3], data_['paras_scaled'][-1]]
                paras_true = [data_['paras'][0], data_['paras'][1], data_['paras'][3], data_['paras'][-1]]
            else:
                paras = [data_['paras_scaled'][0], data_['paras_scaled'][1],
                         data_['paras_scaled'][2], data_['paras_scaled'][3], data_['paras_scaled'][-1]]
                paras_true = [data_['paras'][0], data_['paras'][1], data_['paras'][2],
                              data_['paras'][3], data_['paras'][-1]]

            if data_['primitives'] == primitive:
                self.data_list.append([np.float32(data_['sample']),
                                       np.float32(data_['cls_one_hot']),
                                       np.float32(paras),
                                       np.float32(data_['scales']),
                                       np.float32(data_['translation']),
                                       np.float32(paras_true)])

        self.data_size = len(self.data_list)
        print(f'{primitive} primitive, {partition} dataset size: {self.data_size}')

    def __getitem__(self, item):
        return (self.data_list[item][0], self.data_list[item][1], self.data_list[item][2],
                self.data_list[item][3], self.data_list[item][4], self.data_list[item][5])

    def __len__(self):
        return self.data_size


class Clouds(Dataset):
    def __init__(self, clouds, num=512):
        self.data_list, self.scales, self.translations = [], [], []

        for cloud in clouds:
            cloud = cloud[np.random.choice(cloud.shape[0], num, replace=False)]
            cloud_scale, cloud_translation, cloud = pc_scale(cloud)
            self.data_list.append(np.float32(cloud))
            self.scales.append(cloud_scale)
            self.translations.append(cloud_translation)

        self.data_size = len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item], self.scales[item], self.translations[item]

    def __len__(self):
        return self.data_size


class ISPRS_Vaihingen_primitive(Dataset):
    def __init__(self, data_fd, partition='train'):
        # fp_ = os.path.join(data_fd, 'data_' + partition + '_cls.pkl')
        # file_out = open(fp_, 'wb')
        np.bool = np.bool_

        H = 10
        fp = os.path.join(data_fd, 'data_' + partition + '_cls.pkl')
        file = open(fp, 'rb')
        dataset = pickle.load(file)
        self.data_list = []
        self.types = {'pyramid': 0, 'gable': 1, 'hip': 2, 'mansard': 3, 'shed': 4}
        self.out_dim = np.array([10, 7])

        for item in dataset.keys():
            bldg = dataset[str(item)]
            bbxs = np.array([item['paras'] for item in bldg['bbx'].values()])

            if len(bbxs) < H:
                id = 0
                while bbxs.shape[0] < H:
                    bbxs = np.append(bbxs, np.array([[0.0 for i in range(7)]]), axis=0)
                    id += 1
            else:
                bbxs = bbxs[:H, :]

            self.data_list.append([np.float32(bldg['pc']), np.float32(bbxs.flatten())])

            # cloud = pv.PolyData(bldg['pc'])
            # cloud['point_color'] = bldg['cls'] # just use z coordinate
            #
            # pv.plot(cloud, scalars='point_color', cmap='jet', show_bounds=True, cpos='yz')
        #     primitives = bldg['primitives']
        #     bbxs = bldg['bbx']
        #     sample = bldg['pc']
        #     dists, geoms = [], [o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(sample))]
        #
        #     for pid in primitives.keys():
        #         primitive, bbx = primitives[str(pid)], bbxs[str(pid)]
        #         ps, bs = primitive['paras'], bbx['paras']
        #         func = globals()[primitive['primitive'] + '_geom']
        #         roof_vert, roof_faces = func(ps[:-4])
        #         roof_vert = pc_RT(roof_vert, ps[-4], ps[-3:])
        #         roof = SDF(roof_vert, roof_faces)
        #         wall_vert, wall_faces = box_geom(bs[0:3])
        #         wall_vert = pc_RT(wall_vert, bs[- 4], bs[-3:])
        #         wall = SDF(wall_vert, wall_faces)
        #         dist = np.array([abs(roof(sample)), abs(wall(sample))])
        #         dists.append(dist.T.min(axis=1))
        #
        #         roof_o3d = o3d.geometry.TriangleMesh()
        #         roof_o3d.vertices = o3d.utility.Vector3dVector(roof_vert)
        #         roof_o3d.triangles = o3d.utility.Vector3iVector(roof_faces)
        #         geoms.append(o3d.geometry.LineSet.create_from_triangle_mesh(roof_o3d))
        #
        #         wall_o3d = o3d.geometry.TriangleMesh()
        #         wall_o3d.vertices = o3d.utility.Vector3dVector(wall_vert)
        #         wall_o3d.triangles = o3d.utility.Vector3iVector(wall_faces)
        #         geoms.append(o3d.geometry.LineSet.create_from_triangle_mesh(roof_o3d))
        #
        #     dists_index = np.array(dists).T.argmin(axis=1)
        #     labels = np.array([self.types[primitives[str(pid)]['primitive']] for pid in dists_index])
        #     dataset[str(item)]['cls'] = labels
        #
        # pickle.dump(dataset, file_out)
        # c = 1

            # bbxs = np.array([item['paras'] for item in bldg['bbx'].values()])
            # bbx_num = len(bbxs)
            #
            # if bbx_num < H:
            #     id = 0
            #     while bbxs.shape[0] < H:
            #         bbxs = np.append(bbxs, [bbxs[id]], axis=0)
            #         id += 1
            # else:
            #     bbxs = bbxs[:H, :]
            #
            # data_ = [np.float32(sample), np.float32(bbxs.flatten())]
            # self.data_list.append(data_)

    def __getitem__(self, item):
        return self.data_list[item][0], self.data_list[item][1]

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    train = Roof_Primitive_cls('data/roof_primitive/', partition='train')
    test = Roof_Primitive_cls('data/roof_primitive/', partition='test')
    for data, gt in train:
        print(data.shape)
        c = 1
