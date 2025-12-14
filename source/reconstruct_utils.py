import os
import time
import laspy
import shutil
import trimesh
import subprocess
import numpy as np
import open3d as o3d
import pyvista as pv
from pysdf import SDF
import pyransac3d as pysac
from data_utils import pc_RT
from sklearn.cluster import DBSCAN
from itertools import combinations
from skspatial.objects import Plane
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

os.environ['PATH'] += os.pathsep + 'C:\\Program Files\\CloudCompare'


def read_las(fp, shift=True):
    las = laspy.read(fp)
    xyz = las.xyz

    if shift:
        T = np.mean(xyz, axis=0)
        xyz = xyz - T
        return xyz, T
    else:
        return xyz


def ransac_o3d(pc, sp=15, eps=0.3):
    time_start = time.time()
    clouds = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    z = np.array([0, 0, 1])

    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=15,
        coplanarity_deg=75,
        outlier_ratio=0.8,
        min_plane_edge_length=0,
        min_num_points=sp,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))

    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 1])
        mesh.paint_uniform_color(obox.color)
        N = obox.R[:, 2]

        if abs(np.rad2deg(np.arccos(N @ z)) - 90) > 15:
            mesh = SDF(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            dist = abs(mesh(pc))
            in_ids = dist <= 5 * eps

            if sum(in_ids) > sp:
                clouds.append(pc[in_ids])
                pc = pc[~in_ids]

    # pl = pv.Plotter()
    # for cloud in clouds:
    #     c = np.random.uniform(0, 1, 3)
    #     pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c)
    # pl.show()
    time_end = time.time()
    print(f'  Region Growing, found: {len(clouds)}, elapse time: {time_end - time_start:.2f}s.')
    return clouds


def ransac_cc(pc, sp=30, mnd=20, eps=0.3, sample_res=-1):
    time_start = time.time()
    np.savetxt('pc.txt', pc, delimiter=' ')

    temp_fd = 'temp'
    if os.path.exists(temp_fd):
        shutil.rmtree(temp_fd)
    os.mkdir(temp_fd)

    if sample_res > 0:
        subprocess.call(
            f'cloudcompare.exe -SILENT -AUTO_SAVE off -O pc.txt -C_EXPORT_FMT ASC -RANSAC ENABLE_PRIMITIVE PLANE '
            f'BITMAP_EPSILON_ABSOLUTE {sample_res} EPSILON_ABSOLUTE {eps} SUPPORT_POINTS {sp} MAX_NORMAL_DEV {mnd} '
            f'OUT_CLOUD_DIR temp ')
    else:
        subprocess.call(
            f'cloudcompare.exe -SILENT -AUTO_SAVE off -O pc.txt -C_EXPORT_FMT ASC -RANSAC ENABLE_PRIMITIVE PLANE '
            f'EPSILON_ABSOLUTE {eps} SUPPORT_POINTS {sp} MAX_NORMAL_DEV {mnd} '
            f'OUT_CLOUD_DIR temp ')

    clouds = []
    for i, item in enumerate(os.listdir(temp_fd)):
        in_fp = os.path.join(temp_fd, f'{item}')
        cloud = np.loadtxt(in_fp)[:, 0:3]
        clouds.append(np.round(cloud, 4))
    time_end = time.time()
    print(f'  RANSAC_CC found: {len(clouds)} planes, elapse time: {time_end - time_start:.2f}s.')
    return clouds


def ransac_sac(pc, sp=50, eps=0.15):
    time_start = time.time()
    plane = pysac.Plane()
    clouds = []
    flag = True
    cloud = pc.copy()
    while flag:
        best_eq, best_inliers = plane.fit(cloud, minPoints=sp, thresh=eps)
        temp_cloud = cloud[best_inliers]
        cloud = cloud[list(set(range(len(cloud))) - set(best_inliers))]

        clusters = DBSCAN(eps=10 * eps, min_samples=sp).fit(temp_cloud)
        for lbl in np.unique(clusters.labels_):
            if lbl != -1:
                clouds.append(temp_cloud[clusters.labels_ == lbl])

        if len(best_inliers) > sp:
            flag = True
        else:
            flag = False
    time_end = time.time()
    print(f'  RANSAC_SAC found: {len(clouds)} planes, elapse time: {time_end - time_start:.2f}s.')
    return clouds


def get_thetas(theta, lod=2):
    thetas = [theta, theta + 90, theta + 180, theta + 270]
    for i in range(1, lod):
        sep = 90 / (i + 1)
        for j in range(i):
            thetas.append(theta + sep * (j + 1))
            thetas.append(theta + sep * (j + 1) + 90)
            thetas.append(theta + sep * (j + 1) + 180)
            thetas.append(theta + sep * (j + 1) + 270)
    return np.deg2rad(thetas)


def get_angles(vec_1, vec_2):
    angle_in_rad = np.arctan2(np.cross(vec_1, vec_2), np.dot(vec_1, vec_2))
    return np.degrees(angle_in_rad)


def simplify_by_angle(poly_in, deg_tol=5):
    ext_poly_coords = poly_in.exterior.coords[:]
    vector_rep = np.diff(ext_poly_coords, axis=0)
    num_vectors = len(vector_rep)
    angles_list = []
    for i in range(0, num_vectors):
        angles_list.append(np.abs(get_angles(vector_rep[i], vector_rep[(i + 1) % num_vectors])))

    thresh_vals_by_deg = np.where(abs(np.array(angles_list) - 180) > deg_tol)
    new_idx = list(thresh_vals_by_deg[0] + 1)
    if len(new_idx) < 3:
        return False, None
    else:
        rst = Polygon([ext_poly_coords[idx] for idx in new_idx])
        return True, rst


def vis_primitive(cls, verts, faces, sample, score):
    faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
    pl = pv.Plotter()
    pl.set_background('grey')
    pl.add_text(f'Prediction, Primitive type: {cls}, Score: {score:.4f}')
    pl.add_mesh(pv.PolyData(verts, faces_pl), color='red', opacity=0.5, show_edges=True)
    pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=5, color='blue')
    pl.show()


def vis_primitives(verts, faces, planes, sample):
    pl = pv.Plotter()
    pl.set_background('grey')
    pl.add_points(pv.PolyData(sample), render_points_as_spheres=True, point_size=2, color='blue')

    for idx in range(len(planes)):
        for i in range(len(planes[idx]) - 1):
            pl.add_mesh(pv.Line(planes[idx][i], planes[idx][i + 1]), color='yellow', line_width=2)

    for idx in range(len(verts)):
        faces_pl = [[3, face[0], face[1], face[2]] for face in faces[idx]]
        pl.set_background('grey')
        pl.add_mesh(pv.PolyData(verts[idx], faces_pl), color='red', opacity=0.5, show_edges=True)
    pl.show()


def vis_result(clouds, planes):
    pl = pv.Plotter()

    for i in range(len(clouds)):
        plane = planes[i]
        pl.add_points(pv.PolyData(clouds[i]), render_points_as_spheres=True, point_size=4, color='blue')
        n = plane.shape[0] - 1
        face = [n + 1] + list(range(n)) + [0]
        polygon = pv.PolyData(plane[:-1], faces=face)
        pl.add_mesh(polygon, color='green', opacity=0.5, show_edges=True)
    pl.show()


def facade_remove(planes, th=10):
    out_planes, out_paras = [], []
    pca = PCA(n_components=3)

    for plane in planes:
        center = plane.mean(axis=0)
        pca.fit(plane - center)
        normal = pca.components_[-1]
        if normal[-1] < 0:
            normal = -normal

        # if abs(np.rad2deg(np.arccos(normal @ [0, 0, 1]))) > th:
        #     normal = [0, 0, 1]

        para = Plane(center, normal=normal)
        # para = Plane.best_fit(plane)
        if abs(np.rad2deg(np.arccos(para.normal @ [0, 0, 1])) - 90) > th:
            out_paras.append(para)
            out_planes.append(plane)

    return out_planes, out_paras


def get_connectivity(planes, threshold):
    plane_num = len(planes)
    connectivity = np.zeros((plane_num, plane_num))

    for i in range(plane_num):
        for j in range(i + 1, plane_num):
            try:
                dist_3D = cdist(planes[i], planes[j])
                dist_2D = cdist(planes[i][:, 0:2], planes[j][:, 0:2])
            except:
                planei = planes[i][np.random.randint(len(planes[i]), size=5000)]
                planej = planes[j][np.random.randint(len(planes[j]), size=5000)]
                dist_3D = cdist(planei, planej)
                dist_2D = cdist(planei[:, 0:2], planej[:, 0:2])

            if dist_2D.min() <= threshold:
                if dist_3D.min() <= threshold:
                    connectivity[i, j] = 1
                else:
                    connectivity[i, j] = 2

    connectivity += connectivity.T
    return connectivity


def create_candidate(planes, planes_sample, threshold):
    time_start = time.time()
    plane_num, proposals, candidates = len(planes), [], []
    connectivity = get_connectivity(planes, threshold)

    for pid in range(plane_num):
        nids = np.where(connectivity[pid] == 1)[0]
        comb = list(combinations(nids, 3)) + list(combinations(nids, 2))
        plane_comb = [list(item) + [pid] for item in comb] + [[item, pid] for item in nids]
        for item in plane_comb:
            if set(item) not in proposals:
                proposals.append(set(item))

    for item in proposals:
        cloud = np.concatenate([planes_sample[idx] for idx in item])
        candidates.append(cloud)

    time_end = time.time()
    print(f'    Remaining planes: {plane_num}, elapse time: {time_end - time_start:.2f}s.')
    return candidates, proposals


def get_DTM(verts, faces, pc):
    geom = trimesh.Trimesh(vertices=verts, faces=faces)
    dist = trimesh.proximity.signed_distance(geom, pc)
    return np.sqrt(np.mean(np.square(dist)))


def get_MTD(verts, pc):
    return np.sqrt(np.mean(np.square(cdist(verts, pc).min(axis=1))))


def get_reconstruction_score(verts, faces, pc):
    return get_DTM(verts, faces, pc), get_MTD(verts, pc)


def get_outliers(planes, verts, faces, proposal_ids, threshold):
    pids = []
    for idx, plane in enumerate(planes):
        if idx in proposal_ids:
            pids.append(idx)
        else:
            DTM = get_DTM(verts, faces, plane)
            if DTM <= 3 * threshold:
                pids.append(idx)
    return [item for item in range(len(planes)) if item not in pids]


def obj(x, args):
    verts, faces, sample = args
    verts = pc_RT(verts, x[0])
    mesh = SDF(verts, faces)
    return np.mean(np.square(mesh(sample)))


def smaple_planes(planes, num_points=512):
    plane_samples = []
    plane_nums = [item.shape[0] for item in planes]
    min_num = np.min(plane_nums)
    sample_nums = [int(item / min_num * num_points) for item in plane_nums]

    for i, sample_num in enumerate(sample_nums):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(planes[i])
        pcd.estimate_normals()
        pc_normals = np.asarray(pcd.normals)
        pc_normals[pc_normals[:, -1] < 0] = - pc_normals[pc_normals[:, -1] < 0]
        pcd.normals = o3d.utility.Vector3dVector(pc_normals)
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        radii = [3 * avg_dist, 6 * avg_dist, 9 * avg_dist]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        pcd_sample = mesh.sample_points_uniformly(number_of_points=sample_num)
        plane_samples.append(np.asarray(pcd_sample.points))

        # ashape = alphashape.alphashape(planes[i][:, 0:2], alpha=0.2)
        # if ashape.geom_type != 'MultiPolygon':
        #     sample_2d = np.insert(pointpats.random.poisson(ashape, size=sample_num), 2, 0, axis=1)
        #     sample_3d = np.array([plane_paras[i].intersect_line(Line(point=item, direction=[0, 0, 1]))
        #                           for item in sample_2d])
        #     plane_samples.append(sample_3d)
    return plane_samples


def check_multipolygon(poly, type_return='convex'):
    if poly.geom_type == 'MultiPolygon':
        geoms = [geom for geom in poly.geoms]
        areas = [geom.area for geom in poly.geoms]
        if type_return == 'max':
            out_geom = [geoms[np.argmax(areas)]]
        elif type_return == 'all':
            out_geom = geoms
        else:
            out_geom = [poly.convex_hull]
    elif poly.geom_type == 'Polygon':
        out_geom = [poly]
    else:
        out_geom = [poly.convex_hull]
    return out_geom


def get_vert_z(xy, ps, n):
    z = ps[-1] - (n[0] * (xy[0] - ps[0]) + n[1] * (xy[1] - ps[1])) / n[-1]
    return [xy[0], xy[1], z]


def get_vert_3D(xy, xyz):
    dist = cdist([xy], xyz[:, 0:2])
    idx = np.argmin(dist)
    return xyz[idx]

