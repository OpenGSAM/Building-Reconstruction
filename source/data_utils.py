import os
import laspy
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial.transform import Rotation as R
os.environ['PATH'] += os.pathsep + 'C:\\Program Files\\CloudCompareStereo'


def create_polygon_pv(polygon):
    faces = np.array([len(polygon) - 1] + [item for item in range(len(polygon) - 1)])
    surf = pv.PolyData(polygon, faces)
    return surf


def extrude_polygon_pv(surf, h):
    plane = pv.Plane(center=(surf.center[0], surf.center[1], h),
                     direction=(0, 0, -1), i_size=30, j_size=30)
    surf_extruded = surf.extrude_trim((0, 0, -1.0), plane)
    return surf_extruded


def create_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    return mesh


def translation(xyz):
    T = np.mean(xyz, axis=0)
    xyz = xyz - T
    scale = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
    return xyz, T, scale


def read_las(fp, shift=True):
    las = laspy.read(fp)
    xyz = las.xyz

    if shift:
        T = np.mean(xyz, axis=0)
        xyz = xyz - T
        return xyz, T
    else:
        return xyz


def pc_sampling(pc_raw, sample_num=1024, scale=False):
    # pc_num = pc_raw.shape[0]
    np.random.seed(42)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_raw)
    pcd.estimate_normals()
    pc_normals = np.asarray(pcd.normals)
    pc_normals[pc_normals[:, -1] < 0] = - pc_normals[pc_normals[:, -1] < 0]
    pcd.normals = o3d.utility.Vector3dVector(pc_normals)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [2 * avg_dist, 4 * avg_dist]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, mesh])
    pcd_sample = mesh.sample_points_uniformly(number_of_points=sample_num)
    sample = np.asarray(pcd_sample.points)
    # sample = np.concatenate((pc_raw, sample))
    # sample = sample[np.random.choice(sample.shape[0], sample_num, replace=False)]

    if scale:
        scale, ts, sample = pc_scale(sample)
        return scale, ts, sample
    else:
        return None, sample


def pc_RT(pc, r, t=np.array([0, 0, 0])):
    rm = R.from_euler('z', r).as_matrix()
    return (rm @ pc.T).T + t


def pc_scale_3d(pc):
    scales = pc.max(axis=0) - pc.min(axis=0)
    pc = pc / (pc.max(axis=0) - pc.min(axis=0))
    T = (pc.max(axis=0) + pc.min(axis=0)) / 2
    pc = pc - T
    return scales, T, pc


def pc_scale(pc):
    scales = pc.max(axis=0) - pc.min(axis=0)
    scale = np.max(scales)
    pc = pc / scale
    T = (pc.max(axis=0) + pc.min(axis=0)) / 2
    pc = pc - T
    return scale, T, pc


def pc_scale_inv(pc, scales, T):
    pc = pc + T
    pc = pc * scales
    return pc


def pc_augment_RT(pc, nd=8):
    degrees = [360 / nd * i for i in range(nd)]
    Rs = [R.from_euler('z', np.deg2rad(item)).as_matrix() for item in degrees]
    return degrees, [pc_RT(pc, item) for item in Rs]

