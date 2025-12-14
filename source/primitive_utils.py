import numpy as np
import open3d as o3d
from pysdf import SDF
import pyvista as pv
from primitive_estimate_utils import *
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from shapely.geometry import MultiPoint, Polygon, Point
from scipy.optimize import differential_evolution, minimize


def pyramid_geom(w, l, h):
    verts = np.array([[0, 0, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    faces = np.array([[0, 2, 1],
                      [0, 3, 2],
                      [0, 4, 3],
                      [0, 1, 4],
                      [1, 3, 4],
                      [1, 2, 3]])
    return verts, faces


def gable_geom(w, l, h):
    verts = np.array([[0, + l / 2, h / 2],
                      [0, - l / 2, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    faces = np.array([[0, 1, 2],
                      [1, 3, 2],
                      [1, 0, 5],
                      [1, 5, 4],
                      [1, 4, 3],
                      [0, 2, 5],
                      [2, 3, 4],
                      [2, 4, 5]])
    return verts, faces


def hip_geom(w, l, lr, h):
    verts = np.array([[0, + l * lr / 2, h / 2],
                      [0, - l * lr / 2, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    faces = np.array([[0, 1, 2],
                      [1, 3, 2],
                      [1, 0, 5],
                      [1, 5, 4],
                      [1, 4, 3],
                      [0, 2, 5],
                      [2, 3, 4],
                      [2, 4, 5]])
    return verts, faces


def mansard_geom(w, l, wr, lr, h):
    verts = np.array([[- w * wr / 2, + l * lr / 2, h / 2],
                      [+ w * wr / 2, + l * lr / 2, h / 2],
                      [+ w * wr / 2, - l * lr / 2, h / 2],
                      [- w * wr / 2, - l * lr / 2, h / 2],
                      [- w / 2, + l / 2, - h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2]])
    faces = np.array([[0, 2, 1],
                      [0, 3, 2],
                      [0, 4, 7],
                      [0, 7, 3],
                      [3, 7, 6],
                      [3, 6, 2],
                      [2, 6, 5],
                      [2, 5, 1],
                      [0, 1, 5],
                      [0, 5, 4],
                      [4, 5, 6],
                      [4, 6, 7]])
    return verts, faces


def shed_geom(w, l, h):
    verts = np.array([[- w / 2, + l / 2, + h / 2],
                      [+ w / 2, + l / 2, + h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2],
                      [+ w / 2, + l / 2, - h / 2]])
    faces = np.array([[0, 2, 1],
                      [0, 3, 2],
                      [0, 4, 3],
                      [1, 2, 5],
                      [4, 5, 2],
                      [4, 2, 3],
                      [0, 1, 5],
                      [0, 5, 4]])
    return verts, faces


def box_geom(w, l, h):
    verts = np.array([[- w / 2, + l / 2, h / 2],
                      [+ w / 2, + l / 2, h / 2],
                      [+ w / 2, - l / 2, h / 2],
                      [- w / 2, - l / 2, h / 2],
                      [- w / 2, + l / 2, - h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2]])
    faces = np.array([[0, 2, 1],
                      [0, 3, 2],
                      [0, 4, 7],
                      [0, 7, 3],
                      [3, 7, 6],
                      [3, 6, 2],
                      [2, 6, 5],
                      [2, 5, 1],
                      [0, 1, 5],
                      [0, 5, 4],
                      [4, 5, 6],
                      [4, 6, 7]])
    return verts, faces


def pyramid_opt(x, args):
    verts, faces = pyramid_geom(x[0], x[1], x[2])
    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    dists1 = mesh(args)
    dists2 = cdist(verts, args).min(axis=1)
    return np.mean(np.square(dists1)) + np.mean(np.square(dists2))


def gable_opt(x, args):
    verts, faces = gable_geom(x[0], x[1], x[2])
    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    dists1 = mesh(args)
    dists2 = cdist(verts, args).min(axis=1)
    return np.mean(np.square(dists1)) + np.mean(np.square(dists2))


def hip_opt(x, args):
    verts, faces = hip_geom(x[0], x[1], x[2], x[3])
    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    dists1 = mesh(args)
    dists2 = cdist(verts, args).min(axis=1)
    return np.mean(np.square(dists1)) + np.mean(np.square(dists2))


def mansard_opt(x, args):
    verts, faces = mansard_geom(x[0], x[1], x[2], x[3], x[4])
    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    dists1 = mesh(args)
    dists2 = cdist(verts, args).min(axis=1)
    return np.mean(np.square(dists1)) + np.mean(np.square(dists2))


def shed_opt(x, args):
    verts, faces = shed_geom(x[0], x[1], x[2])
    verts = verts[0:4]
    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces[0:2])
    dists1 = mesh(args)
    dists2 = cdist(verts, args).min(axis=1)
    return np.mean(np.square(dists1)) + np.mean(np.square(dists2))


def evaluate_geom(x, primitive, sample, vis=False):
    if primitive == 'pyramid':
        verts, faces = pyramid_geom(x[0], x[1], x[2])
    elif primitive == 'gable':
        verts, faces = gable_geom(x[0], x[1], x[2])
    elif primitive == 'hip':
        verts, faces = hip_geom(x[0], x[1], x[2], x[3])
    elif primitive == 'mansard':
        verts, faces = mansard_geom(x[0], x[1], x[2], x[3], x[4])
    elif primitive == 'shed':
        verts, faces = shed_geom(x[0], x[1], x[2])

    verts = pc_RT(verts, x[- 4], x[- 3:])
    mesh = SDF(verts, faces)
    dists = mesh(sample)
    RMSE = np.sqrt(np.mean(np.square(dists)))

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(sample)
    geom = o3d.geometry.TriangleMesh()
    geom.vertices = o3d.utility.Vector3dVector(verts)
    geom.triangles = o3d.utility.Vector3iVector(faces)
    geom.compute_vertex_normals()

    if vis:
        o3d.visualization.draw_geometries([pc_o3d, geom])

    return RMSE, geom


def initialize_para(sample):
    bbx = MultiPoint(sample[:, 0:2]).minimum_rotated_rectangle
    x, y = bbx.exterior.coords.xy
    lens = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    w, l, h = min(lens), max(lens), sample[:, 2].max(axis=0) - sample[:, 2].min(axis=0)
    ids = np.argsort(sample[:, 2])[::-1]
    tpc = sample[ids[0:20]]
    pca = PCA(n_components=2)
    pca.fit(tpc[:, 0:2])
    kappa = np.arctan2(pca.components_[1][1], pca.components_[1][0])
    return w, l, h, kappa, bbx.area


def estimate_primitives(pc, types=['pyramid', 'gable', 'hip', 'mansard', 'shed', 'flat', 'curve'], vis=False):
    translation_local = pc.mean(axis=0)
    sample = pc - translation_local
    mw, ml, mh, kappa, area = initialize_para(sample)
    wbnd, lbnd, hbnd = (0.8 * mw, 1.2 * mw), (0.8 * ml, 1.2 * ml), (0.8 * mh, 1.2 * mh)
    kbnd, tbnd, rbnd = (- np.pi, + np.pi), (- max([ml, mh]), + max([ml, mh])), (0.3, 0.99)
    RMSEs, paras, o3d_geoms = [], [], []

    for primitive_type in types:
        if primitive_type == 'pyramid':
            opt_func = pyramid_opt
            initials = np.array([mw, ml, mh, kappa, 0, 0, 0])
            bnds = (wbnd, lbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
        elif primitive_type == 'gable':
            opt_func = gable_opt
            initials = np.array([mw, ml, mh, kappa, 0, 0, 0])
            bnds = (wbnd, lbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
        elif primitive_type == 'hip':
            opt_func = hip_opt
            initials = np.array([mw, ml, 0.8, mh, kappa, 0, 0, 0])
            bnds = (wbnd, lbnd, rbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
        elif primitive_type == 'mansard':
            opt_func = mansard_opt
            initials = np.array([mw, ml, 0.8, 0.8, mh, kappa, 0, 0, 0])
            bnds = (wbnd, lbnd, rbnd, rbnd, hbnd, kbnd, tbnd, tbnd, tbnd)
        else:  # Shed primitive
            opt_func = shed_opt
            initials = np.array([mw, ml, mh, 0, 0, 0, 0])
            bnds = ((0, 1.2 * ml), (0, 1.2 * ml), hbnd, kbnd, tbnd, tbnd, tbnd)

        if primitive_type != 'shed':
            rst = minimize(fun=opt_func, bounds=bnds, x0=initials, args=(sample,), method='Powell')
        else:
            rst = differential_evolution(opt_func, bnds, x0=initials, args=(sample,))

        para = rst.x.copy()
        para[- 3:] += translation_local
        RMSE, o3d_geom = evaluate_geom(para, primitive_type, sample + translation_local, vis)
        o3d_geoms.append(o3d_geom)
        RMSEs.append(RMSE)
        paras.append(para)

    return np.min(RMSEs), paras[np.argmin(RMSEs)], types[np.argmin(RMSEs)], o3d_geoms[np.argmin(RMSEs)]


if __name__ == "__main__":
    w, l, h, wr, lr = 1, 2, 0.6, 0.8, 0.6
    pyramid = pyramid_geom(w, l, h)
    gable = gable_geom(w, l, h)
    hip = hip_geom(w, l, lr, h)
    mansard = mansard_geom(w, l, wr, lr, h)
    shed = shed_geom(w, l, h)
    flat = shed_geom(w, l, 0)

    pyramid_mesh = o3d.geometry.TriangleMesh()
    pyramid_mesh.vertices = o3d.utility.Vector3dVector(pyramid[0])
    pyramid_mesh.triangles = o3d.utility.Vector3iVector(pyramid[1])
    pyramid_mesh.compute_vertex_normals()

    gable_mesh = o3d.geometry.TriangleMesh()
    gable_mesh.vertices = o3d.utility.Vector3dVector(gable[0])
    gable_mesh.triangles = o3d.utility.Vector3iVector(gable[1])
    gable_mesh.compute_vertex_normals()

    hip_mesh = o3d.geometry.TriangleMesh()
    hip_mesh.vertices = o3d.utility.Vector3dVector(hip[0])
    hip_mesh.triangles = o3d.utility.Vector3iVector(hip[1])
    hip_mesh.compute_vertex_normals()

    mansard_mesh = o3d.geometry.TriangleMesh()
    mansard_mesh.vertices = o3d.utility.Vector3dVector(mansard[0])
    mansard_mesh.triangles = o3d.utility.Vector3iVector(mansard[1])
    mansard_mesh.compute_vertex_normals()

    shed_mesh = o3d.geometry.TriangleMesh()
    shed_mesh.vertices = o3d.utility.Vector3dVector(shed[0])
    shed_mesh.triangles = o3d.utility.Vector3iVector(shed[1])
    shed_mesh.compute_vertex_normals()

    flat_mesh = o3d.geometry.TriangleMesh()
    flat_mesh.vertices = o3d.utility.Vector3dVector(flat[0])
    flat_mesh.triangles = o3d.utility.Vector3iVector(flat[1])
    flat_mesh.compute_vertex_normals()

    for mesh in [pyramid_mesh, gable_mesh, hip_mesh, mansard_mesh, shed_mesh, flat_mesh]:
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)