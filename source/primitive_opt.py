import json
import numpy as np
import open3d as o3d
import pyvista as pv

np.bool = np.bool_


def pyramid_geom(x, face_merge=False, return_paras=True):
    w, l, _, _, h = x
    verts = np.array([[0, 0, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    roof_faces = [[[0, 2, 1]],

                  [[0, 3, 2]],

                  [[0, 4, 3]],

                  [[0, 1, 4]]]
    bottom_faces = np.array([[1, 3, 4],
                             [1, 2, 3]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, 0, 0, h]
    else:
        return verts, roof_faces, bottom_faces


def semi_pyramid_geom(x, face_merge=False, return_paras=True):
    w, l, _, _, h = x
    verts = np.array([[0, 0, h / 2],
                      [+ w, + l / 2, - h / 2],
                      [+ w, - l / 2, - h / 2],
                      [0, - l / 2, - h / 2],
                      [0, + l / 2, - h / 2]])
    verts -= np.array([w / 2, 0, 0])
    roof_faces = [[[0, 2, 1]],

                  [[0, 3, 2]],

                  [[0, 1, 4]]]
    side_faces = np.array([[0, 4, 3]])
    bottom_faces = np.array([[1, 3, 4],
                             [1, 2, 3]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, 0, 0, h]
    else:
        return verts, roof_faces, bottom_faces


def gable_geom(x, face_merge=False, return_paras=True):
    w, l, _, _, h = x
    verts = np.array([[0, + l / 2, h / 2],
                      [0, - l / 2, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    roof_faces = [[[0, 1, 2],
                   [1, 3, 2]],

                  [[1, 0, 5],
                   [1, 5, 4]]]
    side_faces = np.array([[0, 2, 5],
                           [1, 4, 3]])
    bottom_faces = np.array([[2, 3, 4],
                             [2, 4, 5]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, 0, 1, h]
    else:
        return verts, roof_faces, bottom_faces


def hip_geom(x, face_merge=False, return_paras=True):
    w, l, _, lr, h = x
    verts = np.array([[0, + lr / 2, h / 2],
                      [0, - lr / 2, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    roof_faces = [[[0, 2, 5]],

                  [[1, 4, 3]],

                  [[0, 1, 2],
                   [1, 3, 2]],

                  [[1, 0, 5],
                   [1, 5, 4]]]
    bottom_faces = np.array([[2, 3, 4],
                             [2, 4, 5]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, 0, lr, h]
    else:
        return verts, roof_faces, bottom_faces


def semi_hip_geom(x, face_merge=False, return_paras=True):
    w, l, _, lr, h = x
    verts = np.array([[0, + lr, h / 2],
                      [0, 0, h / 2],
                      [+ w / 2, + l, - h / 2],
                      [+ w / 2, 0, - h / 2],
                      [- w / 2, 0, - h / 2],
                      [- w / 2, + l, - h / 2]])
    verts -= np.array([0, l / 2, 0])
    side_faces = np.array([[1, 4, 3]])
    roof_faces = [[[0, 2, 5]],

                  [[0, 1, 2],
                   [1, 3, 2]],

                  [[1, 0, 5],
                   [1, 5, 4]]]
    bottom_faces = np.array([[2, 3, 4],
                             [2, 4, 5]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, 0, lr, h]
    else:
        return verts, roof_faces, bottom_faces


def mansard_geom(x, face_merge=False, return_paras=True):
    w, l, wr, lr, h = x
    verts = np.array([[- wr / 2, + lr / 2, h / 2],
                      [+ wr / 2, + lr / 2, h / 2],
                      [+ wr / 2, - lr / 2, h / 2],
                      [- wr / 2, - lr / 2, h / 2],
                      [- w / 2, + l / 2, - h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2]])
    roof_faces = [[[0, 2, 1],
                   [0, 3, 2]],

                  [[0, 4, 7],
                   [0, 7, 3]],

                  [[3, 7, 6],
                   [3, 6, 2]],

                  [[2, 6, 5],
                   [2, 5, 1]],

                  [[0, 1, 5],
                   [0, 5, 4]]]
    bottom_faces = np.array([[4, 5, 6],
                             [4, 6, 7]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, wr, lr, h]
    else:
        return verts, roof_faces, bottom_faces


def semi_mansard_geom(x, face_merge=False, return_paras=True):
    w, l, wr, lr, h = x
    verts = np.array([[0, + lr / 2, h / 2],
                      [+ wr, + lr / 2, h / 2],
                      [+ wr, - lr / 2, h / 2],
                      [0, - lr / 2, h / 2],
                      [0, + l / 2, - h / 2],
                      [+ w, + l / 2, - h / 2],
                      [+ w, - l / 2, - h / 2],
                      [0, - l / 2, - h / 2]])
    verts -= np.array([w / 2, 0, 0])
    roof_faces = [[[0, 2, 1],
                   [0, 3, 2]],

                  [[3, 7, 6],
                   [3, 6, 2]],

                  [[2, 6, 5],
                   [2, 5, 1]],

                  [[0, 1, 5],
                   [0, 5, 4]]]
    side_faces = np.array([[[0, 4, 7], [0, 7, 3]]])
    bottom_faces = np.array([[4, 5, 6],
                             [4, 6, 7]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, wr, lr, h]
    else:
        return verts, roof_faces, bottom_faces


def saltbox_geom(x, face_merge=False, return_paras=True):
    w, l, wr, hr, h = x
    verts = np.array([[0, + l / 2, h / 2],
                      [0, - l / 2, h / 2],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- wr / 2, - l / 2, hr],
                      [- wr / 2, + l / 2, hr]])
    roof_faces = [[[0, 1, 2],
                   [1, 3, 2]],

                  [[1, 0, 5],
                   [1, 5, 4]]]
    side_faces = np.array([[0, 2, 5],
                           [1, 4, 3]])
    bottom_faces = np.array([[2, 3, 4],
                             [2, 4, 5]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, wr, hr, h]
    else:
        return verts, roof_faces, bottom_faces


def gambrel_geom(x, face_merge=False, return_paras=True):
    w, l, wr, hr, h = x
    verts = np.array([[0, + l / 2, h / 2],
                      [0, - l / 2, h / 2],
                      [- wr / 2, - l / 2, hr],
                      [- wr / 2, + l / 2, hr],
                      [+ wr / 2, - l / 2, hr],
                      [+ wr / 2, + l / 2, hr],
                      [+ w / 2, + l / 2, - h / 2],
                      [+ w / 2, - l / 2, - h / 2],
                      [- w / 2, - l / 2, - h / 2],
                      [- w / 2, + l / 2, - h / 2]])
    roof_faces = [[[8, 3, 9],
                   [8, 2, 3]],

                  [[2, 1, 3],
                   [1, 0, 3]],

                  [[0, 1, 5],
                   [1, 4, 5]],

                  [[5, 4, 7],
                   [5, 7, 6]]]

    bottom_faces = np.array([[8, 9, 6],
                             [8, 6, 7]])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, bottom_faces, [w, l, wr, hr, h]
    else:
        return verts, roof_faces, bottom_faces


def round_geom(x, face_merge=False, return_paras=True):
    w, l, _, _, h = x

    ellipsoid = pv.ParametricEllipsoid(w / 2, 1, h)
    slice_y = ellipsoid.slice(normal='y')
    surface = slice_y.extrude([0, l, 0], capping=False).triangulate()
    surface = surface.clip([0, 0, - 1], [0, 0, 0])
    verts = np.array(surface.points) + [0, - l / 2, - h / 2]
    roof_faces = surface.faces.reshape((-1, 4))[:, [3, 2, 1]]
    roof_faces = np.array([roof_faces])

    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(verts)
    # mesh.triangles = o3d.utility.Vector3iVector(roof_faces[0])
    # mesh.compute_triangle_normals()
    # o3d.visualization.draw_geometries([mesh])

    if face_merge:
        roof_faces = np.concatenate(roof_faces)

    if return_paras:
        return verts, roof_faces, None, [w, l, 1, 1, h]
    else:
        return verts, roof_faces, None


if __name__ == "__main__":
    primitive_name = ['pyramid', 'gable', 'hip', 'mansard', 'round',
                      'saltbox', 'gambrel', 'semi_pyramid', 'semi_hip', 'semi_mansard']
    primitive_code = [i for i in range(len(primitive_name))]
    primitive_func = [pyramid_geom, gable_geom, hip_geom, mansard_geom, round_geom, saltbox_geom,
                      gambrel_geom, semi_pyramid_geom, semi_hip_geom, semi_mansard_geom]
    primitive_para_num = [3, 3, 4, 5, 3, 5, 5, 3, 4, 5]
    primitive_cycle = ['p', 'p', 'p', 'p', 'p', '2p', 'p', '2p', '2p', '2p']
    primitive_miss = [True, False, True, True, False, False, False, False, False, False]
    primitive_ambiguity = [True, False, False, True, False, False, False, False, False, False]
    primitive_dict = {}

    for i in primitive_code:
        primitive_dict[primitive_name[i]] = {'code': i,
                                             'cycle': primitive_cycle[i],
                                             'miss': primitive_miss[i],
                                             'func_name': primitive_func[i].__name__,
                                             'para_num': primitive_para_num[i],
                                             'ambiguity': primitive_ambiguity[i]}

    with open('primitive_info.json', 'w') as f:
        json.dump(primitive_dict, f)
