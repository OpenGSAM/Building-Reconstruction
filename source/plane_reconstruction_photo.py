import os.path
import alphashape
import numpy as np
from tqdm import tqdm
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from reconstruct_utils import *
from bldg_regularization import *
from skspatial.objects import Plane
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
from trimesh.creation import triangulate_polygon

np.bool = np.bool_


def reconstruct_planes(planes, planes_para_all, dem_plane):
    DTM, MTD, IOU, pv_meshes, tri_meshes = [], [], [], [], []
    simple_th, diff_th = 0.4, 0.4

    clouds = np.concatenate(planes)
    border_ashape = check_multipolygon(alphashape.alphashape(clouds[:, 0:2], 1).simplify(simple_th))
    border_reg, direction = regularize(border_ashape)
    border_reg = border_reg[0]

    ashapes, planes_para = [], []
    for plane_idx, plane in enumerate(tqdm(planes, desc=f'Computing alphashapes...')):
        shapes = check_multipolygon(alphashape.alphashape(plane[:, 0:2], 1), 'all')
        for shape in shapes:
            ashapes.append(shape.simplify(simple_th).buffer(simple_th / 2, join_style='mitre'))
            planes_para.append(planes_para_all[plane_idx])

    planes_reg = [check_multipolygon(border_reg.intersection(poly), 'max')[0]
                  for poly in regularize(ashapes, False, 2, direction)]
    ids = [item for item in range(len(planes_reg))]

    for i in ids:
        for j in [item for item in ids if item != i]:
            if planes_reg[i].intersects(planes_reg[j]):
                if planes_reg[i].area > planes_reg[j].area:
                    diff = check_multipolygon(planes_reg[j].difference(planes_reg[i]), type_return='max')[0]
                    if diff.area >= diff_th * planes_reg[j].area:
                        planes_reg[j] = diff
                else:
                    diff = check_multipolygon(planes_reg[i].difference(planes_reg[j]), type_return='max')[0]
                    if diff.area >= diff_th * planes_reg[i].area:
                        planes_reg[i] = diff
    planes_para = [planes_para[i] for i in ids if not planes_reg[i].is_empty and
                   planes_reg[i].geom_type != 'LineString']
    planes_reg = [planes_reg[i] for i in ids if not planes_reg[i].is_empty and
                  planes_reg[i].geom_type != 'LineString']

    remaining = border_reg.difference(cascaded_union(planes_reg))
    if not remaining.is_empty:
        remaining_temp = check_multipolygon(remaining, type_return='all')
        remaining = []
        for poly in remaining_temp:
            flag, rst = simplify_by_angle(poly)
            if flag and rst.area >= 1:
                remaining.append(rst)

        remaining_para = []
        for poly in remaining:
            dists = [item.exterior.distance(poly.centroid) for item in planes_reg]
            remaining_para.append(planes_para[np.argmin(dists)])

        planes_reg = planes_reg + remaining
        planes_para = planes_para + remaining_para

    for i in range(len(planes_reg)):
        vertices, faces = triangulate_polygon(planes_reg[i])
        vertices = np.array([get_vert_z(p, planes_para[i].point, planes_para[i].normal) for p in vertices])

        if len(faces) == 0:
            faces = [[3, 0, 1, 2]]
        else:
            faces = np.insert(faces, 0, 3, axis=1)

        roof_mesh = pv.PolyData(vertices, faces)
        mesh = roof_mesh.extrude_trim((0, 0, -1.0), dem_plane).clean().triangulate()
        tri_mesh = trimesh.Trimesh(mesh.points, mesh.faces.reshape((mesh.n_cells, 4))[:, 1:])
        tri_mesh.fix_normals()
        if tri_mesh.is_volume:
            tri_meshes.append(tri_mesh)
        else:
            pv_meshes.append(mesh)

    return DTM, MTD, IOU, pv_meshes, pv.wrap(trimesh.boolean.union(tri_meshes))


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



if __name__ == "__main__":
    np.random.seed(42)
    pc_fd = 'data/AgEagle_A1/bldg_las'
    out_fd = 'data/AgEagle_A1/reconstruction/photo/'
    if not os.path.exists(out_fd):
        os.mkdir(out_fd)

    pc_fps = [os.path.join(pc_fd, item) for item in os.listdir(pc_fd)]
    print('Loading DEM...')
    dem_fp = 'data/AgEagle_A1/dem.tif'
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

    APD = 0.2
    batch_size, vis = 16, True

    for fp in pc_fps:
        fname = os.path.basename(fp).split('.')[0]
        out_fp = os.path.join(out_fd, f'{fname}.obj')

        if os.path.exists(out_fp):
            print(f'{fname} exist, skip')
        else:
            print(f'Working on {fname}')
            pc_raw, t = read_las(fp)
            bldg_ext = np.array(MultiPoint(pc_raw[:, 0:2]).convex_hull.exterior.coords[:]) + t[0:2]
            bldg_ext_xyz = get_bldg_xyzs(bldg_ext, dem, ct, gt_inv) - t
            plane_coef = Plane.best_fit(bldg_ext_xyz)

            smin, smax = pc_raw.min(axis=0), pc_raw.max(axis=0)
            dem_plane = pv.Plane(center=tuple(plane_coef.point), direction=tuple(plane_coef.normal),
                                 i_size=int(smax[0] - smin[0] + 500), j_size=int(smax[1] - smin[1] + 500))

            pc_height = plane_coef.distance_points(pc_raw)
            pc_raw = pc_raw[pc_height >= 2]
            planes, paras = facade_remove(ransac_cc(pc_raw, sp=256, mnd=20, eps=0.15))
            # pl = pv.Plotter()
            # for cloud in planes:
            #     c = np.random.uniform(0, 1, 3)
            #     pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c, opacity=0.6)
            # pl.show()

            # if len(planes) == 0:
            #     planes = ransac_o3d(pc_raw)

            if len(planes) == 0:
                print('No planes found, skip.')
            else:
                plane_DTM, plane_MTD, plane_IOU, pv_meshes, tri_mesh = reconstruct_planes(planes, paras, dem_plane)

                pl = pv.Plotter()
                pl.add_mesh(tri_mesh.translate(t), show_edges=True)
                for mesh in pv_meshes:
                    pl.add_mesh(mesh.translate(t), show_edges=True)
                pl.export_obj(out_fp)
                c = 1
