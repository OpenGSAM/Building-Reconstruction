import alphashape
import networkx as nx
import numpy as np
from tqdm import tqdm
from osgeo import gdal, osr
from reconstruct_utils import *
from bldg_regularization import *
from sklearn.cluster import DBSCAN
from skspatial.objects import Plane
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint, Point, Polygon
from trimesh.creation import triangulate_polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

np.bool = np.bool_


def reconstruct_planes(planes, planes_para_all, dem_plane, theta, APD=2.40, simple_th=0.5, alpha=1.0, lod=2):
    DTM, pv_meshes, tri_meshes = [], [], []
    G = get_connectivity(planes, APD * 2)
    G[G == 2] = 1
    G_nx = nx.from_numpy_array(G)
    planes_cluster = [list(G_nx.subgraph(item).nodes()) for item in nx.connected_components(G_nx)]

    for nodes in tqdm(planes_cluster, desc=f'    Regularizing {len(planes)} planes...'):
        cloud = np.concatenate([planes[i] for i in nodes])[:, 0:2]
        boundary = regularize(check_multipolygon(alphashape.alphashape(cloud, alpha).simplify(simple_th)),
                              False, lod, theta)[0]
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
                      for poly in regularize(ashapes, False, lod, theta)]
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
                if flag and 3.33 <= rst.area <= 333:
                    remaining.append(rst)

            remaining_para = []
            for poly in remaining:
                dists = [item.exterior.distance(poly.centroid) for item in planes_reg]
                remaining_para.append(planes_para[np.argmin(dists)])

            planes_reg = planes_reg + remaining
            planes_para = planes_para + remaining_para

        # Visualization
        for plane in planes:
            color = np.random.uniform(0, 1, 3)
            plt.scatter(plane[:, 0], plane[:, 1], color=color, s=0.1)
        poly_reg = [np.array(poly.exterior.coords) for poly in planes_reg]
        planes_reg = []
        poly_cloud = np.concatenate(poly_reg)
        clusters = DBSCAN(eps=APD / 1.5, min_samples=1).fit(poly_cloud)
        cluster_centroids = []
        for lbl in np.unique(clusters.labels_):
            if lbl != -1:
                cluster_centroids.append(poly_cloud[clusters.labels_ == lbl].mean(axis=0))
        for exterior in poly_reg:
            for i in range(len(exterior)):
                dist = cdist(cluster_centroids, [exterior[i]])[:, 0]
                if Point(exterior[i]).within(boundary):
                    exterior[i] = cluster_centroids[np.argmin(dist)]
            planes_reg.append(Polygon(exterior))
            plt.plot(exterior[:, 0], exterior[:, 1], 'k-', lw=1.5)
        plt.plot(*boundary.exterior.xy, 'b-', lw=2)
        plt.axis('equal')
        plt.axis('off')
        plt.show()

        for plane in planes:
            color = np.random.uniform(0, 1, 3)
            plt.scatter(plane[:, 0], plane[:, 1], color=color, s=0.1)
        plt.plot(*boundary.exterior.xy, 'b-', lw=2)
        plt.axis('equal')
        plt.axis('off')
        plt.show()

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

        clusters = DBSCAN(eps=0.1, min_samples=2).fit(verts_all[:, 0:3])
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


def get_domain_direction(clouds, simple_th=0.5, alpha=1.0, lod=2):
    clouds = np.concatenate(clouds)
    ashape = check_multipolygon(alphashape.alphashape(clouds[:, 0:2], alpha).simplify(simple_th))
    border_reg, direction = regularize(ashape, lod=lod)
    IOU = ashape[0].intersection(border_reg[0]).area / ashape[0].union(border_reg[0]).area
    return border_reg[0], direction, IOU


if __name__ == "__main__":
    np.random.seed(42)
    cloud_fp = 'data/PU_LIDAR/test/PU_3DEP_18_117147178.las'
    dem_fp = f'data/PU_LIDAR/dem.tif'
    feet = 3.28084

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
    APD, lod = feet, 2

    # Read building bottom height
    pc_raw, t = read_las(cloud_fp)
    bldg_ext = np.array(MultiPoint(pc_raw[:, 0:2]).convex_hull.exterior.coords[:]) + t[0:2]
    bldg_ext_xyz = get_bldg_xyzs(bldg_ext, dem, ct, gt_inv) - t
    plane_coef = Plane.best_fit(bldg_ext_xyz)

    smin, smax = pc_raw.min(axis=0), pc_raw.max(axis=0)
    dem_plane = pv.Plane(center=tuple(plane_coef.point), direction=tuple(plane_coef.normal),
                         i_size=int(smax[0] - smin[0] + 500), j_size=int(smax[1] - smin[1] + 500))

    # RANSAC plane detection and facade removal
    planes, paras = facade_remove(ransac_cc(pc_raw, sp=40, mnd=20, eps=0.6, sample_res=-1), th=15)

    pids = [item for item in range(len(planes)) if
            np.mean(plane_coef.distance_points(planes[item])) > 3 * feet]
    planes, paras = [planes[i] for i in pids], [paras[i] for i in pids]

    pl = pv.Plotter()
    _, theta, IOU = get_domain_direction(planes, simple_th=APD, alpha=0.2, lod=lod)
    DTM, pv_meshes, tri_meshes = reconstruct_planes(planes, paras, dem_plane, theta, APD=APD,
                                                    simple_th=feet, alpha=0.33, lod=lod)

    for cloud in planes:
        c = np.random.uniform(0, 1, 3)
        pl.add_points(pv.PolyData(cloud).translate(t), render_points_as_spheres=True, point_size=5,
                      color=c, opacity=0.6)
    for mesh in pv_meshes:
        pl.add_mesh(mesh.translate(t), specular=0.7)
        edges = mesh.translate(t).extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_angle=10,
            manifold_edges=False,
        )
        pl.add_mesh(edges, color='k', line_width=3)
    pl.show()
    c = 1

    # pl = pv.Plotter(shape=(1, 3))
    # for lod in range(1, 4):
    #     _, theta, IOU = get_domain_direction(planes, simple_th=APD, alpha=0.2, lod=lod)
    #     DTM, pv_meshes, tri_meshes = reconstruct_planes(planes, paras, dem_plane, theta, APD=APD,
    #                                                     simple_th=APD, alpha=0.5, lod=lod)
    #     print(f'LOD {lod}: DTM: {DTM:.2f}m, IOU: {IOU:.2f}')
    #
    #     pl.subplot(0, lod - 1)
    #     if len(tri_meshes) > 0:
    #         tri_meshes = pv.wrap(trimesh.boolean.union(tri_meshes))
    #         pl.add_mesh(tri_meshes.translate(t), specular=0.7)
    #         edges = tri_meshes.translate(t).extract_feature_edges(
    #             boundary_edges=True,
    #             non_manifold_edges=False,
    #             feature_angle=10,
    #             manifold_edges=False,
    #         )
    #         # pl.add_mesh(edges, color='k', line_width=3)
    #     for mesh in pv_meshes:
    #         pl.add_mesh(mesh.translate(t), specular=0.7)
    #         edges = mesh.translate(t).extract_feature_edges(
    #             boundary_edges=True,
    #             non_manifold_edges=False,
    #             feature_angle=10,
    #             manifold_edges=False,
    #         )
    #         pl.add_mesh(edges, color='k', line_width=3)
    # pl.link_views()
    # pl.show()
