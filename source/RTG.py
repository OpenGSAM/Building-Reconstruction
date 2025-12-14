import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from reconstruct_utils import *
from bldg_regularization import *
from skspatial.objects import Plane
from shapely.geometry import MultiPoint

np.bool = np.bool_


def get_bldg_xyzs(bldg_ext, dem, ct, gt_inv):
    bldg_ext_xyz = []
    for pt in bldg_ext:
        mapx, mapy, z = ct.TransformPoint(pt[0], pt[1])
        pixel_x, pixel_y = gdal.ApplyGeoTransform(gt_inv, mapx, mapy)
        pixel_x = round(pixel_x)
        pixel_y = round(pixel_y)
        pixel_x = max(min(pixel_x, width - 1), 0)
        pixel_y = max(min(pixel_y, height - 1), 0)
        bldg_ext_xyz.append([pt[0], pt[1], dem[pixel_y, pixel_x] * 3.28083333])
    return np.array(bldg_ext_xyz)


if __name__ == "__main__":
    np.random.seed(42)
    pc_fps = ['D:\\li2887\\BuildingReconstructionDL\\data\\PU_campus\\bldg_las\\PU_3DEP_18267.las']
    print('Loading DEM...')
    dem_fp = 'data/PU_campus/USGS_1M_16_x50y448_IN_Indiana_Statewide_LiDAR_2017_B17.tif'
    ds = gdal.Open(dem_fp)
    gt = ds.GetGeoTransform()
    dem = np.array(ds.ReadAsArray())
    width = ds.RasterXSize
    height = ds.RasterYSize
    point_srs = osr.SpatialReference()
    point_srs.ImportFromEPSG(2968)
    point_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(ds.GetProjection())
    ct = osr.CoordinateTransformation(point_srs, dem_srs)
    gt_inv = gdal.InvGeoTransform(gt)

    APD = 2.33
    batch_size, vis = 16, True

    for fp in pc_fps:
        pc_raw, t = read_las(fp)
        bldg_ext = np.array(MultiPoint(pc_raw[:, 0:2]).convex_hull.exterior.coords[:]) + t[0:2]
        bldg_ext_xyz = get_bldg_xyzs(bldg_ext, dem, ct, gt_inv) - t
        plane_coef = Plane.best_fit(bldg_ext_xyz)

        smin, smax = pc_raw.min(axis=0), pc_raw.max(axis=0)
        dem_plane = pv.Plane(center=tuple(plane_coef.point), direction=tuple(plane_coef.normal),
                             i_size=int(smax[0] - smin[0] + 500), j_size=int(smax[1] - smin[1] + 500))

        pc_height = plane_coef.distance_points(pc_raw)
        pc_raw = pc_raw[pc_height >= 3.28 * 2]
        planes = ransac_cc(pc_raw)
        connectivity = get_connectivity(planes, 1.5 * APD)

        pl = pv.Plotter()
        centers = []
        for idx, cloud in enumerate(planes):
            c = np.random.uniform(0, 1, 3)
            center = cloud.mean(axis=0)
            centers.append(center)
            pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c)

        centers_pv = pv.PolyData(centers)
        centers_pv["Labels"] = [f"{i}" for i in range(centers_pv.n_points)]
        pl.add_point_labels(centers_pv, "Labels", point_size=10)

        pl.show()

        pl = pv.Plotter()
        for idx in [2, 1]:
            c = np.random.uniform(0, 1, 3)
            pl.add_points(pv.PolyData(planes[idx]), render_points_as_spheres=True, point_size=8, color=c)
        pl.show()
        c = 1

