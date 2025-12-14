import os
import laspy
import numpy as np
from osgeo import gdal
from pyproj import CRS


if __name__ == "__main__":
    fps = ['data/NYC_SPG/source/NYC_dsm/dsm/data/0735525w_404825n_20230120T124154Z_dsm.tif',
           'data/NYC_SPG/source/NYC_dsm/dsm/data/0735528w_404612n_20230120T124154Z_dsm.tif',
           'data/NYC_SPG/source/NYC_dsm/dsm/data/0735822w_404614n_20230120T124154Z_dsm.tif',
           'data/NYC_SPG/source/NYC_dsm/dsm/data/0740117w_404615n_20230120T124154Z_dsm.tif']
    out_fd = 'data/NYC_SPG/source/'

    for fp in fps:
        fn = os.path.basename(fp).replace('.tif', '.las')
        out_fp = os.path.join(out_fd, fn)
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        dsm = gdal.Open(fp)
        band = dsm.GetRasterBand(1)
        data = band.ReadAsArray()
        geotransform = dsm.GetGeoTransform()
        wkt = dsm.GetProjection()
        header.add_crs(CRS.from_wkt(wkt))
        x = np.arange(0, dsm.RasterXSize) * geotransform[1] + geotransform[0]
        y = np.arange(0, dsm.RasterYSize) * geotransform[5] + geotransform[3]
        xx, yy = np.meshgrid(x, y)
        points = np.array([xx.flatten(), yy.flatten(), data.flatten()]).T
        points = points[points[:, 2] != -32767]
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.write(out_fp)
        print(f'{out_fp} saved.')