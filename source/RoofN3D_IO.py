import pickle
import shapely
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.distance import cdist


def multipoint_to_numpy(wkt_str):
    wkt_str = wkt_str.replace('MULTIPOINT Z (', '').replace(')', '')
    points = wkt_str.split(',')
    points = np.array([tuple(map(float, point.split())) for point in points])
    return points


def show_pv(points):
    pl = pv.Plotter()
    pl.add_points(pv.PolyData(points), render_points_as_spheres=True, point_size=5, color='blue')
    pl.show()


def remove_duplicates(points, sub_points, threshold=0.01):
    distances = cdist(points, sub_points)
    close_points = np.where(distances < threshold)
    mask = np.ones(len(points), dtype=bool)
    mask[close_points[0]] = False
    return points[mask]


def compute_wl(polygon_np):
    n = len(polygon_np)
    edges = polygon_np[1:n] - polygon_np[:n - 1]
    wl = np.linalg.norm(edges, axis=1)
    return max(wl), min(wl)


if __name__ == "__main__":
    roof_types = ['Saddleback roof', 'Two-sided hip roof', 'Pyramid roof']
    fp = 'data/roofn3d_raw_data/roofn3d_buildings.csv'
    df = pd.read_csv(fp)
    sample_num = 100
    dataset_test = {}


    for i in range(len(roof_types)):
        roof = roof_types[i]
        df_temp = df[df['class'] == roof]
        df_temp = df_temp.sample(sample_num, random_state=42)
        if roof == 'Saddleback roof':
            roof_cls = 1
            roof_type = 'gable'
        elif roof == 'Two-sided hip roof':
            roof_cls = 2
            roof_type = 'hip'
        else:
            roof_cls = 0
            roof_type = 'pyramid'

        for j in tqdm(range(len(df_temp)), desc=f'Processing {roof}'):
            row = df_temp.iloc[j]
            # shapely multipoint to numpy array
            points = multipoint_to_numpy(row['points'])
            try:
                sub_points = multipoint_to_numpy(row['unassignedsurfacegrowingpoints'])
                points = remove_duplicates(points, sub_points)
            except:
                pass
            translation = np.mean(points, axis=0)
            polygon = shapely.from_wkt(row['outline']).minimum_rotated_rectangle
            polygon_np = np.array(polygon.exterior.coords) - translation[0:2]
            length, width = compute_wl(polygon_np)
            points -= translation

            data_ = {'pc': points,
                     'roof_type': roof_type,
                     'roof_type_code': roof_cls,
                     'translation': translation,
                     'outline': polygon_np,
                     'length': length,
                     'width': width}
            dataset_test[str(row['id'])] = data_

    f_test = open('data/RoofN3D.pkl', 'wb')
    pickle.dump(dataset_test, f_test)
    c = 1

