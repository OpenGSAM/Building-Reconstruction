import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from reconstruct_utils import read_las
from matplotlib.ticker import PercentFormatter


if __name__ == '__main__':
    df_fd = 'primitive_old_results_5cm'
    df_fps = [f'{df_fd}/{f}' for f in os.listdir(df_fd) if f.endswith('.csv')]
    dfs = [pd.read_csv(df_fp) for df_fp in df_fps]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(f'{df_fd}/all.csv', index=False)
    # ds = np.array(df[['dw', 'dl', 'dh']]).flatten()
    ds = np.array(df['dw'])
    # props = np.array(df[['w_prop', 'l_prop', 'h_prop']]).flatten()
    props = np.array(df['w_prop'])
    RMSEs = np.array(df['RMSE'])
    index = np.where(abs(ds) < 0.5)
    ds = ds[index]
    props = props[index]
    RMSEs = RMSEs[RMSEs<0.5]
    print(f'Para mean: {np.mean(ds):.4f}, std: {np.std(ds):.4f}, max: {np.max(ds):.4f}, min: {np.min(ds):.4f}, '
          f'RMSE: {np.sqrt(np.mean(ds ** 2)):.4f}')
    print(f'Prop mean: {np.mean(props):.4f}, std: {np.std(props):.4f}, max: {np.max(props):.4f}, '
          f'min: {np.min(props):.4f}, ')
    print(f'Fitting error mean: {np.mean(RMSEs):.4f}, std: {np.std(RMSEs):.4f}, '
          f'max: {np.max(RMSEs):.4f}, min: {np.min(RMSEs):.4f}')

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].hist(ds, bins=30, color='b', alpha=0.7, weights=np.ones(len(ds)) / len(ds))
    axs[0].set_title('Parameter Error (m)')
    axs[0].set_xlabel('Error (m)')
    axs[0].set_ylabel('Frequency')
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[1].hist(props, bins=30, color='r', alpha=0.7, weights=np.ones(len(props)) / len(props))
    axs[1].set_title('Parameter Proportion Error (%)')
    axs[1].set_xlabel('Proportion Error (%)')
    axs[1].set_ylabel('Frequency')
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))
    axs[2].hist(RMSEs, bins=30, color='g', alpha=0.7, weights=np.ones(len(RMSEs)) / len(RMSEs))
    axs[2].set_title('Fitting Error (m)')
    axs[2].set_xlabel('RMSE (m)')
    axs[2].set_ylabel('Frequency')
    axs[2].yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.show()
    c = 1
    # mesh = pv.read('PPT&paper/example/PU_LIDAR_cloud100.obj').clean()
    # cloud_fp = 'data/PU_LIDAR/clouds/PU_LIDAR_cloud100.las'
    # cloud, t = read_las(cloud_fp)
    # # mesh = mesh.translate(-t)
    # cloud_pv = pv.PolyData(cloud)
    # dists = abs(cloud_pv.compute_implicit_distance(mesh)['implicit_distance'])
    #
    # pl = pv.Plotter()
    # edges = mesh.extract_feature_edges(
    #     boundary_edges=True,
    #     non_manifold_edges=False,
    #     feature_angle=10,
    #     manifold_edges=False,
    # )
    # pl.add_mesh(edges, color='k', line_width=2)
    # pl.add_mesh(mesh, specular=0.7)
    # pl.add_mesh(cloud, scalars=dists, cmap='coolwarm', point_size=3,
    #             scalar_bar_args={'title': 'Point-Model-Distance (m)', 'interactive': True, 'width': 0.15})
    # pl.show()
    #
    # c = 1
    #
    # # pl = pv.Plotter(shape=(1, 2))
    # # pl.subplot(0, 0)
    # # pl.add_mesh(old)
    # # pl.subplot(0, 1)
    # # pl.add_mesh(new)
    # # pl.link_views()
    # # pl.show()
