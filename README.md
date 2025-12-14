# Synthetic learning for primitive-based building model reconstruction from point clouds


This repository contains code for the paper "Synthetic learning for primitive-based building model reconstruction from point clouds". The project is implemented in Python and PyTorch, with additional utilities for evaluation and visualization.

## Roof Primitives

Design of the roof primitives, including the parameters and visualizations.
![primitives](https://github.com/user-attachments/assets/83589b53-6b3a-4bfd-9a97-b12af015e2bc)

## Roof Primitives

Design of the training procedure.
![training](https://github.com/user-attachments/assets/e16acfae-5c4c-465a-b93f-e41f41ba6a9a)


## Features

- **Point Cloud Processing**: Implements point cloud sampling, grouping, and feature extraction using neural networks.
- **Roof Primitive Determination**: Predicts roof types and geometric parameters of roof primitives from point cloud data.
- **Visualization**: Uses `pyvista` for 3D visualization of predictions and ground truth.
- **Evaluation Metrics**: Includes metrics like RMSE, reconstruction scores, and parameter errors for model evaluation.

## Repository Structure

- `demo`: Demo scripts for building roof primitive determination from building point clouds.
- `source`: Scripts the work in this repo, including model design, training, testing, and other utilities.
- `main_**`: Maing entrance for building primitive determination.

## Reconstruction Results
![NYC_reconstruction](https://github.com/user-attachments/assets/3cc53f4f-c4b8-46cc-a9eb-60b034cd2ec7)


## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- PyVista
- Matplotlib
- Pandas
- Scipy

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Citations
```bash
@ARTICLE{Li2019-iz,
  title   = "{GEOMETRIC} {OBJECT} {BASED} {BUILDING} {RECONSTRUCTION} {FROM}
             {SATELLITE} {IMAGERY} {DERIVED} {POINT} {CLOUDS}",
  author  = "Li, Zhixin and Xu, Bo and Shan, Jie",
  journal = "International Archives of the Photogrammetry, Remote Sensing \&
             Spatial Information Sciences",
  year    =  2019
}

@ARTICLE{Zhang2021-as,
  title     = "Optimal model fitting for building reconstruction from point
               clouds",
  author    = "Zhang, Wenyuan and Li, Zhixin and Shan, Jie",
  journal   = "IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.",
  publisher = "Institute of Electrical and Electronics Engineers (IEEE)",
  volume    =  14,
  pages     = "9636--9650",
  year      =  2021
}

@ARTICLE{Li2022-zb,
  title     = "{RANSAC}-based multi primitive building reconstruction from {3D}
               point clouds",
  author    = "Li, Zhixin and Shan, Jie",
  journal   = "ISPRS J. Photogramm. Remote Sens.",
  publisher = "Elsevier BV",
  month     =  jan,
  year      =  2022,
  language  = "en"
}
```
