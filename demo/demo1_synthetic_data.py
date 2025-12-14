import tkinter as tk
from tkinter import ttk
from primitive import *
from data_utils import *

np.bool = np.bool_

# This is a demo tkinter GUI for generating synthetic roof data


class APP(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Demo 1 - Synthetic Data Generation')

        with open('primitive_info.json') as f:
            self.primitive_dict = json.load(f)
            self.roof_types = list(self.primitive_dict.keys())

        # App Frames
        self.setting_frame = tk.LabelFrame(self, text="Synthetic Settings")
        self.display_frame = tk.LabelFrame(self, text="Primitive Visualization")

        self.setting_frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
        self.display_frame.grid(row=0, column=1, padx=5, pady=5, sticky='n')

        # Setting Frame
        self.roof_type_label = tk.Label(self.setting_frame, text="Roof primitive type")
        self.roof_type_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.roof_type = ttk.Combobox(self.setting_frame, values=self.roof_types, state='readonly')
        self.roof_type.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.roof_type.current(0)

        self.width_label = tk.Label(self.setting_frame, text="Width")
        self.width_label.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.width = tk.Scale(self.setting_frame, from_=3, to=20, orient='horizontal')
        self.width.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.width.set(10)

        self.length_label = tk.Label(self.setting_frame, text="Length")
        self.length_label.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.length = tk.Scale(self.setting_frame, from_=3, to=20, orient='horizontal')
        self.length.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.length.set(10)

        self.height_label = tk.Label(self.setting_frame, text="Height")
        self.height_label.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.height = tk.Scale(self.setting_frame, from_=3, to=20, orient='horizontal')
        self.height.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.height.set(7)

        self.width_ratio_label = tk.Label(self.setting_frame, text="Width Ratio")
        self.width_ratio_label.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.width_ratio = tk.Scale(self.setting_frame, from_=0.3, to=0.9, resolution=0.01, orient='horizontal')
        self.width_ratio.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        self.width_ratio.set(0.5)

        self.length_ratio_label = tk.Label(self.setting_frame, text="Length Ratio")
        self.length_ratio_label.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

        self.length_ratio = tk.Scale(self.setting_frame, from_=0.3, to=0.9, resolution=0.01, orient='horizontal')
        self.length_ratio.grid(row=5, column=1, sticky="ew", padx=5, pady=5)
        self.length_ratio.set(0.5)

        self.height_ratio_label = tk.Label(self.setting_frame, text="Height Ratio")
        self.height_ratio_label.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        self.height_ratio = tk.Scale(self.setting_frame, from_=0.3, to=0.9, resolution=0.01, orient='horizontal')
        self.height_ratio.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
        self.height_ratio.set(0.7)

        self.noise_level_label = tk.Label(self.setting_frame, text="Noise Level (m)")
        self.noise_level_label.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        self.noise_level = tk.Scale(self.setting_frame, from_=0, to=0.3, resolution=0.01, orient='horizontal')
        self.noise_level.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
        self.noise_level.set(0.05)

        self.edge_label = tk.Label(self.setting_frame, text="Edge Point Proportion")
        self.edge_label.grid(row=8, column=0, sticky="ew", padx=5, pady=5)

        self.edge = tk.Scale(self.setting_frame, from_=0, to=0.3, resolution=0.01, orient='horizontal')
        self.edge.grid(row=8, column=1, sticky="ew", padx=5, pady=5)
        self.edge.set(0.15)

        self.occlusion_label = tk.Label(self.setting_frame, text="Roof Occlusion")
        self.occlusion_label.grid(row=9, column=0, sticky="ew", padx=5, pady=5)

        self.occlusion = ttk.Combobox(self.setting_frame, values=['True', 'False'], state='readonly')
        self.occlusion.grid(row=9, column=1, sticky="ew", padx=5, pady=5)
        self.occlusion.current(0)

        self.point_num_label = tk.Label(self.setting_frame, text="Number of points")
        self.point_num_label.grid(row=10, column=0, sticky="ew", padx=5, pady=5)

        self.point_num = tk.Scale(self.setting_frame, from_=256, to=2048, resolution=16, orient='horizontal')
        self.point_num.grid(row=10, column=1, sticky="ew", padx=5, pady=5)
        self.point_num.set(512)

        self.run_btn = tk.Button(self.setting_frame, text='Generate Data', command=self.generate_data)
        self.run_btn.grid(row=11, column=0, sticky="ew", columnspan=2, padx=5, pady=5)


    def generate_data(self, event=None):
        pt_num = int(self.point_num.get())
        roof = self.roof_type.get()
        w, l, h = float(self.width.get()), float(self.length.get()), float(self.height.get())
        wr, lr, hr = float(self.width_ratio.get()), float(self.length_ratio.get()), float(self.height_ratio.get())
        sigma = float(self.noise_level.get())
        edge = float(self.edge.get()) + 0.001
        occlusion = self.occlusion.get() == 'True'

        func = globals()[self.primitive_dict[roof]['func_name']]
        verts, faces, _, paras = func([w, l, wr, lr, h])
        face_num = len(faces)
        noise = np.random.normal(0, sigma, int(pt_num * (1 - edge)))

        if occlusion and self.primitive_dict[roof]['miss']:
            keep_ids = np.random.choice([item for item in range(face_num)], face_num - 1, replace=False)
            faces = [faces[item] for item in keep_ids]

        faces = np.concatenate(faces)
        faces_pl = [[3, face[0], face[1], face[2]] for face in faces]
        geom_mesh = pv.PolyData(pc_RT(verts, 0.0), faces_pl)
        edges = geom_mesh.extract_feature_edges()
        points = np.array(edges.points)
        lines = edges.lines.reshape(-1, 3)[:, 1:]

        self.geom = o3d.geometry.TriangleMesh()
        self.geom.vertices = o3d.utility.Vector3dVector(verts)
        self.geom.triangles = o3d.utility.Vector3iVector(faces)

        # Sample points on edges
        sample_edge = []
        for line in lines:
            p1, p2 = points[line[0]], points[line[1]]
            dl = p2 - p1
            sample = p1 + dl * np.linspace(0, 1, 200)[:, np.newaxis]
            sample_edge.append(sample)
        sample_edge = np.concatenate(sample_edge)
        sample_edge = sample_edge[np.random.choice(sample_edge.shape[0], int(pt_num * edge), replace=False)]

        sample = self.geom.sample_points_uniformly(int(pt_num * (1 - edge)), True)
        normals, sample = np.asarray(sample.normals), np.asarray(sample.points)
        sample += np.array([normals[:, 0] * noise, normals[:, 1] * noise, normals[:, 2] * noise]).T
        sample = np.concatenate([sample, sample_edge], axis=0)

        self.geom.compute_vertex_normals()
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(sample)
        o3d.visualization.draw_geometries([self.geom, self.pcd],
                                          mesh_show_wireframe=True,
                                          mesh_show_back_face=True,
                                          width=800,
                                          height=600,
                                          left=self.winfo_x() + self.winfo_width() + 20,
                                          top=self.winfo_y() + 30)


if __name__ == "__main__":
    app = APP()
    app.mainloop()

