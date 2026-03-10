import numpy as np
import pyvista as pv

mesh = pv.read("/Users/veronicabodenstein/Downloads/VB_01.27.2026_completescan1.obj")


mesh = mesh.triangulate().clean(tolerance=1e-7)


mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.1)
mesh = mesh.compute_normals(point_normals=True, auto_orient_normals=True)
mesh = mesh.compute_normals(
    point_normals=True,
    cell_normals=False,
    auto_orient_normals=True,
    consistent_normals=True
)

# Curvature
mean_curv = mesh.curvature(curv_type="mean")
gauss_curv = mesh.curvature(curv_type="gaussian")

mesh["mean_curvature"] = mean_curv
mesh["gaussian_curvature"] = gauss_curv

# Principal curvatures from H and K
disc = np.maximum(mean_curv**2 - gauss_curv, 0.0)
root = np.sqrt(disc)
k1 = mean_curv + root
k2 = mean_curv - root
mesh["k1"] = k1
mesh["k2"] = k2

# Curvedness, shape index
mesh["curvedness"] = np.sqrt(0.5 * (k1**2 + k2**2))
eps = 1e-12
mesh["shape_index"] = (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + eps))

def robust_clim(arr, lo=2, hi=98):
    finite = arr[np.isfinite(arr)]
    return np.percentile(finite, [lo, hi])

def show_scalar(name, cmap="turbo", clim=None):
    p = pv.Plotter()
    p.add_mesh(mesh, scalars=name, cmap=cmap, clim=clim, smooth_shading=True)
    p.add_axes()
    p.add_text(name, font_size=12)
    p.show()

show_scalar("mean_curvature", clim=robust_clim(mesh["mean_curvature"]))
show_scalar("gaussian_curvature", clim=robust_clim(mesh["gaussian_curvature"]))
show_scalar("curvedness", clim=robust_clim(mesh["curvedness"]))
show_scalar("shape_index", clim=(-1, 1))

import numpy as np
import pyvista as pv

mesh = pv.read("/Users/veronicabodenstein/3D Scanning MIRL/VB.03.04.2026.obj")


mesh = mesh.triangulate().clean(tolerance=1e-7)


mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.1)
mesh = mesh.compute_normals(point_normals=True, auto_orient_normals=True)
mesh = mesh.compute_normals(
    point_normals=True,
    cell_normals=False,
    auto_orient_normals=True,
    consistent_normals=True
)

# Curvature
mean_curv = mesh.curvature(curv_type="mean")
gauss_curv = mesh.curvature(curv_type="gaussian")

mesh["mean_curvature"] = mean_curv
mesh["gaussian_curvature"] = gauss_curv

# Principal curvatures from H and K
disc = np.maximum(mean_curv**2 - gauss_curv, 0.0)
root = np.sqrt(disc)
k1 = mean_curv + root
k2 = mean_curv - root
mesh["k1"] = k1
mesh["k2"] = k2

# Curvedness, shape index
mesh["curvedness"] = np.sqrt(0.5 * (k1**2 + k2**2))
eps = 1e-12
mesh["shape_index"] = (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + eps))

def robust_clim(arr, lo=2, hi=98):
    finite = arr[np.isfinite(arr)]
    return np.percentile(finite, [lo, hi])

def show_scalar(name, cmap="turbo", clim=None):
    p = pv.Plotter()
    p.add_mesh(mesh, scalars=name, cmap=cmap, clim=clim, smooth_shading=True)
    p.add_axes()
    p.add_text(name, font_size=12)
    p.show()

show_scalar("mean_curvature", clim=robust_clim(mesh["mean_curvature"]))
show_scalar("gaussian_curvature", clim=robust_clim(mesh["gaussian_curvature"]))
show_scalar("curvedness", clim=robust_clim(mesh["curvedness"]))
show_scalar("shape_index", clim=(-1, 1))


#EXPORTING!

import pandas as pd
import os


output_dir = "/Users/veronicabodenstein/3D Scanning MIRL/Analyzed_Exports/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

surfaces = ["mean_curvature", "gaussian_curvature", "curvedness", "shape_index"]

# Export the 3D Meshes (.vtk)
for surf_name in surfaces:
    export_mesh = mesh.copy()
    export_mesh.set_active_scalars(surf_name)
    vtk_filename = f"VB_Surface_{surf_name}.vtk"
    vtk_path = os.path.join(output_dir, vtk_filename)
    export_mesh.save(vtk_path)
    print(f"Mesh Exported: {vtk_filename}")

# Export the Numerical Data (.csv)
csv_data = {s: mesh[s] for s in surfaces}
df = pd.DataFrame(csv_data)

csv_path = os.path.join(output_dir, "VB_Curvature_Data_Analysis.csv")
df.to_csv(csv_path, index=False)

print(f"Quantitative data exported to: {csv_path}")
