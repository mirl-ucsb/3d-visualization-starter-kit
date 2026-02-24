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
