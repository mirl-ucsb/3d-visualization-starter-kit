import numpy as np
import pyvista as pv

mesh = pv.read("/Users/veronicabodenstein/Downloads/VB_01.27.2026_completescan1.obj")
mesh = mesh.triangulate().clean(tolerance=1e-7)
mesh = mesh.compute_normals(point_normals=True, auto_orient_normals=True, consistent_normals=True)

# Light smoothing for stability (doesn't change overall form much)
smooth = mesh.smooth(n_iter=20, relaxation_factor=0.01, feature_smoothing=False)

# Curvature -> k1,k2
H = smooth.curvature("mean")
K = smooth.curvature("gaussian")
disc = np.maximum(H**2 - K, 0.0)
root = np.sqrt(disc)
k1 = H + root
k2 = H - root

# Anisotropy ratio: near 1 = strongly directional (tool-like), near 0 = isotropic
anis = (np.abs(k1) - np.abs(k2)) / (np.abs(k1) + np.abs(k2) + 1e-12)
mesh["anisotropy"] = anis
mesh["curvedness"] = np.sqrt(0.5 * (k1**2 + k2**2))

# View anisotropy: tool marks often show as streaky high-anisotropy regions
p = pv.Plotter()
p.add_mesh(mesh, scalars="anisotropy", cmap="turbo",
           clim=np.percentile(anis[np.isfinite(anis)], [2, 98]),
           smooth_shading=True)
p.add_text("Curvature anisotropy (directional carving tends to be high)", font_size=12)
p.add_axes()
p.show()