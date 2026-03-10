import trimesh
import numpy as np
import json
import csv
import sys
import os

# ============================
# Utility Functions
# ============================

def mesh_report(mesh):
    report = {}

    report["num_vertices"] = int(len(mesh.vertices))
    report["num_faces"] = int(len(mesh.faces))
    report["surface_area"] = float(mesh.area)
    report["bounds"] = mesh.bounds.tolist()
    report["center_mass"] = mesh.center_mass.tolist()
    report["watertight"] = bool(mesh.is_watertight)
    report["euler_number"] = int(mesh.euler_number)
    report["winding_consistent"] = bool(mesh.is_winding_consistent)

    if mesh.is_watertight:
        report["volume"] = float(mesh.volume)
    else:
        report["volume"] = None

    return report


def circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)


def slice_along_z(mesh, n_slices=25):
    z_min, z_max = mesh.bounds[:, 2]
    heights = np.linspace(z_min, z_max, n_slices)

    slice_metrics = []

    for h in heights:
        section = mesh.section(
            plane_origin=[0, 0, h],
            plane_normal=[0, 0, 1]
        )

        if section is not None:
            slice_2D, _ = section.to_planar()

            area = slice_2D.area
            perimeter = slice_2D.length

            slice_metrics.append({
                "z_height": float(h),
                "area": float(area),
                "perimeter": float(perimeter),
                "circularity": float(circularity(area, perimeter))
            })

    return slice_metrics


def surface_sampling(mesh, n_points=5000):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    pq = trimesh.proximity.ProximityQuery(mesh)
    distances = pq.signed_distance(points)

    return {
        "mean_signed_distance": float(np.mean(distances)),
        "std_signed_distance": float(np.std(distances)),
        "max_signed_distance": float(np.max(distances)),
        "min_signed_distance": float(np.min(distances))
    }


# ============================
# Main Execution
# ============================

def main(mesh_path):

    if not os.path.exists(mesh_path):
        print("File not found.")
        return

    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)

    print("Generating mesh report...")
    report = mesh_report(mesh)

    print("Running multi-slice analysis...")
    slice_metrics = slice_along_z(mesh)

    print("Sampling surface...")
    surface_stats = surface_sampling(mesh)

    report["surface_sampling"] = surface_stats
    report["slice_count"] = len(slice_metrics)

    base_name = os.path.splitext(os.path.basename(mesh_path))[0]

    # Save JSON report
    with open(f"{base_name}_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save slice metrics CSV
    with open(f"{base_name}_slices.csv", "w", newline="") as csvfile:
        fieldnames = ["z_height", "area", "perimeter", "circularity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in slice_metrics:
            writer.writerow(row)

    print("Analysis complete.")
    print(f"Report saved as {base_name}_report.json")
    print(f"Slices saved as {base_name}_slices.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_mesh.py your_mesh.obj")
    else:
        main(sys.argv[1])
