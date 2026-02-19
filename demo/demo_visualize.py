# demo/demo_visualize.py
"""
Visualize outputs from src.synthetic_generator:
 - polygon outline
 - sampled floor points (Nx3)
 - occupancy image + bbox
 - overlay of polygon mapped onto occupancy image

Run:
    python demo/demo_visualize.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

# Import your synthetic generator (adjust import if your package name differs)
from src.synthetic_generator import (
    make_realistic_room,
    sample_floor_points,
    polygon_to_occupancy_image,
)

OUTDIR = "demo_out"
os.makedirs(OUTDIR, exist_ok=True)

def plot_polygon_and_points(poly: Polygon, pts: np.ndarray, title=""):
    fig, ax = plt.subplots(figsize=(8,8))
    x,y = poly.exterior.xy
    ax.plot(x, y, linewidth=2, color="tab:blue", label="room outline")
    # draw holes if any (furniture/obstacles)
    hole_label_added = False
    for interior in poly.interiors:
        xi, yi = zip(*interior.coords)
        label = "furniture/obstacles" if not hole_label_added else None
        ax.plot(xi, yi, linewidth=1.5, color="tab:red", label=label, linestyle="--")
        ax.fill(xi, yi, color="tab:red", alpha=0.3)
        hole_label_added = True
    ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.5, color="tab:green", label="sampled points")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax

def show_occupancy_and_overlay(occ, bbox, poly: Polygon, title="occupancy"):
    minx, miny, maxx, maxy = bbox
    H, W = occ.shape
    width_m = maxx - minx
    height_m = maxy - miny
    # show occupancy
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax0, ax1 = ax
    ax0.imshow(occ, cmap="gray", origin="upper")
    ax0.set_title(title + " (pixels)")

    # convert polygon to pixel coords for overlay
    def world_to_pix(x,y):
        u = (x - minx) / width_m * (W-1)
        v = (maxy - y) / height_m * (H-1)  # flip Y
        return u, v
    exterior = np.array(list(poly.exterior.coords))
    pix = np.array([world_to_pix(x,y) for x,y in exterior])
    ax0.plot(pix[:,0], pix[:,1], color="red", linewidth=2)

    # show world coords plot as well with occupancy's bbox to compare
    ax1.imshow(occ, cmap="gray", origin="upper", extent=(minx, maxx, miny, maxy))
    ax1.plot(exterior[:,0], exterior[:,1], color="red", linewidth=2)
    ax1.set_title("occupancy shown in world coordinates (extent set)")
    ax1.set_aspect("equal")
    return fig, ax

def try_open3d_view(points3d, poly):
    try:
        import open3d as o3d
    except Exception:
        print("open3d not installed - skipping 3D view (pip install open3d to enable)")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    # polygon as line set
    coords = np.array(list(poly.exterior.coords))
    # make small lines for polygon edges in XY plane
    lines = []
    colors = []
    for i in range(len(coords)-1):
        lines.append([i, i+1])
        colors.append([1,0,0])
    # create line set
    pts3 = np.hstack([coords, np.zeros((coords.shape[0],1))])  # z=0
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts3), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, line_set])

def main():
    print("Generating realistic room with furniture and obstacles...")
    
    # Generate realistic room with furniture
    poly = make_realistic_room(
        width=6.0,
        height=5.0,
        origin=(0.0, 0.0),
        seed=42,
        furniture_density=0.15,
        add_alcoves=True,
        add_wall_indentations=True,
        wall_irregularity=0.05
    )

    print(f"Room area: {poly.area:.2f} m²")
    print(f"Number of furniture pieces (holes): {len(poly.interiors)}")
    
    pts = sample_floor_points(poly, n_points=5000, seed=123, z_noise=0.005)

    # visualize polygon + points
    fig1, ax1 = plot_polygon_and_points(
        poly, 
        pts[:, :3], 
        title=f"Realistic Room (Area: {poly.area:.2f} m², Furniture: {len(poly.interiors)} pieces)"
    )
    fig1.savefig(os.path.join(OUTDIR, "realistic_room_points.png"), dpi=150, bbox_inches='tight')

    # occupancy raster
    occ, bbox = polygon_to_occupancy_image(poly, resolution=0.03, margin=0.2)
    fig2, ax2 = show_occupancy_and_overlay(occ, bbox, poly, title="Occupancy raster (realistic room)")
    fig2.savefig(os.path.join(OUTDIR, "realistic_room_occupancy.png"), dpi=150, bbox_inches='tight')

    print(f"Saved demo images to {OUTDIR}/")
    print("  - realistic_room_points.png")
    print("  - realistic_room_occupancy.png")
    
    # optional interactive 3D view if open3d is available
    try_open3d_view(pts, poly)
    plt.show()

if __name__ == "__main__":
    main()
