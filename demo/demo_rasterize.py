# demo/demo_rasterize.py
"""
Demo for rasterization pipeline.

Run:
    PYTHONPATH=. python demo/demo_rasterize.py

Outputs (demo_out/):
 - poly_points.png       : polygon outline + sampled points
 - occ_raw.png           : raw occupancy image from sampled points
 - occ_clean.png         : postprocessed occupancy image
 - poly_reconstructed.geojson : polygon extracted from occupancy (GeoJSON)
 - overlay.png           : overlay of reconstructed polygon on original polygon
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping

# Import your modules (adjust if package name differs)
from src.synthetic_generator import make_realistic_room, sample_floor_points, polygon_to_occupancy_image
from src.rasterize import points_to_occupancy, postprocess_occupancy, occupancy_to_polygon

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

def save_geojson(poly, path):
    if poly is None:
        return
    with open(path, "w") as f:
        json.dump(mapping(poly), f)

def plot_poly_and_points(poly, pts, outpath):
    fig, ax = plt.subplots(figsize=(6,6))
    x,y = poly.exterior.xy
    ax.plot(x,y, linewidth=2, color='tab:blue', label='original polygon')
    for interior in poly.interiors:
        xi, yi = zip(*interior.coords)
        ax.fill(xi, yi, color='saddlebrown')
    ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.6, label='sampled points')
    ax.set_aspect('equal')
    ax.set_title("Original polygon + sampled floor points")
    ax.legend()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def img_save(img, path):
    plt.imsave(path, img, cmap='gray', vmin=0, vmax=1)

def main():
    # generate a realistic room
    poly = make_realistic_room(6.0, 4.0, seed=42, furniture_density=0.15,
                               add_alcoves=True, add_wall_indentations=True,
                               wall_irregularity=0.08)
    print("Generated polygon valid:", poly.is_valid, "area (m^2):", poly.area, "holes:", len(poly.interiors))

    # sample points inside polygon
    pts = sample_floor_points(poly, n_points=4000, seed=42, z_noise=0.005)
    print("Sampled points:", pts.shape)

    # visualize polygon + points
    plot_poly_and_points(poly, pts, os.path.join(OUT, "poly_points.png"))

    # Rasterize from points (simulate device floor points)
    occ_raw, bbox, res = points_to_occupancy(pts[:, :2], resolution=0.03, margin=0.2)
    print("Raw occupancy shape:", occ_raw.shape, "occupied px:", int(occ_raw.sum()))
    img_save(occ_raw, os.path.join(OUT, "occ_raw.png"))

    # Postprocess occupancy (close gaps & fill holes)
    occ_clean = postprocess_occupancy(occ_raw, closing_iters=3, min_component_area_px=50)
    print("Clean occupancy occupied px:", int(occ_clean.sum()))
    img_save(occ_clean, os.path.join(OUT, "occ_clean.png"))

    # Convert occupancy -> polygon
    poly_rec = occupancy_to_polygon(occ_clean, bbox, res)
    if poly_rec is None:
        print("No polygon reconstructed from occupancy.")
    else:
        print("Reconstructed polygon valid:", poly_rec.is_valid, "area (m^2):", poly_rec.area)
        save_geojson(poly_rec, os.path.join(OUT, "poly_reconstructed.geojson"))

    # Overlay reconstructed polygon on original polygon visualization
    fig, ax = plt.subplots(figsize=(6,6))
    # plot original polygon
    x,y = poly.exterior.xy
    ax.plot(x,y, linewidth=2, color='tab:blue', label='original poly')
    # plot reconstructed if exists
    if poly_rec is not None:
        xr, yr = poly_rec.exterior.xy
        ax.plot(xr, yr, linewidth=2, color='tab:red', label='reconstructed poly')
        ax.fill(xr, yr, color='tab:red', alpha=0.2)
    # fill furniture holes from original polygon
    for interior in poly.interiors:
        xi, yi = zip(*interior.coords)
        ax.fill(xi, yi, color='saddlebrown')
    ax.set_aspect('equal')
    ax.set_title("Original (blue) vs Reconstructed (red)")
    ax.legend()
    fig.savefig(os.path.join(OUT, "overlay.png"), dpi=150)
    plt.close(fig)

    # numeric comparison
    if poly_rec is not None:
        intersection = poly.intersection(poly_rec)
        i_area = intersection.area if not intersection.is_empty else 0.0
        print("Intersection area (m^2):", i_area)
        print("Original area:", poly.area, "Reconstructed area:", poly_rec.area)
        print("Reconstruction IoU (approx):", i_area / (poly.area + poly_rec.area - i_area + 1e-9))

    print("Demo outputs saved to", OUT)

if __name__ == "__main__":
    main()
