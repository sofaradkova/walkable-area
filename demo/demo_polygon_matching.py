# demo/demo_polygon_matching.py
"""
Demo pipeline: realistic room -> rasterize -> polygon -> simplify/shrink -> sample boundary ->
RANSAC-based geometry alignment -> apply transform -> intersection & IoU -> save visual outputs.

Run:
    PYTHONPATH=. python demo/demo_polygon_matching.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping, MultiPolygon

from src.synthetic_generator import make_realistic_room, sample_floor_points, polygon_to_occupancy_image
from src.rasterize import points_to_occupancy, postprocess_occupancy, occupancy_to_polygon
from src.polygon_ops import (
    simplify_polygon, shrink_polygon, sample_polygon_boundary,
    ransac_similarity_transform, apply_affine_to_polygon
)

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

def save_geojson(poly, path):
    if poly is None: return
    with open(path, "w") as f:
        json.dump(mapping(poly), f)

def plot_polygons(polyA, polyB, polyB_aligned, intersection, outpath):
    fig, ax = plt.subplots(figsize=(7,7))
    if polyA is not None:
        x,y = polyA.exterior.xy
        ax.plot(x,y, color='tab:blue', linewidth=2, label='therapist (A)')
        for interior in polyA.interiors:
            xi, yi = zip(*interior.coords)
            ax.fill(xi, yi, color='saddlebrown')
    if polyB is not None:
        x,y = polyB.exterior.xy
        ax.plot(x,y, color='tab:green', linewidth=2, label='patient raw (B)')
    if polyB_aligned is not None:
        x,y = polyB_aligned.exterior.xy
        ax.plot(x,y, color='tab:red', linestyle='--', linewidth=2, label='patient aligned (B->A)')
        ax.fill(x,y, color='tab:red', alpha=0.2)
    if intersection is not None and not intersection.is_empty:
        # Handle both Polygon and MultiPolygon cases
        if isinstance(intersection, MultiPolygon):
            for i, geom in enumerate(intersection.geoms):
                if hasattr(geom, 'exterior') and geom.exterior is not None:
                    xi, yi = geom.exterior.xy
                    label = 'intersection' if i == 0 else None  # Label only first geometry
                    ax.fill(xi, yi, color='purple', alpha=0.5, label=label)
        elif hasattr(intersection, 'exterior') and intersection.exterior is not None:
            xi, yi = intersection.exterior.xy
            ax.fill(xi, yi, color='purple', alpha=0.5, label='intersection')
    ax.set_aspect('equal')
    ax.set_title('Polygon matching overlay')
    ax.legend()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    # 1) Generate two realistic rooms (therapist A and patient B). We'll create B by transforming A.
    a_poly = make_realistic_room(6.0, 4.0, seed=101, furniture_density=0.12,
                                 add_alcoves=True, add_wall_indentations=True, wall_irregularity=0.06)
    # create B by applying known transform (rotate + translate)
    theta = 18.0  # deg
    tx, ty = 1.0, -0.8
    # sample points from A and then transform points to build B's pointcloud (simulate separate scan)
    pts_a = sample_floor_points(a_poly, n_points=4000, seed=101)
    th = np.deg2rad(theta)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts_b_xy = (pts_a[:, :2] @ R.T) + np.array([tx, ty])
    # optionally remove some points to simulate occlusion
    mask = np.random.default_rng(42).uniform(size=pts_b_xy.shape[0]) > 0.06
    pts_b = np.hstack([pts_b_xy[mask], np.zeros((mask.sum(),1))])

    # 2) Rasterize both point clouds (simulate local processing)
    occ_a, bbox_a, res_a = points_to_occupancy(pts_a[:, :2], resolution=0.03, margin=0.2)
    occ_b, bbox_b, res_b = points_to_occupancy(pts_b[:, :2], resolution=0.03, margin=0.2)

    # 3) Postprocess occupancy -> polygon
    occ_a_clean = postprocess_occupancy(occ_a, closing_iters=3, min_component_area_px=30)
    occ_b_clean = postprocess_occupancy(occ_b, closing_iters=3, min_component_area_px=30)

    poly_a = occupancy_to_polygon(occ_a_clean, bbox_a, res_a)
    poly_b = occupancy_to_polygon(occ_b_clean, bbox_b, res_b)

    print("Poly A valid:", poly_a.is_valid, "area:", poly_a.area if poly_a else None)
    print("Poly B valid:", poly_b.is_valid if poly_b else None, "area:", poly_b.area if poly_b else None)

    save_geojson(poly_a, os.path.join(OUT, "poly_A.geojson"))
    save_geojson(poly_b, os.path.join(OUT, "poly_B.geojson"))

    # 4) Simplify & shrink (safety margin)
    poly_a_s = simplify_polygon(poly_a, tol=0.03)
    poly_b_s = simplify_polygon(poly_b, tol=0.03)
    poly_a_shr = shrink_polygon(poly_a_s, margin=0.12)
    poly_b_shr = shrink_polygon(poly_b_s, margin=0.12)

    save_geojson(poly_a_shr, os.path.join(OUT, "poly_A_shrunk.geojson"))
    save_geojson(poly_b_shr, os.path.join(OUT, "poly_B_shrunk.geojson"))

    # 5) Sample boundaries and run RANSAC similarity (geometry-only)
    ptsA = sample_polygon_boundary(poly_a_shr, n_samples=400)
    ptsB = sample_polygon_boundary(poly_b_shr, n_samples=400)

    affine, inliers = ransac_similarity_transform(ptsB, ptsA, n_iters=800, sample_size=3,
                                                  inlier_thresh=0.18, min_inliers=40, random_seed=1)
    if affine is None:
        print("RANSAC failed to find transform.")
        return
    print("Estimated affine:", affine, "inliers:", len(inliers) if inliers is not None else 0)

    # 6) Apply transform to poly_b_shr -> get aligned polygon in A's frame
    poly_b_aligned = apply_affine_to_polygon(poly_b_shr, affine)

    save_geojson(poly_b_aligned, os.path.join(OUT, "poly_B_aligned.geojson"))

    # 7) Compute intersection & IoU
    inter = poly_a_shr.intersection(poly_b_aligned)
    inter_area = inter.area if inter and not inter.is_empty else 0.0
    union_area = poly_a_shr.union(poly_b_aligned).area if poly_b_aligned is not None else poly_a_shr.area
    iou = inter_area / union_area if union_area > 0 else 0.0
    print("Intersection area:", inter_area, "Union area:", union_area, "IoU:", iou)

    # 8) Visualize overlay
    plot_polygons(poly_a_shr, poly_b_shr, poly_b_aligned, inter, os.path.join(OUT, "matching_overlay.png"))
    print("Saved overlay to demo_out/matching_overlay.png")
    print("All demo outputs saved to", OUT)

if __name__ == "__main__":
    main()

