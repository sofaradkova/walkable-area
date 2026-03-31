# demo/demo_multi_room_tests.py
"""
Run 5 different test cases with various room sizes and shapes to test alignment algorithms.

Run:
    PYTHONPATH=. python demo/demo_multi_room_tests.py
"""
import os, json, time
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping, MultiPolygon

from src.thumbnail_features import (
    compute_polygon_intersection_metrics, verify_intersection_sufficient
)
from src.polygon_ops import (
    apply_affine_to_polygon, find_best_alignment_by_rotation,
    pca_candidate_rotations, largest_walkable_subpolygon,
)
from src.mesh_processing import compute_walkable_polygon

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_PHASE_ORDER = [
    "load_room_a",
    "load_room_b",
    "rotation_alignment_search",
    "verify_save_export",
    "plot_visualization",
]


@contextmanager
def _timed_add(timings, key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t0)


def _print_phase_breakdown(timings, test_num, test_name):
    wall = timings.get("_wall_clock")
    items = []
    for k in _PHASE_ORDER:
        if k in timings and timings[k] > 0:
            items.append((k, timings[k]))
    for k, v in timings.items():
        if k.startswith("_"):
            continue
        if k not in _PHASE_ORDER and v > 0:
            items.append((k, v))
    if not items and wall is None:
        return
    total = sum(v for _, v in items)
    print(f"\n--- Phase timing (test {test_num}: {test_name}) ---")
    for label, sec in sorted(items, key=lambda x: -x[1]):
        pct = 100.0 * sec / total if total > 0 else 0
        print(f"  {label:28s}  {sec:10.4f}s  ({pct:5.1f}% of tracked)")
    if wall is not None:
        print(f"  {'wall_clock (full test)':28s}  {wall:10.4f}s")
    if items:
        print(f"  {'tracked sum':28s}  {total:10.4f}s")
    if wall is not None and total > 0 and abs(wall - total) > 0.05:
        print(f"  {'gap (untimed / overlap)':28s}  {wall - total:10.4f}s")
    print("---")


def _poly_area_and_holes(poly):
    """
    Return (area, num_holes) for Polygon or MultiPolygon.
    """
    if poly is None:
        return 0.0, 0
    if isinstance(poly, MultiPolygon):
        area = sum(g.area for g in poly.geoms)
        holes = sum(len(g.interiors) for g in poly.geoms)
        return area, holes
    # Simple Polygon
    return poly.area, len(poly.interiors)

def save_geojson(poly, path):
    if poly is None: return
    with open(path, "w") as f:
        json.dump(mapping(poly), f)

def invert_affine_transform(affine_params):
    """
    Invert a 2D affine transform.
    
    Affine format: [a, b, d, e, xoff, yoff] where:
        x' = a*x + b*y + xoff
        y' = d*x + e*y + yoff
    
    Returns inverse transform [a', b', d', e', xoff', yoff'].
    """
    if affine_params is None:
        return None
    
    a, b, d, e, xoff, yoff = affine_params
    
    # Compute determinant of the 2x2 rotation/scaling matrix
    det = a * e - b * d
    
    if abs(det) < 1e-10:
        # Singular matrix, cannot invert
        return None
    
    # Inverse of [a b; d e] is (1/det) * [e -b; -d a]
    a_inv = e / det
    b_inv = -b / det
    d_inv = -d / det
    e_inv = a / det
    
    # Inverse translation: -inv_matrix * [xoff; yoff]
    xoff_inv = -(a_inv * xoff + b_inv * yoff)
    yoff_inv = -(d_inv * xoff + e_inv * yoff)
    
    return [a_inv, b_inv, d_inv, e_inv, xoff_inv, yoff_inv]

def plot_test_result(poly_a, poly_b, poly_b_aligned, inter, metrics, test_name, outpath,
                     verification=None, final_affine=None):
    """Plot test result using polygon-based visualizations only."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        3,
        4,
        hspace=0.35,
        wspace=0.35,
        width_ratios=[1.2, 1.2, 1.2, 2.4],
        height_ratios=[1, 1, 1],
    )

    ax_before_a = fig.add_subplot(gs[0, 0])  # Room A before (separate)
    ax_before_b = fig.add_subplot(gs[0, 1])  # Room B before (separate)
    ax_inter_a = fig.add_subplot(gs[1, 0])  # Intersection in A's coordinate system
    ax_inter_b = fig.add_subplot(gs[1, 1])  # Intersection in B's coordinate system
    ax_features = fig.add_subplot(gs[0:2, 2])  # Alignment details
    ax_after = fig.add_subplot(gs[:, 3])  # After alignment (largest panel)

    # 1. Alignment info panel
    info_lines = ['Alignment method: PCA + rotation search', '(walkable intersection scoring)']
    if metrics:
        info_lines += [
            '',
            f"IoU: {metrics['iou']:.4f}",
            f"Intersection: {metrics['intersection_area']:.2f} m²",
            f"Overlap A: {metrics['overlap_pct_a']:.1f}%",
            f"Overlap B: {metrics['overlap_pct_b']:.1f}%",
        ]
    if verification:
        status = 'VERIFIED' if verification['verified'] else 'NOT VERIFIED'
        info_lines += ['', f"Status: {status}", verification.get('reason', '')]
    ax_features.text(0.05, 0.95, '\n'.join(info_lines),
                     ha='left', va='top', transform=ax_features.transAxes,
                     fontsize=9, family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_features.set_title('Alignment Info')
    ax_features.axis('off')
    
    # Helper function to plot polygon
    def plot_polygon(ax, poly, color, linestyle='-', linewidth=2, alpha=0.3, label=None, fill=True):
        if poly is None or poly.is_empty:
            return
        hole_color = ax.get_facecolor()
        if isinstance(poly, MultiPolygon):
            for i, geom in enumerate(poly.geoms):
                if hasattr(geom, 'exterior') and geom.exterior is not None:
                    x, y = geom.exterior.xy
                    plot_label = label if i == 0 else None
                    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=plot_label)
                    if fill:
                        ax.fill(x, y, color=color, alpha=alpha)
                    for interior in geom.interiors:
                        xi, yi = zip(*interior.coords)
                        ax.fill(xi, yi, color=hole_color, alpha=1.0, zorder=3)
                        ax.plot(xi, yi, color=color, linestyle=':', linewidth=1.0, alpha=0.8, zorder=4)
        elif hasattr(poly, 'exterior') and poly.exterior is not None:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if fill:
                ax.fill(x, y, color=color, alpha=alpha)
            for interior in poly.interiors:
                xi, yi = zip(*interior.coords)
                ax.fill(xi, yi, color=hole_color, alpha=1.0, zorder=3)
                ax.plot(xi, yi, color=color, linestyle=':', linewidth=1.0, alpha=0.8, zorder=4)
    
    # 2. Before alignment - show rooms separately
    plot_polygon(ax_before_a, poly_a, 'tab:blue', label=None, alpha=0.3)
    ax_before_a.set_title('Room A (Before Alignment)')
    ax_before_a.set_aspect('equal')
    ax_before_a.grid(True, alpha=0.3)
    ax_before_a.set_xlabel('X (m)')
    ax_before_a.set_ylabel('Y (m)')
    
    plot_polygon(ax_before_b, poly_b, 'tab:green', label=None, alpha=0.3)
    ax_before_b.set_title('Room B (Before Alignment)')
    ax_before_b.set_aspect('equal')
    ax_before_b.grid(True, alpha=0.3)
    ax_before_b.set_xlabel('X (m)')
    ax_before_b.set_ylabel('Y (m)')
    
    # 3. Intersection in each coordinate system (separate panels)
    # Intersection in Room A's coordinate system
    plot_polygon(ax_inter_a, poly_a, 'tab:blue', label=None, alpha=0.2)
    if inter and not inter.is_empty:
        plot_polygon(ax_inter_a, inter, 'purple', label=None, 
                    alpha=0.7, linestyle='-', linewidth=2)
    ax_inter_a.set_title('Intersection in A Coordinate System')
    ax_inter_a.set_aspect('equal')
    ax_inter_a.grid(True, alpha=0.3)
    ax_inter_a.set_xlabel('X (m)')
    ax_inter_a.set_ylabel('Y (m)')
    
    # Intersection in Room B's original coordinate system
    plot_polygon(ax_inter_b, poly_b, 'tab:green', label=None, alpha=0.2)
    inter_in_b_coords = None
    if inter and not inter.is_empty and final_affine is not None:
        inv_affine = invert_affine_transform(final_affine)
        if inv_affine is not None:
            from src.polygon_ops import apply_affine_to_polygon
            inter_in_b_coords = apply_affine_to_polygon(inter, inv_affine)
            if inter_in_b_coords and not inter_in_b_coords.is_empty:
                plot_polygon(ax_inter_b, inter_in_b_coords, 'purple', 
                            label=None, 
                            alpha=0.7, linestyle='-', linewidth=2)
    ax_inter_b.set_title('Intersection in B Coordinate System')
    ax_inter_b.set_aspect('equal')
    ax_inter_b.grid(True, alpha=0.3)
    ax_inter_b.set_xlabel('X (m)')
    ax_inter_b.set_ylabel('Y (m)')
    
    # 4. After alignment with intersection
    plot_polygon(ax_after, poly_a, 'tab:blue', label='Room A', alpha=0.2)
    if poly_b_aligned:
        plot_polygon(ax_after, poly_b_aligned, 'tab:green', linestyle='--', 
                    label='Room B (aligned)', alpha=0.2)
    
    # Highlight intersection
    if inter and not inter.is_empty:
        if isinstance(inter, MultiPolygon):
            for i, geom in enumerate(inter.geoms):
                if hasattr(geom, 'exterior') and geom.exterior is not None:
                    xi, yi = geom.exterior.xy
                    label = f'Intersection ({metrics["intersection_area"]:.2f} m²)' if i == 0 else None
                    ax_after.fill(xi, yi, color='purple', alpha=0.6, label=label, 
                            edgecolor='darkviolet', linewidth=2)
        elif hasattr(inter, 'exterior') and inter.exterior is not None:
            xi, yi = inter.exterior.xy
            ax_after.fill(xi, yi, color='purple', alpha=0.6, 
                    label=f'Intersection ({metrics["intersection_area"]:.2f} m²)', 
                    edgecolor='darkviolet', linewidth=2)
    
    
    ax_after.set_title(f'{test_name}\nAfter Alignment (IoU: {metrics["iou"]:.3f})')
    ax_after.set_aspect('equal')
    ax_after.grid(True, alpha=0.3)
    ax_after.legend()
    ax_after.set_xlabel('X (m)')
    ax_after.set_ylabel('Y (m)')
    
    plt.suptitle(test_name, fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def _load_room_polygon(room_cfg):
    """
    Load a room polygon.

    Supports two modes:
      - Real: config has 'mesh' and 'info' → use compute_walkable_polygon
    """
    if "mesh" in room_cfg and "info" in room_cfg:
        mesh_path = os.path.join(DATA_DIR, room_cfg["mesh"])
        info_path = os.path.join(DATA_DIR, room_cfg["info"])
        print(f"Loading real room from mesh={mesh_path}, info={info_path}")
        poly = compute_walkable_polygon(
            mesh_path,
            info_path,
            visualize=False,
            debug=True,
        )
        return poly


def run_single_test(test_config, test_num):
    """Run a single test case."""
    t_wall0 = time.perf_counter()
    timings = {}
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_config['name']}")
    print(f"{'='*60}")
    try:
        with _timed_add(timings, "load_room_a"):
            a_poly = _load_room_polygon(test_config["room_a"])
        with _timed_add(timings, "load_room_b"):
            b_poly = _load_room_polygon(test_config["room_b"])

        area_a, furn_a = _poly_area_and_holes(a_poly)
        area_b, furn_b = _poly_area_and_holes(b_poly)
        print(f"Room A area: {area_a:.2f} m², furniture-like holes: {furn_a}")
        print(f"Room B area: {area_b:.2f} m², furniture-like holes: {furn_b}")

        # Use the mesh-derived walkable polygons directly for alignment and intersection.
        # These already have obstacles removed and are identical regardless of which test
        # number the room appears in.
        poly_a_world = a_poly
        poly_b_world = b_poly

        # PCA-seeded rotation search, scored by largest contiguous walkable intersection
        with _timed_add(timings, "rotation_alignment_search"):
            pca_angles = pca_candidate_rotations(poly_a_world, poly_b_world)
            # Build candidate list: PCA hints (4) + fine grid around each (±20° in 5° steps)
            seen = set()
            rotation_angles = []
            for base in pca_angles:
                for delta in range(-20, 25, 5):
                    a = int(round(base + delta)) % 360
                    if a not in seen:
                        seen.add(a)
                        rotation_angles.append(a)
            # Also include the full 15° coarse grid so nothing obvious is missed
            for a in range(0, 360, 15):
                if a not in seen:
                    seen.add(a)
                    rotation_angles.append(a)

            best_affine, best_walkable_area, best_inter_area = find_best_alignment_by_rotation(
                poly_a_world,
                poly_b_world,
                rotation_angles=rotation_angles,
                use_centroids=True,
                min_passage_width=0.3,
            )

        if best_affine is None:
            print("Failed to find alignment")
            return None

        final_affine = best_affine
        transform_method = "pca_rotation_search"
        poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine)
        metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
        print(
            f"Alignment: PCA + rotation search — walkable area: {best_walkable_area:.2f} m², "
            f"raw intersection: {best_inter_area:.2f} m²"
        )

        if metrics is None:
            print("Failed to compute metrics")
            return None

        # Replace raw intersection with the largest actually-walkable region:
        # morphological opening (erode → largest piece → dilate) removes thin slivers
        # and disconnected fragments that a person could not reach.
        MIN_PASSAGE = 0.3  # metres
        raw_inter = metrics["intersection"]
        walkable_inter = largest_walkable_subpolygon(raw_inter, min_passage_width=MIN_PASSAGE)
        if walkable_inter is not None and not walkable_inter.is_empty:
            walkable_area = walkable_inter.area
            metrics["intersection"] = walkable_inter
            metrics["intersection_area"] = walkable_area
            metrics["iou"] = walkable_area / metrics["union_area"] if metrics["union_area"] > 0 else 0.0
            metrics["overlap_pct_a"] = walkable_area / metrics["area_a"] * 100 if metrics["area_a"] > 0 else 0.0
            metrics["overlap_pct_b"] = walkable_area / metrics["area_b"] * 100 if metrics["area_b"] > 0 else 0.0
            print(f"Walkable intersection area (after opening): {walkable_area:.2f} m²")
        else:
            metrics["intersection"] = None
            metrics["intersection_area"] = 0.0
            metrics["iou"] = 0.0
            metrics["overlap_pct_a"] = 0.0
            metrics["overlap_pct_b"] = 0.0
            print("No walkable intersection found after removing thin passages.")

        with _timed_add(timings, "verify_save_export"):
            verification = verify_intersection_sufficient(
                metrics,
                min_iou=0.1,
                min_intersection_area=0.5,
                min_overlap_pct=5.0,
            )

            inter = metrics["intersection"]
            verified_inter = inter if verification["verified"] else None

            test_base = f"test_{test_num:02d}_{test_config['name'].replace(' ', '_').lower()}"
            outpath = os.path.join(OUT, f"{test_base}.png")

            if verified_inter is not None and not verified_inter.is_empty:
                save_geojson(
                    poly_a_world, os.path.join(OUT, f"{test_base}_room_a.geojson")
                )
                save_geojson(
                    poly_b_world,
                    os.path.join(OUT, f"{test_base}_room_b_original.geojson"),
                )
                save_geojson(
                    poly_b_aligned,
                    os.path.join(OUT, f"{test_base}_room_b_aligned.geojson"),
                )
                save_geojson(
                    verified_inter,
                    os.path.join(OUT, f"{test_base}_intersection_verified.geojson"),
                )

            result_data = {
                "test_name": test_config["name"],
                "test_num": test_num,
                "transform": {
                    "method": transform_method,
                    "affine_matrix": list(final_affine)
                    if final_affine is not None
                    else None,
                    "affine_params": {
                        "a": float(final_affine[0])
                        if final_affine is not None
                        else None,
                        "b": float(final_affine[1])
                        if final_affine is not None
                        else None,
                        "c": float(final_affine[2])
                        if final_affine is not None
                        else None,
                        "d": float(final_affine[3])
                        if final_affine is not None
                        else None,
                        "e": float(final_affine[4])
                        if final_affine is not None
                        else None,
                        "f": float(final_affine[5])
                        if final_affine is not None
                        else None,
                    }
                    if final_affine is not None
                    else None,
                },
                "metrics": {
                    "iou": float(metrics["iou"]),
                    "intersection_area": float(metrics["intersection_area"]),
                    "union_area": float(metrics["union_area"]),
                    "overlap_pct_a": float(metrics["overlap_pct_a"]),
                    "overlap_pct_b": float(metrics["overlap_pct_b"]),
                    "area_a": float(metrics["area_a"]),
                    "area_b": float(metrics["area_b"]),
                },
                "verification": {
                    "verified": verification["verified"],
                    "iou_check": verification["iou_check"],
                    "area_check": verification["area_check"],
                    "overlap_check": verification["overlap_check"],
                    "reason": verification["reason"],
                },
                "verification_thresholds": {
                    "min_iou": 0.1,
                    "min_intersection_area": 0.5,
                    "min_overlap_pct": 5.0,
                },
            }

            json_path = os.path.join(OUT, f"{test_base}_results.json")
            with open(json_path, "w") as f:
                json.dump(result_data, f, indent=2)

        with _timed_add(timings, "plot_visualization"):
            plot_test_result(
                poly_a_world,
                poly_b_world,
                poly_b_aligned,
                inter,
                metrics,
                test_config["name"],
                outpath,
                verification=verification,
                final_affine=final_affine,
            )

        status_icon = "✓ VERIFIED" if verification["verified"] else "✗ NOT VERIFIED"
        print(
            f"{status_icon} Test {test_num} complete: IoU={metrics['iou']:.4f}, "
            f"Intersection={metrics['intersection_area']:.2f} m²"
        )
        if not verification["verified"]:
            print(f"  Verification failed: {verification['reason']}")
        print(f"  Saved visualization to {outpath}")
        print(f"  Saved results JSON to {json_path}")
        if verified_inter is not None:
            print(f"  Saved verified intersection GeoJSON")

        return {
            **metrics,
            "verification": verification,
            "verified_intersection": verified_inter,
        }
    finally:
        timings["_wall_clock"] = time.perf_counter() - t_wall0
        _print_phase_breakdown(timings, test_num, test_config["name"])

def _discover_rooms():
    """Find all mesh+info pairs in DATA_DIR and return list of room configs."""
    rooms = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("mesh_semantic_") and f.endswith(".ply"):
            stem = f.replace("mesh_semantic_", "").replace(".ply", "")
            info_name = f"info_semantic_{stem}.json"
            info_path = os.path.join(DATA_DIR, info_name)
            if os.path.isfile(info_path):
                rooms.append({
                    "name": stem.replace("_", " ").title(),
                    "mesh": f,
                    "info": info_name,
                })
    return rooms


def main():
    rooms = _discover_rooms()
    if len(rooms) < 2:
        print("Need at least 2 rooms (mesh_semantic_*.ply + info_semantic_*.json) in data/")
        return

    # All unordered pairs of rooms
    test_cases = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            ra, rb = rooms[i], rooms[j]
            test_cases.append({
                "name": f"{ra['name']} vs {rb['name']}",
                "room_a": {"mesh": ra["mesh"], "info": ra["info"]},
                "room_b": {"mesh": rb["mesh"], "info": rb["info"]},
            })

    print(f"Running {len(test_cases)} test(s) for all room pairs: {[t['name'] for t in test_cases]}\n")

    results = []
    verified_count = 0
    for idx, test_config in enumerate(test_cases, start=1):
        r = run_single_test(test_config, idx)
        if r is not None:
            results.append(r)
            if r.get("verification", {}).get("verified"):
                verified_count += 1

    print(f"\n{'='*60}")
    print(f"Done: {verified_count}/{len(results)} verified (of {len(test_cases)} tests run)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
