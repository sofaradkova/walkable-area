# demo/demo_multi_room_tests.py
"""
Run 5 different test cases with various room sizes and shapes to test alignment algorithms.

Run:
    PYTHONPATH=. python demo/demo_multi_room_tests.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import mapping, MultiPolygon

from src.rasterize import (
    points_to_occupancy,
    postprocess_occupancy,
    occupancy_to_polygon,
    sample_floor_points,
)
from src.thumbnail_features import (
    render_thumbnail_from_occupancy, _occ_to_thumbnail_mapping,
    detect_orb_features, match_descriptors_knn_ratio, estimate_transform_from_matches,
    compute_polygon_intersection_metrics, verify_intersection_sufficient
)
from src.polygon_ops import (
    estimate_rigid_umeyama, apply_affine_to_polygon, find_best_alignment_by_rotation
)
from src.mesh_processing import compute_walkable_polygon

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


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
                     occ_a=None, occ_b=None, bbox_a=None, bbox_b=None, 
                     thumbA=None, thumbB=None, matches=None, ptsA_uv=None, ptsB_uv=None, inlier_mask=None,
                     verification=None, final_affine=None):
    """Plot comprehensive test result with occupancy images and feature visualization."""
    fig = plt.figure(figsize=(24, 12))
    # Grid: 4 rows, 6 cols - alignment panel gets 2 cols x all 4 rows (largest)
    gs = fig.add_gridspec(4, 6, hspace=0.35, wspace=0.3, 
                         width_ratios=[1, 1, 1, 1, 2, 2],
                         height_ratios=[1, 1, 1, 1])
    
    # Create subplots - smaller panels on left, large alignment on right
    ax_occ_a = fig.add_subplot(gs[0, 0])  # Room A occupancy
    ax_occ_b = fig.add_subplot(gs[0, 1])  # Room B occupancy
    ax_thumb_a = fig.add_subplot(gs[0, 2])  # Room A thumbnail
    ax_thumb_b = fig.add_subplot(gs[0, 3])  # Room B thumbnail
    ax_before_a = fig.add_subplot(gs[1, 0])  # Room A before (separate)
    ax_before_b = fig.add_subplot(gs[1, 1])  # Room B before (separate)
    ax_inter_a = fig.add_subplot(gs[2, 0])  # Intersection in A's coordinate system
    ax_inter_b = fig.add_subplot(gs[2, 1])  # Intersection in B's coordinate system
    ax_features = fig.add_subplot(gs[1:3, 2:4])  # Feature matches (spans 2 rows, 2 cols)
    ax_after = fig.add_subplot(gs[0:, 4:])  # After alignment (LARGE - spans all rows, 2 rightmost cols)
    
    # 1. Occupancy images
    if occ_a is not None and bbox_a is not None:
        minx, miny, maxx, maxy = bbox_a
        ax_occ_a.imshow(occ_a, cmap='gray', origin='upper', 
                       extent=(minx, maxx, miny, maxy), aspect='equal')
        ax_occ_a.set_title('Room A Occupancy')
        ax_occ_a.set_xlabel('X (m)')
        ax_occ_a.set_ylabel('Y (m)')
        ax_occ_a.grid(True, alpha=0.3)
    
    if occ_b is not None and bbox_b is not None:
        minx, miny, maxx, maxy = bbox_b
        ax_occ_b.imshow(occ_b, cmap='gray', origin='upper', 
                       extent=(minx, maxx, miny, maxy), aspect='equal')
        ax_occ_b.set_title('Room B Occupancy')
        ax_occ_b.set_xlabel('X (m)')
        ax_occ_b.set_ylabel('Y (m)')
        ax_occ_b.grid(True, alpha=0.3)
    
    # 2. Thumbnail images
    if thumbA is not None:
        ax_thumb_a.imshow(thumbA, cmap='gray', origin='upper')
        ax_thumb_a.set_title('Room A Thumbnail')
        ax_thumb_a.axis('off')
        if ptsA_uv is not None and len(ptsA_uv) > 0:
            ax_thumb_a.scatter(ptsA_uv[:, 0], ptsA_uv[:, 1], c='red', s=5, alpha=0.5)
            ax_thumb_a.text(5, 10, f'{len(ptsA_uv)} keypoints', 
                          color='red', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    if thumbB is not None:
        ax_thumb_b.imshow(thumbB, cmap='gray', origin='upper')
        ax_thumb_b.set_title('Room B Thumbnail')
        ax_thumb_b.axis('off')
        if ptsB_uv is not None and len(ptsB_uv) > 0:
            ax_thumb_b.scatter(ptsB_uv[:, 0], ptsB_uv[:, 1], c='blue', s=5, alpha=0.5)
            ax_thumb_b.text(5, 10, f'{len(ptsB_uv)} keypoints', 
                          color='blue', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 3. Feature matches visualization
    if thumbA is not None and thumbB is not None and matches is not None and len(matches) > 0:
        try:
            import cv2
            # Convert to cv2.KeyPoint format
            cv_kpsA = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in ptsA_uv] if ptsA_uv is not None else []
            cv_kpsB = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in ptsB_uv] if ptsB_uv is not None else []
            
            if len(cv_kpsA) > 0 and len(cv_kpsB) > 0:
                draw_img = cv2.drawMatches(thumbA, cv_kpsA, thumbB, cv_kpsB, matches, None,
                                         matchColor=(0,255,0), singlePointColor=(255,0,0),
                                         matchesMask=(inlier_mask.astype(int).tolist() if inlier_mask is not None else None))
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                ax_features.imshow(draw_img)
                ax_features.set_title(f'Feature Matches ({len(matches)} total, {int(inlier_mask.sum()) if inlier_mask is not None else len(matches)} inliers)')
                ax_features.axis('off')
        except Exception as e:
            ax_features.text(0.5, 0.5, f'Could not draw matches:\n{str(e)}', 
                           ha='center', va='center', transform=ax_features.transAxes)
            ax_features.axis('off')
    else:
        ax_features.text(0.5, 0.5, 'No feature matches available', 
                        ha='center', va='center', transform=ax_features.transAxes)
        ax_features.axis('off')
    
    # Helper function to plot polygon
    def plot_polygon(ax, poly, color, linestyle='-', linewidth=2, alpha=0.3, label=None, fill=True):
        if poly is None or poly.is_empty:
            return
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
                        ax.fill(xi, yi, color='gray', alpha=0.2)
        elif hasattr(poly, 'exterior') and poly.exterior is not None:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if fill:
                ax.fill(x, y, color=color, alpha=alpha)
            for interior in poly.interiors:
                xi, yi = zip(*interior.coords)
                ax.fill(xi, yi, color='gray', alpha=0.2)
    
    # 4. Before alignment - show rooms separately
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
    
    # 5. Intersection in each coordinate system (separate panels)
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
    
    # 6. After alignment with intersection
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
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_config['name']}")
    print(f"{'='*60}")
    
    # Load Room A
    a_poly = _load_room_polygon(test_config['room_a'])
    
    # Load Room B
    b_poly = _load_room_polygon(test_config['room_b'])
    
    area_a, furn_a = _poly_area_and_holes(a_poly)
    area_b, furn_b = _poly_area_and_holes(b_poly)
    print(f"Room A area: {area_a:.2f} m², furniture-like holes: {furn_a}")
    print(f"Room B area: {area_b:.2f} m², furniture-like holes: {furn_b}")
    
    # Sample points (use configured seeds if present, otherwise deterministic defaults)
    seed_a = test_config['room_a'].get('seed', 100 + test_num)
    seed_b = test_config['room_b'].get('seed', 200 + test_num)
    pts_a = sample_floor_points(a_poly, n_points=4000, seed=seed_a)
    pts_b = sample_floor_points(b_poly, n_points=3500, seed=seed_b)
    
    # Rasterize
    occ_a, bbox_a, res_a = points_to_occupancy(pts_a[:, :2], resolution=0.03, margin=0.2)
    occ_b, bbox_b, res_b = points_to_occupancy(pts_b[:, :2], resolution=0.03, margin=0.2)
    
    # Postprocess
    occ_a_clean = postprocess_occupancy(occ_a, closing_iters=3, min_component_area_px=30)
    occ_b_clean = postprocess_occupancy(occ_b, closing_iters=3, min_component_area_px=30)
    
    # Convert to polygons
    poly_a_world = occupancy_to_polygon(occ_a_clean, bbox_a, res_a)
    poly_b_world = occupancy_to_polygon(occ_b_clean, bbox_b, res_b)
    
    if poly_a_world is None or poly_b_world is None:
        print(f"ERROR: Failed to create polygons for test {test_num}")
        return None
    
    # Generate thumbnails and detect features
    thumbA = render_thumbnail_from_occupancy(occ_a_clean, bbox_a, out_size=256)
    thumbB = render_thumbnail_from_occupancy(occ_b_clean, bbox_b, out_size=256)
    
    kpsA, ptsA_uv, descA = detect_orb_features(thumbA, n_features=600)
    kpsB, ptsB_uv, descB = detect_orb_features(thumbB, n_features=600)
    
    print(f"Detected keypoints: {len(kpsA)} {len(kpsB)}")
    
    # Match features
    matches = match_descriptors_knn_ratio(descA, descB, ratio=0.85)
    print(f"Matches: {len(matches)}")
    
    # Variables for visualization (initialize)
    matches_for_viz = matches if matches else []
    inlier_mask_for_viz = None
    final_affine = None  # Track the final transform used
    transform_method = None  # Track which method was used
    
    if len(matches) < 6:
        print("Too few matches, trying rotation search only...")
        best_affine, best_iou, best_inter = find_best_alignment_by_rotation(
            poly_a_world, poly_b_world, rotation_angles=list(range(0, 360, 15)), use_centroids=True
        )
        if best_affine is None:
            print("Failed to find alignment")
            return None
        final_affine = best_affine
        transform_method = "rotation_search"
        poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine)
        metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
    else:
        # Estimate transform from matches
        result = estimate_transform_from_matches(ptsA_uv, ptsB_uv, matches, ransac_thresh=4.0)
        if result is None or result[0] is None:
            print("Feature matching failed, trying rotation search only...")
            best_affine, best_iou, best_inter = find_best_alignment_by_rotation(
                poly_a_world, poly_b_world, rotation_angles=list(range(0, 360, 15)), use_centroids=True
            )
            if best_affine is None:
                print("Failed to find alignment")
                return None
            final_affine = best_affine
            transform_method = "rotation_search_fallback"
            poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine)
            metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
        else:
            M, inlier_mask = result
            inlier_mask_for_viz = inlier_mask  # Store for visualization
            
            # Map thumbnail pixels to world coordinates
            _, _, thumbA_to_world, _ = _occ_to_thumbnail_mapping(bbox_a, occ_a_clean.shape, 256)
            _, _, thumbB_to_world, _ = _occ_to_thumbnail_mapping(bbox_b, occ_b_clean.shape, 256)
            
            src_uv = np.array([ptsA_uv[m.queryIdx] for i,m in enumerate(matches) 
                              if inlier_mask is None or inlier_mask[i]], dtype=float)
            dst_uv = np.array([ptsB_uv[m.trainIdx] for i,m in enumerate(matches) 
                              if inlier_mask is None or inlier_mask[i]], dtype=float)
            
            if len(src_uv) < 4:
                print("Not enough inlier matches, using rotation search...")
                best_affine, best_iou, best_inter = find_best_alignment_by_rotation(
                    poly_a_world, poly_b_world, rotation_angles=list(range(0, 360, 15)), use_centroids=True
                )
                final_affine = best_affine
                transform_method = "rotation_search_insufficient_inliers"
                poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine)
                metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
            else:
                src_world = np.array([thumbA_to_world(u,v) for (u,v) in src_uv])
                dst_world = np.array([thumbB_to_world(u,v) for (u,v) in dst_uv])
                
                # Estimate rigid transform
                world_affine_feature = estimate_rigid_umeyama(dst_world, src_world)
                
                # Try rotation search and compare
                best_affine_rotation, best_iou_rotation, best_inter_rotation = find_best_alignment_by_rotation(
                    poly_a_world, poly_b_world, 
                    rotation_angles=list(range(0, 360, 15)), 
                    use_centroids=True
                )
                
                # Compare and use best
                poly_b_feature = apply_affine_to_polygon(poly_b_world, world_affine_feature)
                if poly_b_feature and not poly_b_feature.is_empty:
                    metrics_feature = compute_polygon_intersection_metrics(poly_a_world, poly_b_feature)
                    if best_iou_rotation > metrics_feature['iou']:
                        print(f"Using rotation search (IoU: {best_iou_rotation:.4f} > {metrics_feature['iou']:.4f})")
                        final_affine = best_affine_rotation
                        transform_method = "rotation_search_comparison"
                        poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine_rotation)
                        metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
                    else:
                        print(f"Using feature-based (IoU: {metrics_feature['iou']:.4f} >= {best_iou_rotation:.4f})")
                        final_affine = world_affine_feature
                        transform_method = "feature_based_umeyama"
                        poly_b_aligned = poly_b_feature
                        metrics = metrics_feature
                else:
                    final_affine = best_affine_rotation
                    transform_method = "rotation_search_feature_failed"
                    poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine_rotation)
                    metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
    
    if metrics is None:
        print("Failed to compute metrics")
        return None
    
    # VERIFICATION STEP: Verify that the area is sufficient (per flowchart)
    verification = verify_intersection_sufficient(
        metrics,
        min_iou=0.1,  # Minimum 10% IoU
        min_intersection_area=0.5,  # Minimum 0.5 m² intersection
        min_overlap_pct=5.0  # Minimum 5% overlap of either polygon
    )
    
    inter = metrics['intersection']
    verified_inter = inter if verification['verified'] else None
    
    # STORAGE: Save verified results (per flowchart - "Verified intersection polygon")
    test_base = f"test_{test_num:02d}_{test_config['name'].replace(' ', '_').lower()}"
    outpath = os.path.join(OUT, f"{test_base}.png")
    
    # Save GeoJSON files for verified intersection
    if verified_inter is not None and not verified_inter.is_empty:
        save_geojson(poly_a_world, os.path.join(OUT, f"{test_base}_room_a.geojson"))
        save_geojson(poly_b_world, os.path.join(OUT, f"{test_base}_room_b_original.geojson"))
        save_geojson(poly_b_aligned, os.path.join(OUT, f"{test_base}_room_b_aligned.geojson"))
        save_geojson(verified_inter, os.path.join(OUT, f"{test_base}_intersection_verified.geojson"))
    
    # Save metrics and verification results as JSON (including transform matrix)
    result_data = {
        'test_name': test_config['name'],
        'test_num': test_num,
        'transform': {
            'method': transform_method,
            'affine_matrix': list(final_affine) if final_affine is not None else None,
            'affine_params': {
                'a': float(final_affine[0]) if final_affine is not None else None,
                'b': float(final_affine[1]) if final_affine is not None else None,
                'c': float(final_affine[2]) if final_affine is not None else None,
                'd': float(final_affine[3]) if final_affine is not None else None,
                'e': float(final_affine[4]) if final_affine is not None else None,
                'f': float(final_affine[5]) if final_affine is not None else None
            } if final_affine is not None else None
        },
        'metrics': {
            'iou': float(metrics['iou']),
            'intersection_area': float(metrics['intersection_area']),
            'union_area': float(metrics['union_area']),
            'overlap_pct_a': float(metrics['overlap_pct_a']),
            'overlap_pct_b': float(metrics['overlap_pct_b']),
            'area_a': float(metrics['area_a']),
            'area_b': float(metrics['area_b'])
        },
        'verification': {
            'verified': verification['verified'],
            'iou_check': verification['iou_check'],
            'area_check': verification['area_check'],
            'overlap_check': verification['overlap_check'],
            'reason': verification['reason']
        },
        'verification_thresholds': {
            'min_iou': 0.1,
            'min_intersection_area': 0.5,
            'min_overlap_pct': 5.0
        }
    }
    
    json_path = os.path.join(OUT, f"{test_base}_results.json")
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Visualize with occupancy images and features (include verification status)
    plot_test_result(poly_a_world, poly_b_world, poly_b_aligned, inter, metrics, 
                    test_config['name'], outpath,
                    occ_a=occ_a_clean, occ_b=occ_b_clean,
                    bbox_a=bbox_a, bbox_b=bbox_b,
                    thumbA=thumbA, thumbB=thumbB,
                    matches=matches_for_viz,
                    ptsA_uv=ptsA_uv, ptsB_uv=ptsB_uv,
                    inlier_mask=inlier_mask_for_viz,
                    verification=verification,
                    final_affine=final_affine)
    
    # Print verification status
    status_icon = "✓ VERIFIED" if verification['verified'] else "✗ NOT VERIFIED"
    print(f"{status_icon} Test {test_num} complete: IoU={metrics['iou']:.4f}, "
          f"Intersection={metrics['intersection_area']:.2f} m²")
    if not verification['verified']:
        print(f"  Verification failed: {verification['reason']}")
    print(f"  Saved visualization to {outpath}")
    print(f"  Saved results JSON to {json_path}")
    if verified_inter is not None:
        print(f"  Saved verified intersection GeoJSON")
    
    # Return metrics with verification status
    return {
        **metrics,
        'verification': verification,
        'verified_intersection': verified_inter
    }

def main():
    # Define tests between real rooms (from data/)
    test_cases = [
        {
            'name': 'Room0 vs Room1',
            'room_a': {
                'mesh': 'mesh_semantic_room0.ply',
                'info': 'info_semantic_room0.json',
            },
            'room_b': {
                'mesh': 'mesh_semantic_room1.ply',
                'info': 'info_semantic_room1.json',
            }
        },
        {
            'name': 'Room0 vs Office4',
            'room_a': {
                'mesh': 'mesh_semantic_room0.ply',
                'info': 'info_semantic_room0.json',
            },
            'room_b': {
                'mesh': 'mesh_semantic_office4.ply',
                'info': 'info_semantic_office4.json',
            }
        },
        {
            'name': 'Room1 vs Office4',
            'room_a': {
                'mesh': 'mesh_semantic_room1.ply',
                'info': 'info_semantic_room1.json',
            },
            'room_b': {
                'mesh': 'mesh_semantic_office4.ply',
                'info': 'info_semantic_office4.json',
            }
        }
    ]
    
    results = []
    verified_count = 0

    # TEMP: only run first test (Room0 vs Room1) for debugging
    run_single_test(test_cases[0], 1)

if __name__ == "__main__":
    main()
