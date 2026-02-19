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

from src.synthetic_generator import make_realistic_room, sample_floor_points
from src.rasterize import points_to_occupancy, postprocess_occupancy, occupancy_to_polygon
from src.thumbnail_features import (
    render_thumbnail_from_occupancy, _occ_to_thumbnail_mapping,
    detect_orb_features, match_descriptors_knn_ratio, estimate_transform_from_matches,
    compute_polygon_intersection_metrics
)
from src.polygon_ops import (
    estimate_rigid_umeyama, apply_affine_to_polygon, find_best_alignment_by_rotation
)

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

def save_geojson(poly, path):
    if poly is None: return
    with open(path, "w") as f:
        json.dump(mapping(poly), f)

def plot_test_result(poly_a, poly_b, poly_b_aligned, inter, metrics, test_name, outpath,
                     occ_a=None, occ_b=None, bbox_a=None, bbox_b=None, 
                     thumbA=None, thumbB=None, matches=None, ptsA_uv=None, ptsB_uv=None, inlier_mask=None):
    """Plot comprehensive test result with occupancy images and feature visualization."""
    fig = plt.figure(figsize=(22, 10))
    # Grid: 3 rows, 5 cols - alignment panel gets 2 cols x all 3 rows (largest)
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.3, 
                         width_ratios=[1, 1, 1, 2, 2],
                         height_ratios=[1, 1, 1])
    
    # Create subplots - smaller panels on left, large alignment on right
    ax_occ_a = fig.add_subplot(gs[0, 0])  # Room A occupancy
    ax_occ_b = fig.add_subplot(gs[0, 1])  # Room B occupancy
    ax_thumb_a = fig.add_subplot(gs[0, 2])  # Room A thumbnail
    ax_before_a = fig.add_subplot(gs[1, 0])  # Room A before (separate)
    ax_before_b = fig.add_subplot(gs[1, 1])  # Room B before (separate)
    ax_thumb_b = fig.add_subplot(gs[1, 2])  # Room B thumbnail
    ax_features = fig.add_subplot(gs[2, 0:3])  # Feature matches (smaller, spans 3 cols in bottom row)
    ax_after = fig.add_subplot(gs[0:, 3:])  # After alignment (LARGE - spans all rows, 2 rightmost cols)
    
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
    plot_polygon(ax_before_a, poly_a, 'tab:blue', label='Room A', alpha=0.3)
    ax_before_a.set_title('Room A (Before Alignment)')
    ax_before_a.set_aspect('equal')
    ax_before_a.grid(True, alpha=0.3)
    ax_before_a.legend()
    ax_before_a.set_xlabel('X (m)')
    ax_before_a.set_ylabel('Y (m)')
    
    plot_polygon(ax_before_b, poly_b, 'tab:green', label='Room B (original)', alpha=0.3)
    ax_before_b.set_title('Room B (Before Alignment)')
    ax_before_b.set_aspect('equal')
    ax_before_b.grid(True, alpha=0.3)
    ax_before_b.legend()
    ax_before_b.set_xlabel('X (m)')
    ax_before_b.set_ylabel('Y (m)')
    
    # 5. After alignment with intersection
    plot_polygon(ax_after, poly_a, 'tab:blue', label='Room A', alpha=0.2)
    if poly_b_aligned:
        plot_polygon(ax_after, poly_b_aligned, 'tab:red', linestyle='--', 
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
    
    # Add metrics text
    metrics_text = (f'IoU: {metrics["iou"]:.3f}\n'
                   f'Overlap: {metrics["intersection_area"]:.2f} m²\n'
                   f'Room A: {metrics["area_a"]:.2f} m²\n'
                   f'Room B: {metrics["area_b"]:.2f} m²')
    ax_after.text(0.02, 0.98, metrics_text, transform=ax_after.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_after.set_title(f'{test_name}\nAfter Alignment (IoU: {metrics["iou"]:.3f})')
    ax_after.set_aspect('equal')
    ax_after.grid(True, alpha=0.3)
    ax_after.legend()
    ax_after.set_xlabel('X (m)')
    ax_after.set_ylabel('Y (m)')
    
    plt.suptitle(test_name, fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def run_single_test(test_config, test_num):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_config['name']}")
    print(f"{'='*60}")
    
    # Generate Room A
    print(f"Generating Room A: {test_config['room_a']['width']}m × {test_config['room_a']['height']}m")
    a_poly = make_realistic_room(**test_config['room_a'])
    
    # Generate Room B
    print(f"Generating Room B: {test_config['room_b']['width']}m × {test_config['room_b']['height']}m")
    b_poly = make_realistic_room(**test_config['room_b'])
    
    print(f"Room A area: {a_poly.area:.2f} m², furniture: {len(a_poly.interiors)}")
    print(f"Room B area: {b_poly.area:.2f} m², furniture: {len(b_poly.interiors)}")
    
    # Sample points
    pts_a = sample_floor_points(a_poly, n_points=4000, seed=test_config['room_a']['seed'])
    pts_b = sample_floor_points(b_poly, n_points=3500, seed=test_config['room_b']['seed'])
    
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
    
    if len(matches) < 6:
        print("Too few matches, trying rotation search only...")
        best_affine, best_iou, best_inter = find_best_alignment_by_rotation(
            poly_a_world, poly_b_world, rotation_angles=list(range(0, 360, 15)), use_centroids=True
        )
        if best_affine is None:
            print("Failed to find alignment")
            return None
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
                        poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine_rotation)
                        metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
                    else:
                        print(f"Using feature-based (IoU: {metrics_feature['iou']:.4f} >= {best_iou_rotation:.4f})")
                        poly_b_aligned = poly_b_feature
                        metrics = metrics_feature
                else:
                    poly_b_aligned = apply_affine_to_polygon(poly_b_world, best_affine_rotation)
                    metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
    
    if metrics is None:
        print("Failed to compute metrics")
        return None
    
    # Visualize with occupancy images and features
    inter = metrics['intersection']
    outpath = os.path.join(OUT, f"test_{test_num:02d}_{test_config['name'].replace(' ', '_').lower()}.png")
    
    plot_test_result(poly_a_world, poly_b_world, poly_b_aligned, inter, metrics, 
                    test_config['name'], outpath,
                    occ_a=occ_a_clean, occ_b=occ_b_clean,
                    bbox_a=bbox_a, bbox_b=bbox_b,
                    thumbA=thumbA, thumbB=thumbB,
                    matches=matches_for_viz,
                    ptsA_uv=ptsA_uv, ptsB_uv=ptsB_uv,
                    inlier_mask=inlier_mask_for_viz)
    
    print(f"✓ Test {test_num} complete: IoU={metrics['iou']:.4f}, "
          f"Intersection={metrics['intersection_area']:.2f} m²")
    print(f"  Saved to {outpath}")
    
    return metrics

def main():
    # Define 5 different test cases
    test_cases = [
        {
            'name': 'Test 1: Large vs Small',
            'room_a': {
                'width': 8.0, 'height': 6.0, 'origin': (0.0, 0.0),
                'seed': 101, 'furniture_density': 0.12,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.05
            },
            'room_b': {
                'width': 4.5, 'height': 3.5, 'origin': (0.0, 0.0),
                'seed': 202, 'furniture_density': 0.18,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.08
            }
        },
        {
            'name': 'Test 2: Wide vs Tall',
            'room_a': {
                'width': 9.0, 'height': 4.0, 'origin': (0.0, 0.0),
                'seed': 303, 'furniture_density': 0.15,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.06
            },
            'room_b': {
                'width': 4.5, 'height': 7.0, 'origin': (0.0, 0.0),
                'seed': 404, 'furniture_density': 0.16,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.07
            }
        },
        {
            'name': 'Test 3: Similar Size, Different Layout',
            'room_a': {
                'width': 6.0, 'height': 5.0, 'origin': (0.0, 0.0),
                'seed': 505, 'furniture_density': 0.14,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.05
            },
            'room_b': {
                'width': 5.5, 'height': 5.5, 'origin': (0.0, 0.0),
                'seed': 606, 'furniture_density': 0.17,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.09
            }
        },
        {
            'name': 'Test 4: Very Different Sizes',
            'room_a': {
                'width': 10.0, 'height': 7.0, 'origin': (0.0, 0.0),
                'seed': 707, 'furniture_density': 0.13,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.04
            },
            'room_b': {
                'width': 3.5, 'height': 3.0, 'origin': (0.0, 0.0),
                'seed': 808, 'furniture_density': 0.22,
                'add_alcoves': False, 'add_wall_indentations': True, 'wall_irregularity': 0.12
            }
        },
        {
            'name': 'Test 5: Medium Rooms, High Irregularity',
            'room_a': {
                'width': 6.5, 'height': 5.5, 'origin': (0.0, 0.0),
                'seed': 909, 'furniture_density': 0.16,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.12
            },
            'room_b': {
                'width': 5.5, 'height': 4.5, 'origin': (0.0, 0.0),
                'seed': 1010, 'furniture_density': 0.19,
                'add_alcoves': True, 'add_wall_indentations': True, 'wall_irregularity': 0.15
            }
        }
    ]
    
    results = []
    for i, test_config in enumerate(test_cases, 1):
        metrics = run_single_test(test_config, i)
        if metrics:
            results.append({
                'test': test_config['name'],
                'iou': metrics['iou'],
                'intersection_area': metrics['intersection_area'],
                'area_a': metrics['area_a'],
                'area_b': metrics['area_b']
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['test']}:")
        print(f"  IoU: {r['iou']:.4f}")
        print(f"  Intersection: {r['intersection_area']:.2f} m²")
        print(f"  Room A: {r['area_a']:.2f} m², Room B: {r['area_b']:.2f} m²")
        print()
    
    print(f"All test visualizations saved to {OUT}/")
    print("Files: test_01_*.png through test_05_*.png")

if __name__ == "__main__":
    main()
