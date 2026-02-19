# demo/demo_feature_refine.py
"""
Demo: use thumbnails + ORB features to estimate a transform between two rooms,
map transform back to world coordinates, apply to polygon, compute IoU.

Run:
    PYTHONPATH=. python demo/demo_feature_refine.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping, MultiPolygon

from src.synthetic_generator import make_realistic_room, sample_floor_points
from src.rasterize import points_to_occupancy, postprocess_occupancy, occupancy_to_polygon
from src.thumbnail_features import (
    render_thumbnail_from_occupancy, _occ_to_thumbnail_mapping,
    detect_orb_features, match_descriptors_knn_ratio, estimate_transform_from_matches,
    compute_polygon_intersection_metrics
)

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)

def save_geojson(poly, path):
    if poly is None:
        return
    with open(path, "w") as f:
        json.dump(mapping(poly), f)

def imshow_and_save(img, path, cmap='gray'):
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap=cmap, origin='upper')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_matches(imgA, imgB, kpsA, kpsB, matches, inlier_mask=None, outpath=None):
    # draw matches using OpenCV drawMatches (requires conversion of kps to cv2.KeyPoint; we only have pts)
    # Build cv2.KeyPoint lists
    cv_kpsA = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in kpsA]
    cv_kpsB = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in kpsB]
    # matches is list of cv2.DMatch -> draw
    draw_img = cv2.drawMatches(imgA, cv_kpsA, imgB, cv_kpsB, matches, None,
                               matchColor=(0,255,0), singlePointColor=(255,0,0),
                               matchesMask=(inlier_mask.astype(int).tolist() if inlier_mask is not None else None))
    # convert BGR->RGB for matplotlib
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(draw_img)
    plt.axis('off')
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

import cv2  # local import for drawing

def main():
    # Generate two completely different rooms with different sizes, layouts, and furniture
    print("Generating Room A (therapist)...")
    a_poly = make_realistic_room(
        width=7.0,           # Larger width
        height=5.5,          # Larger height
        origin=(0.0, 0.0),
        seed=101,
        furniture_density=0.15,  # More furniture
        add_alcoves=True,
        add_wall_indentations=True,
        wall_irregularity=0.08  # More irregular walls
    )
    
    print("Generating Room B (patient)...")
    b_poly = make_realistic_room(
        width=5.0,           # Smaller width
        height=4.0,          # Smaller height
        origin=(0.0, 0.0),
        seed=999,            # Very different seed for different layout
        furniture_density=0.20,  # Even more furniture, different arrangement
        add_alcoves=True,
        add_wall_indentations=True,
        wall_irregularity=0.10  # More irregular walls
    )
    
    print(f"Room A area: {a_poly.area:.2f} m², furniture pieces: {len(a_poly.interiors)}")
    print(f"Room B area: {b_poly.area:.2f} m², furniture pieces: {len(b_poly.interiors)}")
    
    # Sample points from both rooms independently (simulate separate scans)
    pts_a = sample_floor_points(a_poly, n_points=4000, seed=101)
    pts_b = sample_floor_points(b_poly, n_points=3500, seed=999)  # Different number of points

    # rasterize both
    occ_a, bbox_a, res_a = points_to_occupancy(pts_a[:, :2], resolution=0.03, margin=0.2)
    occ_b, bbox_b, res_b = points_to_occupancy(pts_b[:, :2], resolution=0.03, margin=0.2)

    occ_a_clean = postprocess_occupancy(occ_a, closing_iters=3, min_component_area_px=30)
    occ_b_clean = postprocess_occupancy(occ_b, closing_iters=3, min_component_area_px=30)

    poly_a = occupancy_to_polygon(occ_a_clean, bbox_a, res_a)
    poly_b = occupancy_to_polygon(occ_b_clean, bbox_b, res_b)
    save_geojson(poly_a, os.path.join(OUT, "polyA_for_features.geojson"))
    save_geojson(poly_b, os.path.join(OUT, "polyB_for_features.geojson"))

    # render thumbnails
    thumbA = render_thumbnail_from_occupancy(occ_a_clean, bbox_a, out_size=256)
    thumbB = render_thumbnail_from_occupancy(occ_b_clean, bbox_b, out_size=256)
    imshow_and_save(thumbA, os.path.join(OUT, "thumbA.png"))
    imshow_and_save(thumbB, os.path.join(OUT, "thumbB.png"))

    # get thumbnail<->world mapping helpers
    occ_to_thumb_A, thumb_to_occ_A, thumbA_to_world, world_to_thumbA = _occ_to_thumbnail_mapping(bbox_a, occ_a_clean.shape, 256)
    occ_to_thumb_B, thumb_to_occ_B, thumbB_to_world, world_to_thumbB = _occ_to_thumbnail_mapping(bbox_b, occ_b_clean.shape, 256)

    # detect ORB features
    kpsA, ptsA_uv, descA = detect_orb_features(thumbA, n_features=800)
    kpsB, ptsB_uv, descB = detect_orb_features(thumbB, n_features=800)
    print("Detected keypoints:", len(kpsA), len(kpsB))

    # match descriptors
    matches = match_descriptors_knn_ratio(descA, descB, ratio=0.85)
    print("Matches after ratio:", len(matches))

    if len(matches) < 6:
        print("Too few matches to estimate transform reliably.")
        return

    # estimate image->image transform from thumbnail pixel coords
    result = estimate_transform_from_matches(ptsA_uv, ptsB_uv, matches, ransac_thresh=4.0)
    if result is None or result[0] is None:
        print("Failed to estimate transform from thumbnails.")
        return
    M, inlier_mask = result
    print("Estimated image affine M:", M)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print("Inlier matches:", inlier_count, "out of", len(matches))

    # save match visualization (draw matches with inliers highlighted)
    try:
        visualize_matches(thumbA, thumbB, ptsA_uv, ptsB_uv, matches, inlier_mask, outpath=os.path.join(OUT, "feature_matches.png"))
    except Exception as e:
        print("Failed to draw match visualization:", e)

    # Convert image affine M (maps ptsA_uv -> ptsB_uv) into a world->world transform:
    # We want a transform T_world that maps points in B's world frame into A's world frame (so we can align B to A).
    # Steps:
    #  - For a thumbnail pixel (uA,vA) in A, thumbnail_to_world_A(uA,vA) -> (xA,yA)
    #  - For a thumbnail pixel (uB,vB) in B, thumbnail_to_world_B(uB,vB) -> (xB,yB)
    #  - M approximates: [uB,vB,1]^T ≈ M * [uA, vA, 1]^T  (if M maps A->B)
    # estimate_transform_from_matches above used src=thumbA -> dst=thumbB, so:
    #    [uB]   [ m00 m01 m02 ] [uA]
    #    [vB] = [ m10 m11 m12 ] [vA]
    # We want X_B_world(uB,vB) and X_A_world(uA,vA). So the composed mapping:
    #    world_B = thumbB_to_world( M * [uA,vA] )
    #    world_A = thumbA_to_world( uA, vA )
    # Find affine in world coords that sends world_B -> world_A (or inverse depending on convention).
    # Practical approach: sample many matched inlier thumbnail keypoints, map them to world coords and run Umeyama.

    # collect inlier matched pairs in thumbnail pixels (A_uv -> B_uv)
    src_uv = np.array([ptsA_uv[m.queryIdx] for i,m in enumerate(matches) if inlier_mask is None or inlier_mask[i]], dtype=float)
    dst_uv = np.array([ptsB_uv[m.trainIdx] for i,m in enumerate(matches) if inlier_mask is None or inlier_mask[i]], dtype=float)
    if src_uv.shape[0] < 4:
        print("Not enough inlier matches for stable world transform.")
        return

    # map thumbnail pixels to world coordinates
    src_world = np.array([ thumbA_to_world(u,v) for (u,v) in src_uv ])
    dst_world = np.array([ thumbB_to_world(u,v) for (u,v) in dst_uv ])

    # estimate Umeyama transform mapping B_world -> A_world (so transform that maps B->A)
    # Use rigid transform (no scale) to preserve Room A's scale
    from src.polygon_ops import estimate_rigid_umeyama, apply_affine_to_polygon, find_best_alignment_by_rotation
    # estimate mapping from dst_world -> src_world (so transform that maps B->A)
    # Using rigid transform to preserve scale (Room A should remain unchanged)
    world_affine_feature = estimate_rigid_umeyama(dst_world, src_world)
    print("Estimated world affine from features (B -> A, rigid, no scale):", world_affine_feature)
    
    # Get Room B polygon for rotation search
    poly_b_world_temp = occupancy_to_polygon(occ_b_clean, bbox_b, res_b)
    poly_a_world_temp = occupancy_to_polygon(occ_a_clean, bbox_a, res_a)
    
    if poly_b_world_temp is not None and poly_a_world_temp is not None:
        print("\nTrying different rotation angles to find best alignment...")
        # Try rotations around the feature-based estimate (±30 degrees in 5-degree steps)
        feature_rotation = np.arctan2(world_affine_feature[2], world_affine_feature[0]) * 180 / np.pi
        print(f"Feature-based rotation estimate: {feature_rotation:.1f}°")
        
        # Create rotation angles: around feature estimate ±30°, plus major angles
        base_angles = [0, 90, 180, 270]
        fine_angles = list(range(int(feature_rotation - 30), int(feature_rotation + 35), 5))
        rotation_angles = sorted(set(base_angles + fine_angles))
        
        best_affine_rotation, best_iou_rotation, best_inter_rotation = find_best_alignment_by_rotation(
            poly_a_world_temp, poly_b_world_temp, rotation_angles=rotation_angles, use_centroids=True
        )
        
        print(f"Best rotation found: IoU={best_iou_rotation:.4f}, intersection={best_inter_rotation:.2f} m²")
        
        # Compare feature-based vs rotation-search results
        # Apply feature-based transform to check its IoU
        poly_b_feature_aligned = apply_affine_to_polygon(poly_b_world_temp, world_affine_feature)
        if poly_b_feature_aligned and not poly_b_feature_aligned.is_empty:
            inter_feature = poly_a_world_temp.intersection(poly_b_feature_aligned)
            inter_area_feature = inter_feature.area if inter_feature and not inter_feature.is_empty else 0.0
            union_feature = poly_a_world_temp.union(poly_b_feature_aligned)
            union_area_feature = union_feature.area if union_feature and not union_feature.is_empty else poly_a_world_temp.area
            iou_feature = inter_area_feature / union_area_feature if union_area_feature > 0 else 0.0
            
            print(f"Feature-based alignment: IoU={iou_feature:.4f}, intersection={inter_area_feature:.2f} m²")
            
            # Use the better alignment
            if best_iou_rotation > iou_feature:
                print(f"✓ Using rotation-search alignment (better IoU: {best_iou_rotation:.4f} > {iou_feature:.4f})")
                world_affine = best_affine_rotation
            else:
                print(f"✓ Using feature-based alignment (better IoU: {iou_feature:.4f} >= {best_iou_rotation:.4f})")
                world_affine = world_affine_feature
        else:
            print("Feature-based transform produced invalid result, using rotation-search result")
            world_affine = best_affine_rotation
    else:
        print("Could not perform rotation search, using feature-based alignment")
        world_affine = world_affine_feature
    
    print("Final world affine (B -> A):", world_affine)

    # Apply world_affine to polygon B to align into A's frame
    poly_b_world = occupancy_to_polygon(occ_b_clean, bbox_b, res_b)  # patient polygon
    if poly_b_world is None:
        print("No polygon B reconstructed to apply transform")
        return
    
    # Validate transform before applying
    a, b, d, e, xoff, yoff = world_affine
    scale = np.sqrt(a*a + d*d)
    rotation_deg = np.arctan2(d, a) * 180 / np.pi
    print(f"Transform details: scale={scale:.3f}, rotation={rotation_deg:.2f}°, translation=({xoff:.2f}, {yoff:.2f})")
    
    # Validate scale is close to 1.0 (rigid transform should preserve scale)
    if abs(scale - 1.0) > 0.1:
        print(f"WARNING: Estimated scale ({scale:.3f}) deviates significantly from 1.0!")
        print("This may indicate issues with thumbnail-to-world mapping or room size differences.")
    
    # Get Room A polygon BEFORE any transforms (it should remain unchanged)
    poly_a_world = occupancy_to_polygon(occ_a_clean, bbox_a, res_a)
    if poly_a_world is None or poly_a_world.is_empty:
        print("ERROR: Polygon A is invalid/empty")
        return
    
    area_a_before = poly_a_world.area
    print(f"Room A area (should remain unchanged): {area_a_before:.2f} m²")
    
    # Apply transform to Room B only (Room A remains unchanged)
    poly_b_aligned = apply_affine_to_polygon(poly_b_world, world_affine)
    
    # Validate transformed polygon
    if poly_b_aligned is None or poly_b_aligned.is_empty:
        print("ERROR: Transform produced invalid/empty polygon")
        return
    
    # Verify Room A hasn't been accidentally transformed
    area_a_after = poly_a_world.area
    if abs(area_a_before - area_a_after) > 0.01:
        print(f"ERROR: Room A area changed from {area_a_before:.2f} to {area_a_after:.2f} m²!")
        print("Room A should NOT be transformed - only Room B should be aligned.")
    else:
        print(f"✓ Room A area verified unchanged: {area_a_after:.2f} m²")
    
    # Compute intersection metrics using helper function (handles rotated polygons automatically)
    metrics = compute_polygon_intersection_metrics(poly_a_world, poly_b_aligned)
    if metrics is None:
        print("ERROR: Failed to compute intersection metrics")
        return
    
    inter = metrics['intersection']
    inter_area = metrics['intersection_area']
    union_area = metrics['union_area']
    iou = metrics['iou']
    area_a = metrics['area_a']
    area_b_aligned = metrics['area_b']
    
    print("\n=== Intersection Analysis (Rotated Polygons) ===")
    print(f"Room A area: {area_a:.2f} m²")
    print(f"Room B area (aligned): {area_b_aligned:.2f} m²")
    print(f"Intersection area: {inter_area:.2f} m²")
    print(f"Union area: {union_area:.2f} m²")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Overlap percentage (relative to A): {metrics['overlap_pct_a']:.2f}%")
    print(f"Overlap percentage (relative to B): {metrics['overlap_pct_b']:.2f}%")

    # save outputs
    save_geojson(poly_a_world, os.path.join(OUT, "poly_A_features.geojson"))
    save_geojson(poly_b_world, os.path.join(OUT, "poly_B_features.geojson"))
    save_geojson(poly_b_aligned, os.path.join(OUT, "poly_B_aligned_features.geojson"))
    # save thumbnails and matches already saved
    print("Saved feature-match debug images to", OUT)
    # Enhanced visualization showing rotated polygons and intersection
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax1, ax2 = axes
    
    # Helper function to plot polygon (handles MultiPolygon)
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
                    # Plot interiors (furniture)
                    for interior in geom.interiors:
                        xi, yi = zip(*interior.coords)
                        ax.fill(xi, yi, color='gray', alpha=0.2)
        elif hasattr(poly, 'exterior') and poly.exterior is not None:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if fill:
                ax.fill(x, y, color=color, alpha=alpha)
            # Plot interiors (furniture)
            for interior in poly.interiors:
                xi, yi = zip(*interior.coords)
                ax.fill(xi, yi, color='gray', alpha=0.2)
    
    # Left plot: Original polygons before alignment
    plot_polygon(ax1, poly_a_world, 'tab:blue', label='Room A', alpha=0.2)
    plot_polygon(ax1, poly_b_world, 'tab:green', label='Room B (original)', alpha=0.2)
    ax1.set_title('Before Alignment\n(Room B may be rotated/translated)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: After alignment with intersection
    plot_polygon(ax2, poly_a_world, 'tab:blue', label='Room A', alpha=0.2)
    plot_polygon(ax2, poly_b_aligned, 'tab:red', linestyle='--', label=f'Room B (aligned, rotated {rotation_deg:.1f}°)', alpha=0.2)
    
    # Highlight intersection
    if inter is not None and not inter.is_empty:
        if isinstance(inter, MultiPolygon):
            for i, geom in enumerate(inter.geoms):
                if hasattr(geom, 'exterior') and geom.exterior is not None:
                    xi, yi = geom.exterior.xy
                    label = f'Intersection ({inter_area:.2f} m²)' if i == 0 else None
                    ax2.fill(xi, yi, color='purple', alpha=0.6, label=label, edgecolor='darkviolet', linewidth=2)
        elif hasattr(inter, 'exterior') and inter.exterior is not None:
            xi, yi = inter.exterior.xy
            ax2.fill(xi, yi, color='purple', alpha=0.6, label=f'Intersection ({inter_area:.2f} m²)', 
                    edgecolor='darkviolet', linewidth=2)
    
    # Add text annotation with metrics
    ax2.text(0.02, 0.98, f'IoU: {iou:.3f}\nOverlap: {inter_area:.2f} m²\nUnion: {union_area:.2f} m²',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_title(f'After Feature-Based Alignment\n(IoU: {iou:.3f})')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "features_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved overlay to demo_out/features_overlay.png")

if __name__ == "__main__":
    main()
