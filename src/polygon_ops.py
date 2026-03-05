# src/polygon_ops.py
"""
Polygon operations + geometry-based alignment.

Functions:
- estimate_rigid_umeyama(src, dst): estimate a rigid 2D transform (rotation + translation, no scale)
  between corresponding point sets.
- apply_affine_to_polygon(poly, affine_params): apply a 2D affine transform to a Shapely polygon.
- find_best_alignment_by_rotation(poly_a, poly_b, rotation_angles=None, use_centroids=True):
  search over rotations to maximize polygon intersection / IoU and return the best affine transform.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import affine_transform, rotate, translate

def estimate_rigid_umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Estimate rigid transform (rotation+translation, NO scale) mapping src -> dst
    using Umeyama method with scale fixed to 1.0.
    src, dst: (N,2) arrays with correspondences (N>=2)
    Returns affine params [a, b, d, e, xoff, yoff] suitable for shapely.affine_transform
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= 2

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s_src = src - mu_s
    s_dst = dst - mu_d

    cov = (s_dst.T @ s_src) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1,1] = -1
    R = U @ S @ Vt
    # Force scale to 1.0 for rigid transform
    A = R  # No scaling
    t = mu_d - A @ mu_s
    a, b = A[0,0], A[0,1]
    d, e = A[1,0], A[1,1]
    xoff, yoff = t[0], t[1]
    return [a, b, d, e, xoff, yoff]

def apply_affine_to_polygon(poly: Polygon, affine_params):
    if poly is None or affine_params is None:
        return poly
    return affine_transform(poly, affine_params)

def find_best_alignment_by_rotation(poly_a, poly_b, rotation_angles=None, use_centroids=True):
    """
    Try different rotation angles and find the one that maximizes intersection/IoU.
    
    Args:
        poly_a: Reference polygon (remains unchanged)
        poly_b: Polygon to align (will be rotated)
        rotation_angles: List of angles in degrees to try. If None, tries [0, 90, 180, 270] and fine-grained around best
        use_centroids: If True, centers both polygons before rotation
    
    Returns:
        best_affine: Affine transform [a, b, d, e, xoff, yoff] that gives best alignment
        best_iou: IoU value for best alignment
        best_inter_area: Intersection area for best alignment
    """
    if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
        return None, 0.0, 0.0
    
    from shapely.affinity import rotate, translate
    
    # Ensure polygons are valid before processing
    if not poly_a.is_valid:
        try:
            poly_a = poly_a.buffer(0)
        except:
            return None, 0.0, 0.0
    
    if not poly_b.is_valid:
        try:
            poly_b = poly_b.buffer(0)
        except:
            return None, 0.0, 0.0
    
    # Store original poly_a (don't modify it in the loop)
    poly_a_original = poly_a
    
    # Center both polygons for rotation
    if use_centroids:
        centroid_a = poly_a.centroid
        centroid_b = poly_b.centroid
        poly_a_centered = translate(poly_a, xoff=-centroid_a.x, yoff=-centroid_a.y)
        poly_b_centered = translate(poly_b, xoff=-centroid_b.x, yoff=-centroid_b.y)
        
        # Validate centered polygons
        if not poly_a_centered.is_valid:
            try:
                poly_a_centered = poly_a_centered.buffer(0)
            except:
                return None, 0.0, 0.0
        if not poly_b_centered.is_valid:
            try:
                poly_b_centered = poly_b_centered.buffer(0)
            except:
                return None, 0.0, 0.0
    else:
        centroid_a = None
        centroid_b = None
        poly_a_centered = poly_a
        poly_b_centered = poly_b
    
    # Default rotation angles: coarse grid first
    if rotation_angles is None:
        rotation_angles = list(range(0, 360, 15))  # Try every 15 degrees
    
    best_iou = 0.0
    best_inter_area = 0.0
    best_angle = 0.0
    best_affine = None
    
    for angle_deg in rotation_angles:
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate poly_b around its centroid
        poly_b_rotated_centered = rotate(poly_b_centered, angle_deg, origin=(0, 0), use_radians=False)
        
        # For each rotation, align centroids for optimal translation
        if use_centroids:
            # Rotate poly_b around its centroid, then translate to align with poly_a's centroid
            centroid_a_centered = poly_a_centered.centroid
            centroid_b_rotated = poly_b_rotated_centered.centroid
            
            # Translation to align centroids in centered space
            tx = centroid_a_centered.x - centroid_b_rotated.x
            ty = centroid_a_centered.y - centroid_b_rotated.y
            
            # Apply translation in centered space
            poly_b_aligned_centered = translate(poly_b_rotated_centered, xoff=tx, yoff=ty)
            
            # Translate back to original coordinates
            poly_b_rotated = translate(poly_b_aligned_centered, xoff=centroid_a.x, yoff=centroid_a.y)
            
            # Build affine transform: rotate around centroid_b, then translate
            # T = T(centroid_a + offset) * R(angle) * T(-centroid_b)
            rotated_centroid_b_world = np.array([
                cos_a * centroid_b.x - sin_a * centroid_b.y,
                sin_a * centroid_b.x + cos_a * centroid_b.y
            ])
            # Final translation: centroid_a - rotated_centroid_b_world + offset from centroid alignment
            offset_world = np.array([centroid_a.x, centroid_a.y]) - rotated_centroid_b_world
            # Add the centroid alignment offset (converted to world coordinates)
            final_translation = offset_world + np.array([tx, ty])
            
            best_affine_candidate = [
                cos_a, -sin_a,
                sin_a, cos_a,
                final_translation[0], final_translation[1]
            ]
        else:
            # No centering: simple rotation around origin
            poly_b_rotated = rotate(poly_b_centered, angle_deg, origin=(0, 0), use_radians=False)
            best_affine_candidate = [cos_a, -sin_a, sin_a, cos_a, 0.0, 0.0]
        
        # Validate polygons before computing intersection (use original poly_a, don't modify it)
        poly_a_for_inter = poly_a_original  # Use original poly_a (already validated at start)
        
        if not poly_b_rotated.is_valid:
            try:
                poly_b_rotated = poly_b_rotated.buffer(0)
                if poly_b_rotated.is_empty:
                    continue
            except:
                continue  # Skip this rotation if poly_b_rotated can't be fixed
        
        # Compute intersection with this rotation+translation
        try:
            inter = poly_a_for_inter.intersection(poly_b_rotated)
            inter_area = inter.area if inter and not inter.is_empty else 0.0
            
            union = poly_a_for_inter.union(poly_b_rotated)
            union_area = union.area if union and not union.is_empty else poly_a_for_inter.area
            iou = inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            # Skip this rotation if intersection fails (invalid geometry)
            # Try with buffered polygons to fix precision issues
            try:
                poly_a_buf = poly_a_for_inter.buffer(0.001)
                poly_b_buf = poly_b_rotated.buffer(0.001)
                if poly_a_buf.is_empty or poly_b_buf.is_empty:
                    continue
                inter = poly_a_buf.intersection(poly_b_buf)
                inter_area = inter.area if inter and not inter.is_empty else 0.0
                union = poly_a_buf.union(poly_b_buf)
                union_area = union.area if union and not union.is_empty else poly_a_buf.area
                iou = inter_area / union_area if union_area > 0 else 0.0
            except:
                continue  # Skip this rotation entirely if still fails
        
        if iou > best_iou or (iou == best_iou and inter_area > best_inter_area):
            best_iou = iou
            best_inter_area = inter_area
            best_angle = angle_deg
            best_affine = best_affine_candidate
    
    return best_affine, best_iou, best_inter_area

