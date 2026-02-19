# src/polygon_ops.py
"""
Polygon operations + geometry-based alignment.

APIs:
- simplify_polygon(poly, tol)
- shrink_polygon(poly, margin)
- sample_polygon_boundary(poly, n_samples)
- estimate_similarity_umeyama(src_pts, dst_pts)
- ransac_similarity_transform(src_pts, dst_pts, n_iters=500, sample_size=3, inlier_thresh=0.2)
- apply_affine_to_polygon(poly, affine_params)

Affine format returned / accepted is [a, b, d, e, xoff, yoff] for shapely.affine_transform:
    x' = a*x + b*y + xoff
    y' = d*x + e*y + yoff
"""
from typing import Tuple, Optional
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import affine_transform, rotate, translate
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import math

def simplify_polygon(poly: Polygon, tol: float = 0.03) -> Optional[Polygon]:
    if poly is None:
        return None
    s = poly.simplify(tol)
    if s.is_empty:
        return poly
    return s

def shrink_polygon(poly: Polygon, margin: float = 0.2) -> Optional[Polygon]:
    if poly is None:
        return None
    shrunk = poly.buffer(-margin)
    # if buffer removed everything, fallback to tiny shrink
    if shrunk.is_empty:
        shrunk = poly.buffer(-min(margin, 0.01))
    # if still empty, return original polygon
    if shrunk.is_empty:
        return poly
    # ensure we return single polygon (take largest part)
    if shrunk.geom_type == 'MultiPolygon':
        geom = max(shrunk.geoms, key=lambda p: p.area)
        return geom
    return shrunk

def sample_polygon_boundary(poly: Polygon, n_samples: int = 300) -> np.ndarray:
    """
    Sample points equally along the exterior boundary of polygon.
    Returns Nx2 array.
    """
    if poly is None:
        return np.zeros((0,2))
    ext = poly.exterior
    length = ext.length
    if length <= 0:
        return np.zeros((0,2))
    dists = np.linspace(0, length, n_samples, endpoint=False)
    pts = [ext.interpolate(d) for d in dists]
    arr = np.array([(p.x, p.y) for p in pts], dtype=float)
    return arr

# ----------------- Umeyama similarity estimator -----------------
def estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Estimate similarity transform (scale+rotation+translation) mapping src -> dst
    using Umeyama method.
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
    var_s = (s_src ** 2).sum() / src.shape[0]
    # D from SVD is 1D array of singular values
    scale = np.trace(np.diag(D) @ S) / var_s if var_s != 0 else 1.0
    A = scale * R
    t = mu_d - A @ mu_s
    a, b = A[0,0], A[0,1]
    d, e = A[1,0], A[1,1]
    xoff, yoff = t[0], t[1]
    return [a, b, d, e, xoff, yoff]

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

# ----------------- RANSAC wrapper -----------------
def ransac_similarity_transform(src_pts: np.ndarray,
                                dst_pts: np.ndarray,
                                n_iters: int = 500,
                                sample_size: int = 3,
                                inlier_thresh: float = 0.25,
                                min_inliers: int = 30,
                                random_seed: Optional[int] = None):
    """
    Robustly estimate similarity transform mapping src_pts -> dst_pts.
    Approach:
      - Build KD-tree for dst_pts
      - Randomly sample sample_size correspondences (indices) from src_pts and dst_pts,
        estimate candidate transform, compute inliers as src_pts transformed close to nearest dst_pts.
      - Keep transform with most inliers and return transform and inlier mask.

    Note: This is a 'shape-only' RANSAC: no descriptor-based matching. It assumes
    both point sets are sampled from boundaries of similar shapes with some overlap.
    """
    rng = np.random.default_rng(random_seed)
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return None, None  # not enough points

    kdt = cKDTree(dst)
    best = {'inliers': [], 'affine': None}
    N_src = src.shape[0]

    for it in range(n_iters):
        # pick sample_size random indices from src and dst
        s_idx = rng.choice(N_src, size=sample_size, replace=False)
        d_idx = rng.choice(dst.shape[0], size=sample_size, replace=False)
        try:
            A = estimate_similarity_umeyama(src[s_idx], dst[d_idx])
        except Exception:
            continue
        # apply A to all src pts
        src_h = src.copy()
        # transform function
        a,b,d,e,xoff,yoff = A
        transformed = np.empty_like(src_h)
        transformed[:,0] = a * src_h[:,0] + b * src_h[:,1] + xoff
        transformed[:,1] = d * src_h[:,0] + e * src_h[:,1] + yoff
        # compute distance to nearest dst point
        dists, _ = kdt.query(transformed, k=1)
        inlier_mask = dists <= inlier_thresh
        inliers_idx = np.nonzero(inlier_mask)[0]
        nin = int(inliers_idx.size)
        if nin > len(best['inliers']):
            best['inliers'] = inliers_idx
            best['affine'] = A
            # early exit if very good
            if nin >= max(min_inliers, int(0.8 * N_src)):
                break

    if best['affine'] is None:
        return None, None

    # Optionally refine transform using inliers with Umeyama
    in_idx = best['inliers']
    # find corresponding dst points by nearest neighbor
    transformed = None
    a,b,d,e,xoff,yoff = best['affine']
    src_trans = np.empty_like(src)
    src_trans[:,0] = a * src[:,0] + b * src[:,1] + xoff
    src_trans[:,1] = d * src[:,0] + e * src[:,1] + yoff
    # for each inlier index in src, find nearest dst and use as correspondence
    src_in = src[in_idx]
    if src_in.shape[0] >= 2:
        # Transform inlier points and stack into Nx2 array for kdtree query
        src_in_transformed = np.column_stack([
            a * src_in[:,0] + b * src_in[:,1] + xoff,
            d * src_in[:,0] + e * src_in[:,1] + yoff
        ])
        _, dst_idx_for_in = kdt.query(src_in_transformed, k=1)
        dst_idx_for_in = dst_idx_for_in.flatten()  # Ensure 1D array
        dst_corr = dst[dst_idx_for_in]
        try:
            refined = estimate_similarity_umeyama(src_in, dst_corr)
            return refined, in_idx
        except Exception:
            return best['affine'], in_idx
    else:
        return best['affine'], in_idx

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
    
    # Center both polygons for rotation
    if use_centroids:
        centroid_a = poly_a.centroid
        centroid_b = poly_b.centroid
        poly_a_centered = translate(poly_a, xoff=-centroid_a.x, yoff=-centroid_a.y)
        poly_b_centered = translate(poly_b, xoff=-centroid_b.x, yoff=-centroid_b.y)
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
        
        # Compute intersection with this rotation+translation
        inter = poly_a.intersection(poly_b_rotated)
        inter_area = inter.area if inter and not inter.is_empty else 0.0
        union = poly_a.union(poly_b_rotated)
        union_area = union.area if union and not union.is_empty else poly_a.area
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        if iou > best_iou or (iou == best_iou and inter_area > best_inter_area):
            best_iou = iou
            best_inter_area = inter_area
            best_angle = angle_deg
            best_affine = best_affine_candidate
    
    return best_affine, best_iou, best_inter_area

