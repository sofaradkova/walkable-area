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
from shapely.affinity import affine_transform
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

