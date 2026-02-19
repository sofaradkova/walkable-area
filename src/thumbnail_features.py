# src/thumbnail_features.py
"""
Thumbnail rendering + feature detection + matching + image->world transform helpers.

APIs:
- render_thumbnail_from_occupancy(occ, bbox, out_size=256, pad=0)
- world_to_thumbnail_pixels(x, y, bbox, occ_shape, out_size)
- thumbnail_pixels_to_world(u, v, bbox, occ_shape, out_size)
- detect_orb_features(image_gray, n_features=500)
- match_descriptors_ratio(desc1, desc2, ratio=0.75, cross_check=True)
- estimate_transform_from_matches(kp1, kp2, matches, method=cv2.RANSAC, ransac_thresh=3.0)

Notes:
- Uses OpenCV for feature detection/matching and transform estimation.
- The thumbnail produced is grayscale uint8 (0..255).
"""
from typing import Tuple, List, Optional
import numpy as np
import cv2

# ---------- thumbnail render helpers ----------
def render_thumbnail_from_occupancy(occ: np.ndarray,
                                    bbox: Tuple[float,float,float,float],
                                    out_size: int = 256,
                                    pad: int = 0) -> np.ndarray:
    """
    Render a thumbnail from occupancy image + bbox.
    - occ: HxW binary uint8 (1 occupied)
    - bbox: (minx, miny, maxx, maxy) world coords of occ
    - out_size: thumbnail pixels (square)
    - pad: additional padding around occ when scaling into thumbnail (pixels)
    Returns: grayscale uint8 thumbnail (out_size x out_size)
    """
    H, W = occ.shape
    # create an RGBA-like image so we can draw boundaries: here, just use occ but upscale to square
    # First create an image with occ but pad if requested
    canvas = (occ * 255).astype(np.uint8)
    # Resize to out_size while keeping aspect ratio
    # compute scale to fit bounding box into square
    scale_x = out_size / W
    scale_y = out_size / H
    scale = min(scale_x, scale_y)
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    resized = cv2.resize(canvas, (newW, newH), interpolation=cv2.INTER_AREA)
    # place centered in square
    thumb = np.zeros((out_size, out_size), dtype=np.uint8)
    ox = (out_size - newW) // 2
    oy = (out_size - newH) // 2
    thumb[oy:oy+newH, ox:ox+newW] = resized
    # enhance edges (optional): compute simple distance transform to create texture
    # Uncomment below to add weak texture which may help in very empty scenes:
    # dt = cv2.distanceTransform(255 - thumb, cv2.DIST_L2, 5)
    # dt = (dt / (dt.max()+1e-9) * 255).astype(np.uint8)
    # thumb = cv2.addWeighted(thumb, 0.7, dt, 0.3, 0)
    return thumb

# ---------- pixel/world mapping ----------
def _occ_to_thumbnail_mapping(bbox: Tuple[float,float,float,float], occ_shape: Tuple[int,int], out_size: int):
    """
    Return functions to map between occupancy pixel coordinates and thumbnail pixel coordinates,
    and between thumbnail and world coords as needed.
    occ_shape = (H, W)
    """
    H, W = occ_shape
    minx, miny, maxx, maxy = bbox
    width_m = maxx - minx
    height_m = maxy - miny

    # occupancy pixel -> world:
    # u_occ (0..W-1), v_occ (0..H-1) => x = minx + (u / (W-1)) * width_m
    #                                    y = maxy - (v / (H-1)) * height_m
    def occpix_to_world(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = minx + (u / max(1, (W-1))) * width_m
        y = maxy - (v / max(1, (H-1))) * height_m
        return x, y

    # occupancy pixel -> thumbnail pixel (centered as in render_thumbnail_from_occupancy)
    scale_x = out_size / W
    scale_y = out_size / H
    scale = min(scale_x, scale_y)
    newW = int(round(W * scale))
    newH = int(round(H * scale))
    ox = (out_size - newW) // 2
    oy = (out_size - newH) // 2

    def occpix_to_thumb(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        ut = u * scale + ox
        vt = v * scale + oy
        return ut, vt

    def thumb_to_occpix(ut, vt):
        ut = np.asarray(ut, dtype=float)
        vt = np.asarray(vt, dtype=float)
        u = (ut - ox) / scale
        v = (vt - oy) / scale
        return u, v

    # thumbnail pixel -> world (compose)
    def thumb_to_world(ut, vt):
        u, v = thumb_to_occpix(ut, vt)
        return occpix_to_world(u, v)

    def world_to_thumb(x, y):
        # world -> occpix:
        u = ( (np.asarray(x) - minx) / max(1e-9, width_m) ) * max(1, (W-1))
        v = ( (maxy - np.asarray(y)) / max(1e-9, height_m) ) * max(1, (H-1))
        return occpix_to_thumb(u, v)

    return occpix_to_thumb, thumb_to_occpix, thumb_to_world, world_to_thumb

# ---------- ORB detect & matching ----------
def detect_orb_features(image_gray: np.ndarray, n_features: int = 600):
    """
    Detect ORB keypoints & descriptors.
    Returns:
    - kps: list of cv2.KeyPoint
    - pts: Nx2 numpy float32 array of (u,v) pixel coordinates
    - desc: numpy array descriptors (uint8) or None
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, desc = orb.detectAndCompute(image_gray, None)
    if kps is None or len(kps) == 0:
        return [], np.zeros((0,2), dtype=np.float32), None
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)  # (u,v)
    return kps, pts, desc

def match_descriptors_ratio(desc1, desc2, ratio: float = 0.75, cross_check: bool = True):
    """
    KNN ratio test + optional cross-check.
    Returns list of cv2.DMatch objects (or simple (i,j,dist) tuples).
    """
    if desc1 is None or desc2 is None:
        return []
    # Use BFMatcher with Hamming for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # KNN match k=2
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m in knn:
        if len(m) < 2:
            continue
        m1, m2 = m[0], m[1]
        if m1.distance < ratio * m2.distance:
            good.append(m1)
    if cross_check:
        # cross-check: ensure best match reciprocates
        bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn2 = bf2.knnMatch(desc2, desc1, k=1)
        reverse_best = {m[0].trainIdx: m[0].queryIdx for m in knn2 if len(m) >= 1}
        good2 = []
        for m in good:
            # m.queryIdx from desc1 -> m.trainIdx in desc2
            if reverse_best.get(m.trainIdx, None) == m.queryIdx:
                good2.append(m)
        return good2
    else:
        return good

# ---------- estimate transform using cv2 (RANSAC) ----------
def estimate_transform_from_matches(pts1_uv: np.ndarray,
                                    pts2_uv: np.ndarray,
                                    matches,
                                    ransac_thresh: float = 3.0,
                                    ransac_conf: float = 0.99):
    """
    Given keypoint pixel coords in image1 (pts1_uv) and image2 (pts2_uv) and matches (list of DMatches),
    produce an affine transform matrix (2x3) mapping pts1 -> pts2 using RANSAC.
    Returns: M (2x3 float) or None, inlier_mask (Nx1 bool)
    """
    if len(matches) < 3:
        return None, None
    src_pts = np.float32([pts1_uv[m.queryIdx] for m in matches]).reshape(-1,2)
    dst_pts = np.float32([pts2_uv[m.trainIdx] for m in matches]).reshape(-1,2)
    # use estimateAffinePartial2D which estimates similarity/affine with RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh, confidence=ransac_conf,
                                             maxIters=2000)
    # inliers is a mask array (N,) where 1 means inlier, 0 means outlier
    if M is None:
        return None, None
    inlier_mask = (inliers.flatten() == 1) if inliers is not None else None
    return M, inlier_mask

def compute_polygon_intersection_metrics(poly_a, poly_b_aligned):
    """
    Compute intersection metrics between two polygons (handles rotated polygons).
    
    Args:
        poly_a: First polygon (reference)
        poly_b_aligned: Second polygon (already transformed/aligned)
    
    Returns:
        dict with keys: 'intersection', 'intersection_area', 'union_area', 'iou',
                       'overlap_pct_a', 'overlap_pct_b', 'area_a', 'area_b'
    """
    from shapely.geometry import MultiPolygon
    
    if poly_a is None or poly_b_aligned is None:
        return None
    
    if poly_a.is_empty or poly_b_aligned.is_empty:
        return {
            'intersection': None,
            'intersection_area': 0.0,
            'union_area': poly_a.area if not poly_a.is_empty else 0.0,
            'iou': 0.0,
            'overlap_pct_a': 0.0,
            'overlap_pct_b': 0.0,
            'area_a': poly_a.area if not poly_a.is_empty else 0.0,
            'area_b': poly_b_aligned.area if not poly_b_aligned.is_empty else 0.0
        }
    
    # Validate and fix polygons if needed
    try:
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b_aligned.is_valid:
            poly_b_aligned = poly_b_aligned.buffer(0)
    except:
        # If buffering fails, return zero metrics
        return {
            'intersection': None,
            'intersection_area': 0.0,
            'union_area': poly_a.area if poly_a and not poly_a.is_empty else 0.0,
            'iou': 0.0,
            'overlap_pct_a': 0.0,
            'overlap_pct_b': 0.0,
            'area_a': poly_a.area if poly_a and not poly_a.is_empty else 0.0,
            'area_b': poly_b_aligned.area if poly_b_aligned and not poly_b_aligned.is_empty else 0.0
        }
    
    # Compute intersection (handles rotated polygons automatically)
    try:
        inter = poly_a.intersection(poly_b_aligned)
        inter_area = inter.area if inter and not inter.is_empty else 0.0
        
        # Compute union
        union = poly_a.union(poly_b_aligned)
        union_area = union.area if union and not union.is_empty else poly_a.area
    except Exception as e:
        # If intersection fails due to precision issues, try with buffered polygons
        try:
            poly_a_buf = poly_a.buffer(0.001)
            poly_b_buf = poly_b_aligned.buffer(0.001)
            if poly_a_buf.is_empty or poly_b_buf.is_empty:
                inter = None
                inter_area = 0.0
                union_area = poly_a.area
            else:
                inter = poly_a_buf.intersection(poly_b_buf)
                inter_area = inter.area if inter and not inter.is_empty else 0.0
                union = poly_a_buf.union(poly_b_buf)
                union_area = union.area if union and not union.is_empty else poly_a_buf.area
        except:
            # If still fails, return zero metrics
            inter = None
            inter_area = 0.0
            union_area = poly_a.area if poly_a and not poly_a.is_empty else 0.0
    
    # Compute metrics
    area_a = poly_a.area if poly_a and not poly_a.is_empty else 0.0
    area_b = poly_b_aligned.area if poly_b_aligned and not poly_b_aligned.is_empty else 0.0
    iou = inter_area / union_area if union_area > 0 else 0.0
    overlap_pct_a = (inter_area / area_a * 100) if area_a > 0 else 0.0
    overlap_pct_b = (inter_area / area_b * 100) if area_b > 0 else 0.0
    
    return {
        'intersection': inter,
        'intersection_area': inter_area,
        'union_area': union_area,
        'iou': iou,
        'overlap_pct_a': overlap_pct_a,
        'overlap_pct_b': overlap_pct_b,
        'area_a': area_a,
        'area_b': area_b
    }

def match_descriptors_knn_ratio(desc1, desc2, ratio: float = 0.85):
    """KNN ratio test only (no reciprocal cross-check)."""
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m in knn:
        if len(m) < 2:
            continue
        m1, m2 = m[0], m[1]
        if m1.distance < ratio * m2.distance:
            good.append(m1)
    return good