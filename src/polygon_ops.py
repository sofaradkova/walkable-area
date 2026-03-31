# src/polygon_ops.py
"""
Polygon operations + geometry-based alignment.

Functions:
- estimate_rigid_umeyama(src, dst): estimate a rigid 2D transform (rotation + translation, no scale)
  between corresponding point sets.
- apply_affine_to_polygon(poly, affine_params): apply a 2D affine transform to a Shapely polygon.
- pca_candidate_rotations(poly_a, poly_b): use PCA of each polygon's vertices to generate 4
  candidate alignment angles that orient their principal axes.
- find_best_alignment_by_rotation(poly_a, poly_b, rotation_angles=None, use_centroids=True,
  min_passage_width=0.5): search over rotations to maximize the largest contiguous walkable
  intersection (eroded by min_passage_width/2 to remove thin impassable connections).
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
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


def _pca_polygon(poly):
    """Return the angle (radians) of the principal axis of a polygon's vertex cloud."""
    if isinstance(poly, MultiPolygon):
        coords = np.concatenate([np.array(g.exterior.coords[:-1]) for g in poly.geoms])
    else:
        coords = np.array(poly.exterior.coords[:-1])
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    _, vecs = np.linalg.eigh(cov)
    principal = vecs[:, -1]
    return np.arctan2(principal[1], principal[0])


def pca_candidate_rotations(poly_a, poly_b):
    """
    Use PCA of each polygon's vertex cloud to generate 4 candidate alignment angles
    (degrees). Aligning principal axes produces 4 candidates due to the 180° sign
    ambiguity of PCA axes (0 / 90 / 180 / 270 degree variants).
    """
    angle_a = _pca_polygon(poly_a)
    angle_b = _pca_polygon(poly_b)
    delta = np.rad2deg(angle_a - angle_b)
    return [(delta + k * 90) % 360 for k in range(4)]


def _largest_walkable_intersection_area(poly_a, poly_b_aligned, min_passage_width=0.5):
    """
    Return the area of the largest contiguous walkable region in the intersection.

    Erodes the intersection by min_passage_width / 2 to remove narrow connections
    that are geometrically present but not actually walkable (e.g. thin slivers or
    corridors narrower than a person). Returns the area of the largest remaining
    contiguous piece (0.0 if the eroded result is empty).
    """
    try:
        inter = poly_a.intersection(poly_b_aligned)
        if inter is None or inter.is_empty:
            return 0.0
        eroded = inter.buffer(-min_passage_width / 2.0)
        if eroded is None or eroded.is_empty:
            return 0.0
        if isinstance(eroded, MultiPolygon):
            return max(g.area for g in eroded.geoms)
        return eroded.area
    except Exception:
        return 0.0


def find_best_alignment_by_rotation(poly_a, poly_b, rotation_angles=None, use_centroids=True,
                                    min_passage_width=0.5):
    """
    Try different rotation angles and find the one that maximizes the largest
    contiguous walkable intersection area (not raw IoU).

    The intersection is eroded by min_passage_width/2 before scoring so that thin
    connections narrower than a person can walk are ignored. The largest remaining
    contiguous piece is what gets maximised.

    Args:
        poly_a: Reference polygon (remains unchanged)
        poly_b: Polygon to align (will be rotated)
        rotation_angles: List of angles in degrees to try. If None, uses PCA-derived
            candidate angles (4) plus a full 15° coarse grid (24), deduplicated.
        use_centroids: If True, centers both polygons before rotation
        min_passage_width: Minimum walkable passage width in metres (default 0.5 m).
            Connections narrower than this are treated as impassable and excluded
            from the score.

    Returns:
        best_affine: Affine transform [a, b, d, e, xoff, yoff] that gives best alignment
        best_walkable_area: Largest walkable intersection area for best alignment
        best_inter_area: Raw intersection area for best alignment
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
    
    # Default: PCA-derived candidates + full 15° coarse grid, deduplicated
    if rotation_angles is None:
        pca_angles = pca_candidate_rotations(poly_a, poly_b)
        coarse = list(range(0, 360, 15))
        # Round PCA angles to nearest degree and merge with coarse grid
        pca_rounded = [round(a) % 360 for a in pca_angles]
        seen = set()
        rotation_angles = []
        for a in pca_rounded + coarse:
            if a not in seen:
                seen.add(a)
                rotation_angles.append(a)

    best_walkable_area = 0.0
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
        
        # Score by largest contiguous walkable intersection (eroded to remove thin passages)
        try:
            inter = poly_a_for_inter.intersection(poly_b_rotated)
            inter_area = inter.area if inter and not inter.is_empty else 0.0
        except Exception:
            try:
                inter = poly_a_for_inter.buffer(0.001).intersection(poly_b_rotated.buffer(0.001))
                inter_area = inter.area if inter and not inter.is_empty else 0.0
            except Exception:
                continue

        walkable_area = _largest_walkable_intersection_area(
            poly_a_for_inter, poly_b_rotated, min_passage_width
        )

        if walkable_area > best_walkable_area or (
            walkable_area == best_walkable_area and inter_area > best_inter_area
        ):
            best_walkable_area = walkable_area
            best_inter_area = inter_area
            best_angle = angle_deg
            best_affine = best_affine_candidate

    return best_affine, best_walkable_area, best_inter_area

