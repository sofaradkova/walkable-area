# src/alignment.py
"""
Align two 2D walkable-area polygons with a rigid transform (rotation + translation).

The search is seeded by PCA of each polygon's vertex cloud, which provides 4
geometrically-motivated candidate angles, then expanded with a fine angular grid.
Scoring uses a morphological erosion so that thin geometric slivers that are not
actually walkable do not inflate the score.

Public API:
- apply_affine_to_polygon(poly, affine_params)
- pca_candidate_rotations(poly_a, poly_b) -> list[float]
- find_best_alignment_by_rotation(poly_a, poly_b, ...) -> (affine, walkable_poly, raw_inter_area)
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform, rotate, translate

from src.intersection import largest_walkable_subpolygon


def apply_affine_to_polygon(poly: Polygon, affine_params):
    """Apply a 2D affine transform [a, b, d, e, xoff, yoff] to a Shapely polygon."""
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
    _, vecs = np.linalg.eigh(np.cov(coords.T))
    return np.arctan2(vecs[-1, 1], vecs[-1, 0])


def pca_candidate_rotations(poly_a, poly_b):
    """
    Generate 4 candidate alignment angles (degrees) by aligning the PCA principal
    axes of each polygon.  Four candidates arise from the 180° sign ambiguity of
    PCA axes (0 / 90 / 180 / 270 degree variants).
    """
    delta = np.rad2deg(_pca_polygon(poly_a) - _pca_polygon(poly_b))
    return [(delta + k * 90) % 360 for k in range(4)]


def find_best_alignment_by_rotation(poly_a, poly_b, rotation_angles=None,
                                     use_centroids=True, min_passage_width=0.3):
    """
    Search over rotation angles to find the rigid transform that maximises the
    largest contiguous walkable intersection area.

    Args:
        poly_a: Reference polygon (unchanged).
        poly_b: Polygon to align (rotated + translated).
        rotation_angles: Angles in degrees to try.  If None, uses PCA-derived
            candidates (4) plus a 15° coarse grid, deduplicated.
        use_centroids: Align centroids after each rotation (recommended).
        min_passage_width: Minimum walkable corridor width in metres (default 0.3).

    Returns:
        (affine_params, best_walkable_poly, best_raw_inter_area)
        affine_params is [a, b, d, e, xoff, yoff] for shapely.affinity.affine_transform.
        best_walkable_poly is the filtered intersection polygon for the winning candidate,
        ready to use directly — no need to recompute it after the search.
    """
    if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
        return None, 0.0, 0.0

    try:
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)
    except Exception:
        return None, 0.0, 0.0

    if use_centroids:
        centroid_a = poly_a.centroid
        centroid_b = poly_b.centroid
        poly_a_c = translate(poly_a, xoff=-centroid_a.x, yoff=-centroid_a.y)
        poly_b_c = translate(poly_b, xoff=-centroid_b.x, yoff=-centroid_b.y)
        try:
            if not poly_a_c.is_valid:
                poly_a_c = poly_a_c.buffer(0)
            if not poly_b_c.is_valid:
                poly_b_c = poly_b_c.buffer(0)
        except Exception:
            return None, 0.0, 0.0
    else:
        centroid_a = centroid_b = None
        poly_a_c = poly_a
        poly_b_c = poly_b

    if rotation_angles is None:
        pca_angles = pca_candidate_rotations(poly_a, poly_b)
        seen, rotation_angles = set(), []
        for a in [round(x) % 360 for x in pca_angles] + list(range(0, 360, 15)):
            if a not in seen:
                seen.add(a)
                rotation_angles.append(a)

    best_walkable_area = 0.0
    best_inter_area = 0.0
    best_affine = None
    best_walkable_poly = None

    for angle_deg in rotation_angles:
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        poly_b_rot = rotate(poly_b_c, angle_deg, origin=(0, 0), use_radians=False)

        if use_centroids:
            cx_a = poly_a_c.centroid
            cx_b = poly_b_rot.centroid
            tx, ty = cx_a.x - cx_b.x, cx_a.y - cx_b.y
            poly_b_rot = translate(translate(poly_b_rot, xoff=tx, yoff=ty),
                                   xoff=centroid_a.x, yoff=centroid_a.y)
            rot_cb = np.array([cos_a * centroid_b.x - sin_a * centroid_b.y,
                                sin_a * centroid_b.x + cos_a * centroid_b.y])
            t_final = np.array([centroid_a.x, centroid_a.y]) - rot_cb + np.array([tx, ty])
            affine_candidate = [cos_a, -sin_a, sin_a, cos_a, t_final[0], t_final[1]]
        else:
            poly_b_rot = rotate(poly_b_c, angle_deg, origin=(0, 0), use_radians=False)
            affine_candidate = [cos_a, -sin_a, sin_a, cos_a, 0.0, 0.0]

        if not poly_b_rot.is_valid:
            try:
                poly_b_rot = poly_b_rot.buffer(0)
                if poly_b_rot.is_empty:
                    continue
            except Exception:
                continue

        try:
            inter = poly_a.intersection(poly_b_rot)
            inter_area = inter.area if inter and not inter.is_empty else 0.0
        except Exception:
            try:
                inter = poly_a.buffer(0.001).intersection(poly_b_rot.buffer(0.001))
                inter_area = inter.area if inter and not inter.is_empty else 0.0
            except Exception:
                continue

        walkable_poly = largest_walkable_subpolygon(inter, min_passage_width)
        walkable_area = walkable_poly.area if walkable_poly is not None else 0.0

        if walkable_area > best_walkable_area or (
            walkable_area == best_walkable_area and inter_area > best_inter_area
        ):
            best_walkable_area = walkable_area
            best_inter_area = inter_area
            best_affine = affine_candidate
            best_walkable_poly = walkable_poly

    return best_affine, best_walkable_poly, best_inter_area
