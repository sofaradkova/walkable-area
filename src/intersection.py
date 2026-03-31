# src/intersection.py
"""
Compute, filter, and verify the walkable intersection between two aligned polygons.

Public API:
- largest_walkable_subpolygon(poly, min_passage_width=0.3) -> Polygon | None
- compute_intersection_metrics(poly_a, poly_b_aligned) -> dict
- verify_intersection(metrics, ...) -> dict
"""
from shapely.geometry import MultiPolygon


def largest_walkable_subpolygon(poly, min_passage_width=0.3):
    """
    Return the largest contiguous region of *poly* that is at least
    ``min_passage_width`` wide everywhere (i.e. physically traversable).

    Applies a morphological opening:
      1. Erode by min_passage_width/2  — removes thin slivers and narrow corridors.
      2. Keep only the largest contiguous piece.
      3. Dilate back by min_passage_width/2 — restores original width where possible.

    Returns the cleaned polygon, or None if nothing survives.
    """
    if poly is None or poly.is_empty:
        return None
    r = min_passage_width / 2.0
    eroded = poly.buffer(-r)
    if eroded is None or eroded.is_empty:
        return None
    largest = max(eroded.geoms, key=lambda g: g.area) if isinstance(eroded, MultiPolygon) else eroded
    result = largest.buffer(r)
    return result if result and not result.is_empty else None


def compute_intersection_metrics(poly_a, poly_b_aligned):
    """
    Compute intersection metrics between two aligned polygons.

    Returns a dict with keys:
        intersection, intersection_area, union_area, iou,
        overlap_pct_a, overlap_pct_b, area_a, area_b
    """
    if poly_a is None or poly_b_aligned is None:
        return None

    _zero = {
        'intersection': None, 'intersection_area': 0.0,
        'union_area': poly_a.area if not poly_a.is_empty else 0.0,
        'iou': 0.0, 'overlap_pct_a': 0.0, 'overlap_pct_b': 0.0,
        'area_a': poly_a.area if not poly_a.is_empty else 0.0,
        'area_b': poly_b_aligned.area if not poly_b_aligned.is_empty else 0.0,
    }
    if poly_a.is_empty or poly_b_aligned.is_empty:
        return _zero

    try:
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b_aligned.is_valid:
            poly_b_aligned = poly_b_aligned.buffer(0)
    except Exception:
        return _zero

    try:
        inter = poly_a.intersection(poly_b_aligned)
        inter_area = inter.area if inter and not inter.is_empty else 0.0
        union = poly_a.union(poly_b_aligned)
        union_area = union.area if union and not union.is_empty else poly_a.area
    except Exception:
        try:
            pa = poly_a.buffer(0.001)
            pb = poly_b_aligned.buffer(0.001)
            inter = pa.intersection(pb)
            inter_area = inter.area if inter and not inter.is_empty else 0.0
            union = pa.union(pb)
            union_area = union.area if union and not union.is_empty else pa.area
        except Exception:
            return _zero

    area_a = poly_a.area
    area_b = poly_b_aligned.area
    return {
        'intersection': inter,
        'intersection_area': inter_area,
        'union_area': union_area,
        'iou': inter_area / union_area if union_area > 0 else 0.0,
        'overlap_pct_a': inter_area / area_a * 100 if area_a > 0 else 0.0,
        'overlap_pct_b': inter_area / area_b * 100 if area_b > 0 else 0.0,
        'area_a': area_a,
        'area_b': area_b,
    }


def verify_intersection(metrics,
                        min_iou: float = 0.1,
                        min_intersection_area: float = 0.5,
                        min_overlap_pct: float = 5.0):
    """
    Check whether the walkable intersection is large enough to be meaningful.

    Criteria (all must pass):
    - IoU >= min_iou
    - intersection_area >= min_intersection_area (m²)
    - overlap percentage of either polygon >= min_overlap_pct

    Returns a dict with keys: verified, iou_check, area_check, overlap_check, reason
    """
    if metrics is None:
        return {'verified': False, 'iou_check': False, 'area_check': False,
                'overlap_check': False, 'reason': 'Metrics are None'}

    iou = metrics.get('iou', 0.0)
    inter_area = metrics.get('intersection_area', 0.0)
    ov_a = metrics.get('overlap_pct_a', 0.0)
    ov_b = metrics.get('overlap_pct_b', 0.0)

    iou_check = iou >= min_iou
    area_check = inter_area >= min_intersection_area
    overlap_check = ov_a >= min_overlap_pct or ov_b >= min_overlap_pct
    verified = iou_check and area_check and overlap_check

    if not verified:
        failures = []
        if not iou_check:
            failures.append(f'IoU {iou:.4f} < {min_iou:.4f}')
        if not area_check:
            failures.append(f'Area {inter_area:.2f} m² < {min_intersection_area:.2f} m²')
        if not overlap_check:
            failures.append(f'Overlap {max(ov_a, ov_b):.1f}% < {min_overlap_pct:.1f}%')
        reason = '; '.join(failures)
    else:
        reason = 'All checks passed'

    return {'verified': verified, 'iou_check': iou_check,
            'area_check': area_check, 'overlap_check': overlap_check, 'reason': reason}
