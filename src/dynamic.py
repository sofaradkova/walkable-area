# src/dynamic.py
"""
Utilities for dynamic obstacle simulation.

Public API:
- make_blob(cx, cy, base_radius, amp1, amp2, n_pts, seed) -> Polygon
- apply_dynamic_obstacle(walkable_poly, blob) -> Polygon
- make_trajectory(poly_a, inter_poly, n_frames, angle_deg) -> list[np.ndarray]
"""
import numpy as np
from shapely.geometry import Polygon


def make_blob(cx, cy, base_radius=0.3, amp1=0.30, amp2=0.15, n_pts=32, seed=42):
    """
    Build an irregular blob centred at (cx, cy).

    Radii are perturbed with two overlaid sine waves at different frequencies
    so the shape is organic but stays roughly base_radius in size.

    Args:
        cx, cy:      World-space centre position.
        base_radius: Mean radius in metres.
        amp1, amp2:  Amplitudes of the two sine perturbations, as fractions of
                     base_radius. Higher values = more deformed.
        n_pts:       Number of polygon vertices.
        seed:        RNG seed — same seed gives the same shape at every position.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    freq1, freq2 = int(rng.integers(2, 5)), int(rng.integers(3, 7))
    phase1, phase2 = rng.uniform(0, 2 * np.pi, 2)
    radii = base_radius * (
        1.0
        + amp1 * np.sin(freq1 * angles + phase1)
        + amp2 * np.sin(freq2 * angles + phase2)
    )
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return Polygon(zip(xs, ys)).buffer(0)


def apply_dynamic_obstacle(walkable_poly, blob):
    """
    Subtract a dynamic obstacle footprint from a walkable polygon.

    Args:
        walkable_poly: Shapely Polygon/MultiPolygon representing the current
                       walkable area.
        blob:          Shapely geometry representing the obstacle footprint
                       in the same coordinate frame.

    Returns:
        Walkable polygon with the obstacle area removed.
    """
    if walkable_poly is None or walkable_poly.is_empty:
        return walkable_poly
    if blob is None or blob.is_empty:
        return walkable_poly
    result = walkable_poly.difference(blob)
    return result if result and not result.is_empty else walkable_poly


def make_trajectory(poly_a, inter_poly, n_frames, angle_deg=0.0):
    """
    Build a straight-line trajectory that passes through the intersection
    centroid at a given angle, spanning most of poly_a's extent.

    Args:
        poly_a:     Room A walkable polygon (defines the traversal space).
        inter_poly: Static walkable intersection (used to anchor the midpoint).
        n_frames:   Number of positions along the trajectory.
        angle_deg:  Direction of travel in degrees (0 = left→right,
                    90 = bottom→top, etc.).

    Returns:
        List of n_frames np.ndarray([x, y]) world-space positions.
    """
    bounds = poly_a.bounds
    minx, miny, maxx, maxy = bounds

    if inter_poly is not None and not inter_poly.is_empty:
        cx, cy = inter_poly.centroid.x, inter_poly.centroid.y
    else:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    half_span = max(maxx - minx, maxy - miny) * 0.45
    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad) * half_span
    dy = np.sin(angle_rad) * half_span

    start = np.array([cx - dx, cy - dy])
    end   = np.array([cx + dx, cy + dy])
    return [start + (end - start) * i / (n_frames - 1) for i in range(n_frames)]
