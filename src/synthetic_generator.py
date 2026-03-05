"""
Point-sampling utilities for pipeline tests.

Currently exposes:
- sample_floor_points(polygon, n_points, seed=None)

All coordinates are in meters in the XY plane. Returned points are Nx3 (x, y, z).
"""
from typing import Optional
import numpy as np
from shapely.geometry import Polygon, Point


def sample_floor_points(polygon: Polygon, n_points: int = 5000, seed: int = None, z_noise: float = 0.01) -> np.ndarray:
    """
    Uniformly sample n_points inside the polygon (2D) and return Nx3 array (x,y,z_noise).
    Deterministic with seed.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    # rejection sampling - fine for simple polygons and moderate n
    attempts = 0
    while len(pts) < n_points and attempts < n_points * 50:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            z = float(rng.normal(0.0, z_noise))
            pts.append((x, y, z))
    if len(pts) < n_points:
        # if polygon is tiny or sampling failed, tile a grid fallback
        xs = np.linspace(minx + 1e-6, maxx - 1e-6, int(np.ceil(np.sqrt(n_points))))
        ys = np.linspace(miny + 1e-6, maxy - 1e-6, int(np.ceil(np.sqrt(n_points))))
        grid = [(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))]
        pts = [(x, y, float(rng.normal(0, z_noise))) for (x, y) in grid[:n_points]]
    return np.array(pts, dtype=float)
