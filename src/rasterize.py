# src/rasterize.py
"""
Rasterization utilities: points -> occupancy -> cleaned occupancy -> polygon (world coords).

Functions:
- points_to_occupancy(points_xy, resolution=0.05, margin=0.5): rasterize 2D point samples into a binary
  occupancy grid with world-coordinate bounding box.
- postprocess_occupancy(occ, closing_iters=3, min_component_area_px=None): clean a binary occupancy image
  with morphological closing, hole filling, and optional small-component removal.
- occupancy_to_polygon(occ_bin, bbox, resolution): recover the largest Shapely polygon in world
  coordinates from a binary occupancy grid.
- sample_floor_points(polygon, n_points=5000, seed=None, z_noise=0.01): sample noisy 3D floor points
  uniformly inside a 2D polygon.
"""
from typing import Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.ndimage import binary_closing, binary_fill_holes
import cv2

def points_to_occupancy(points_xy: np.ndarray,
                        resolution: float = 0.05,
                        margin: float = 0.5) -> Tuple[np.ndarray, Tuple[float,float,float,float], float]:
    """
    Convert Nx2 array of XY points (meters) -> binary occupancy image + bbox + resolution.
    Returns:
      occ: HxW uint8 array where 1 = occupied
      bbox: (minx, miny, maxx, maxy) world coords of the image
      resolution: meters/pixel (echoed)
    """
    assert points_xy.ndim == 2 and points_xy.shape[1] >= 2, "points_xy must be Nx2 or Nx>=2 array"
    xs = points_xy[:,0]
    ys = points_xy[:,1]
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    minx -= margin; miny -= margin; maxx += margin; maxy += margin
    width_m = maxx - minx
    height_m = maxy - miny
    W = max(3, int(np.ceil(width_m / resolution)))
    H = max(3, int(np.ceil(height_m / resolution)))
    # avoid degenerate dims
    if W <= 0: W = 3
    if H <= 0: H = 3

    # convert points to pixel indices
    # u = ((x - minx) / width_m) * (W-1)
    # v = ((maxy - y) / height_m) * (H-1)  # flip y
    scale_x = (W-1) / width_m if width_m > 0 else 1.0
    scale_y = (H-1) / height_m if height_m > 0 else 1.0
    u = np.round((points_xy[:,0] - minx) * scale_x).astype(int)
    v = np.round((maxy - points_xy[:,1]) * scale_y).astype(int)
    u = np.clip(u, 0, W-1)
    v = np.clip(v, 0, H-1)
    occ = np.zeros((H, W), dtype=np.uint8)
    occ[v, u] = 1
    return occ, (minx, miny, maxx, maxy), resolution

def postprocess_occupancy(occ: np.ndarray, closing_iters: int = 3, min_component_area_px: Optional[int] = None) -> np.ndarray:
    """
    Clean binary occupancy image:
      - morphological closing (to fill cracks)
      - fill holes
      - optionally remove tiny components by area threshold (in pixels)
    Returns cleaned binary (uint8) image.
    """
    if occ.dtype != bool:
        occ_bool = occ > 0
    else:
        occ_bool = occ
    closed = binary_closing(occ_bool, iterations=closing_iters)
    filled = binary_fill_holes(closed)
    out = (filled > 0).astype(np.uint8)

    # optionally remove small connected components
    if min_component_area_px is not None and min_component_area_px > 0:
        num_labels, labels_im = cv2.connectedComponents(out.astype('uint8'))
        cleaned = np.zeros_like(out)
        for lab in range(1, num_labels):
            mask = labels_im == lab
            if mask.sum() >= min_component_area_px:
                cleaned[mask] = 1
        out = cleaned
    return out

def occupancy_to_polygon(occ_bin: np.ndarray,
                         bbox: Tuple[float,float,float,float],
                         resolution: float) -> Optional[Polygon]:
    """
    Convert binary occupancy -> largest Shapely polygon in world coordinates.
    Arguments:
      occ_bin: HxW binary image (uint8 or bool), origin assumed top-left.
      bbox: (minx, miny, maxx, maxy) world coordinates covered by image.
      resolution: meters/pixel (the rasterization resolution used)
    Returns:
      shapely.geometry.Polygon (or None if no contour found)
    """
    if occ_bin.dtype != np.uint8:
        img = (occ_bin.astype(np.uint8) * 255).copy()
    else:
        img = (occ_bin * 255).copy()

    # find contours (OpenCV expects uint8; RETR_EXTERNAL to get outer contours)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # pick the largest contour by area (in pixels)
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    cnt = contours[idx].squeeze()
    if cnt.ndim != 2 or cnt.shape[0] < 3:
        return None

    H, W = occ_bin.shape
    minx, miny, maxx, maxy = bbox
    width_m = maxx - minx
    height_m = maxy - miny
    # mapping pixel->world:
    # x = minx + (u / (W-1)) * width_m
    # y = maxy - (v / (H-1)) * height_m
    world_pts = []
    for (px, py) in cnt:
        u = float(px)
        v = float(py)
        x = minx + (u / max(1, (W-1))) * width_m
        y = maxy - (v / max(1, (H-1))) * height_m
        world_pts.append((x, y))
    poly = Polygon(world_pts).buffer(0)
    if poly.is_empty:
        return None
    return poly

def sample_floor_points(polygon: Polygon,
                        n_points: int = 5000,
                        seed: int = None,
                        z_noise: float = 0.01) -> np.ndarray:
    """
    Uniformly sample n_points inside the polygon (2D) and return Nx3 array (x, y, z_noise).
    Deterministic with seed.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    # Rejection sampling – fine for simple polygons and moderate n.
    attempts = 0
    while len(pts) < n_points and attempts < n_points * 50:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            z = float(rng.normal(0.0, z_noise))
            pts.append((x, y, z))

    if len(pts) < n_points:
        # If polygon is tiny or sampling failed, tile a grid fallback.
        xs = np.linspace(minx + 1e-6, maxx - 1e-6, int(np.ceil(np.sqrt(n_points))))
        ys = np.linspace(miny + 1e-6, maxy - 1e-6, int(np.ceil(np.sqrt(n_points))))
        grid = [(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))]
        pts = [(x, y, float(rng.normal(0, z_noise))) for (x, y) in grid[:n_points]]

    return np.array(pts, dtype=float)
