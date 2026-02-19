# synthetic_generator.py
"""
Synthetic 2D polygon & point sampling utilities for pipeline unit tests.

Dependencies:
    pip install shapely numpy pillow

APIs:
- make_rectangle(width, height, origin=(0,0))
- make_rectangle_with_hole(width, height, hole_rect, origin=(0,0))
- make_lshape(outer_w, outer_h, cut_w, cut_h, origin=(0,0))
- sample_floor_points(polygon, n_points, seed=None)
- polygon_to_occupancy_image(polygon, resolution=0.05, margin=0.5)

All coordinates are in meters, XY plane. Returned points are Nx3 (x,y,z).
"""
from typing import Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from PIL import Image, ImageDraw

def make_rectangle(width: float, height: float, origin: Tuple[float,float]=(0.0,0.0)) -> Polygon:
    ox, oy = origin
    coords = [(ox, oy), (ox+width, oy), (ox+width, oy+height), (ox, oy+height)]
    return Polygon(coords)

def make_rectangle_with_hole(width: float, height: float, hole_rect: Tuple[float,float,float,float], origin: Tuple[float,float]=(0.0,0.0)) -> Polygon:
    """
    hole_rect: (x_min, y_min, x_max, y_max) relative to origin coordinates.
    """
    outer = make_rectangle(width, height, origin=origin)
    hx0, hy0, hx1, hy1 = hole_rect
    hole = [(origin[0]+hx0, origin[1]+hy0), (origin[0]+hx1, origin[1]+hy0),
            (origin[0]+hx1, origin[1]+hy1), (origin[0]+hx0, origin[1]+hy1)]
    return Polygon(outer.exterior.coords, [hole])

def make_lshape(outer_w: float, outer_h: float, cut_w: float, cut_h: float, origin: Tuple[float,float]=(0.0,0.0)) -> Polygon:
    """
    Create an L-shape by subtracting a rectangle cut from top-right of outer rectangle.
    cut_w, cut_h define the size of the cut rectangle anchored at (origin[0] + outer_w - cut_w, origin[1] + outer_h - cut_h)
    """
    outer = make_rectangle(outer_w, outer_h, origin=origin)
    cx0 = origin[0] + outer_w - cut_w
    cy0 = origin[1] + outer_h - cut_h
    cut = [(cx0, cy0), (cx0 + cut_w, cy0), (cx0 + cut_w, cy0 + cut_h), (cx0, cy0 + cut_h)]
    return Polygon(outer.exterior.coords, [cut])

def sample_floor_points(polygon: Polygon, n_points: int=5000, seed: int=None, z_noise: float=0.01) -> np.ndarray:
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
        grid = [(x,y) for x in xs for y in ys if polygon.contains(Point(x,y))]
        pts = [(x,y,float(rng.normal(0,z_noise))) for (x,y) in grid[:n_points]]
    return np.array(pts, dtype=float)

def polygon_to_occupancy_image(polygon: Polygon, resolution: float=0.05, margin: float=0.5) -> Tuple[np.ndarray, Tuple[float,float,float,float]]:
    """
    Rasterize polygon to a binary occupancy image.
    Returns:
      - occ: HxW numpy uint8 array with 1==occupied, 0==free
      - bbox: (minx, miny, maxx, maxy) in world coordinates of the raster grid
    resolution in meters per pixel, margin in meters around polygon bounds.
    """
    minx, miny, maxx, maxy = polygon.bounds
    minx -= margin; miny -= margin; maxx += margin; maxy += margin
    width_m = maxx - minx; height_m = maxy - miny
    W = max(3, int(np.ceil(width_m / resolution)))
    H = max(3, int(np.ceil(height_m / resolution)))
    # Image coordinate system: origin top-left; we'll draw polygon in pixel coords
    scale_x = W / width_m
    scale_y = H / height_m
    def world_to_pix(x,y):
        u = int((x - minx) * scale_x)
        v = int((maxy - y) * scale_y)  # flip vertical
        # clip to image extents
        u = max(0, min(W-1, u))
        v = max(0, min(H-1, v))
        return (u,v)

    # create blank image and draw polygon
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    exterior = [world_to_pix(x,y) for x,y in polygon.exterior.coords]
    draw.polygon(exterior, outline=1, fill=1)
    # holes
    for hole in polygon.interiors:
        hole_pts = [world_to_pix(x,y) for x,y in hole.coords]
        draw.polygon(hole_pts, outline=0, fill=0)
    occ = (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)
    return occ, (minx, miny, maxx, maxy)
