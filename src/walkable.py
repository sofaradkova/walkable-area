# src/walkable.py
"""
Compute a 2D walkable-area polygon from a Replica-format semantic mesh.

Floor triangles are rasterised to a binary occupancy grid, obstacle footprints
(semantic objects touching the floor) are subtracted, and the result is
recovered as a Shapely Polygon/MultiPolygon with interior holes for obstacles.

Public API:
- compute_walkable_polygon(mesh_path, info_semantic_path, ...) -> Polygon
"""
import os
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _faces_per_object(object_ids):
    out = {}
    o = np.asarray(object_ids)
    for fi in range(len(o)):
        oid = int(o[fi])
        out.setdefault(oid, []).append(fi)
    return out


def _rasterize_triangles_chunked(tris_xy, minx, miny, maxx, maxy, H, W,
                                  max_workers: Optional[int]):
    width_m = maxx - minx
    height_m = maxy - miny
    scale_x = (W - 1) / width_m if width_m > 1e-12 else 1.0
    scale_y = (H - 1) / height_m if height_m > 1e-12 else 1.0
    maxy_ref = maxy

    def raster_chunk(tris_chunk):
        occ = np.zeros((H, W), dtype=np.uint8)
        for tri in tris_chunk:
            us = np.round((tri[:, 0] - minx) * scale_x).astype(np.int32)
            vs = np.round((maxy_ref - tri[:, 1]) * scale_y).astype(np.int32)
            pts = np.stack([us, vs], axis=1).reshape(-1, 1, 2)
            cv2.fillConvexPoly(occ, pts, 1)
        return occ

    n = len(tris_xy)
    if n == 0:
        return np.zeros((H, W), dtype=np.uint8)

    workers = min(8, os.cpu_count() or 4) if max_workers is None else max(1, max_workers)
    if workers <= 1 or n < 8000:
        return raster_chunk(tris_xy)

    chunks = np.array_split(tris_xy, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        masks = list(ex.map(raster_chunk, [c for c in chunks if len(c) > 0]))
    if not masks:
        return np.zeros((H, W), dtype=np.uint8)
    out = masks[0].copy()
    for m in masks[1:]:
        np.maximum(out, m, out=out)
    return out


def _world_xy_to_pixel(xy, minx, maxy, W, H, width_m, height_m):
    scale_x = (W - 1) / width_m if width_m > 1e-12 else 1.0
    scale_y = (H - 1) / height_m if height_m > 1e-12 else 1.0
    us = np.round((xy[:, 0] - minx) * scale_x).astype(np.int32)
    vs = np.round((maxy - xy[:, 1]) * scale_y).astype(np.int32)
    return us, vs


def _binary_mask_to_shapely(occ, bbox):
    """Convert a binary occupancy mask to a Shapely geometry (with holes)."""
    minx, miny, maxx, maxy = bbox
    H, W = occ.shape
    width_m = maxx - minx
    height_m = maxy - miny

    def pix_to_world(pt):
        u, v = float(pt[0]), float(pt[1])
        x = minx + (u / max(1, W - 1)) * width_m
        y = maxy - (v / max(1, H - 1)) * height_m
        return (x, y)

    img = (occ > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return None

    hier = hierarchy[0]
    polys = []
    for i in range(len(contours)):
        if hier[i][3] != -1:
            continue
        cnt = contours[i]
        if len(cnt) < 3:
            continue
        shell = [pix_to_world(p) for p in cnt.reshape(-1, 2)]
        hole_rings = []
        c = hier[i][2]
        while c != -1:
            c_cnt = contours[c]
            if len(c_cnt) >= 3:
                hole_rings.append([pix_to_world(p) for p in c_cnt.reshape(-1, 2)])
            c = hier[c][0]
        try:
            g = Polygon(shell, holes=hole_rings if hole_rings else None).buffer(0)
            if not g.is_empty:
                polys.append(g)
        except Exception:
            continue

    if not polys:
        return None
    return polys[0] if len(polys) == 1 else unary_union(polys)


# ---------------------------------------------------------------------------
# Two computation paths
# ---------------------------------------------------------------------------

_IGNORED_OBSTACLE_CLASSES = {"floor", "wall", "ceiling", "rug"}
_FLOOR_CLASS = 40
_RUG_CLASS = 98


def _walkable_from_raster(mesh, floor_mesh, floor_vertices, object_ids,
                           info_semantic, proj_axes, axis_order, debug,
                           raster_resolution, raster_margin, raster_workers):
    tris_xy = floor_mesh.triangles[:, :, proj_axes].astype(np.float64, copy=True)
    tris_xy[:, :, 0] *= -1

    flat = tris_xy.reshape(-1, 2)
    minx = float(flat[:, 0].min()) - raster_margin
    miny = float(flat[:, 1].min()) - raster_margin
    maxx = float(flat[:, 0].max()) + raster_margin
    maxy = float(flat[:, 1].max()) + raster_margin
    width_m = maxx - minx
    height_m = maxy - miny

    W = max(3, int(np.ceil(width_m / raster_resolution)))
    H = max(3, int(np.ceil(height_m / raster_resolution)))

    floor_occ = _rasterize_triangles_chunked(
        tris_xy, minx, miny, maxx, maxy, H, W, raster_workers
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    floor_occ = cv2.morphologyEx(floor_occ, cv2.MORPH_CLOSE, kernel, iterations=1)

    vertical_axis = axis_order[0]
    floor_level = np.median(floor_vertices[:, vertical_axis])
    obs_occ = np.zeros((H, W), dtype=np.uint8)
    face_map = _faces_per_object(object_ids)
    dilate_px = max(1, int(round(0.02 / raster_resolution)))

    for obj in info_semantic["objects"]:
        if obj.get("class_name") in _IGNORED_OBSTACLE_CLASSES:
            continue
        obj_id = int(obj["id"])
        face_idx = face_map.get(obj_id)
        if not face_idx:
            continue
        face_idx = np.asarray(face_idx, dtype=np.intp)
        vert_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        verts3d = mesh.vertices[vert_idx]
        if verts3d[:, vertical_axis].min() > floor_level + 0.2:
            continue
        pts2d = verts3d[:, proj_axes].copy().astype(np.float64)
        pts2d[:, 0] *= -1
        if pts2d.shape[0] < 3:
            continue
        hull = cv2.convexHull(pts2d.astype(np.float32).reshape(-1, 1, 2))
        hp = hull.reshape(-1, 2)
        if hp.shape[0] < 3 or cv2.contourArea(hp.astype(np.float32)) < 0.05:
            continue
        us, vs = _world_xy_to_pixel(hp, minx, maxy, W, H, width_m, height_m)
        cv2.fillConvexPoly(obs_occ, np.stack([us, vs], axis=1).reshape(-1, 1, 2), 1)

    if dilate_px > 1:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        obs_occ = cv2.dilate(obs_occ, dk, iterations=1)

    walk_occ = ((floor_occ > 0) & (obs_occ == 0)).astype(np.uint8)
    walkable_geom = _binary_mask_to_shapely(walk_occ, (minx, miny, maxx, maxy))

    if debug:
        print(f"Floor level (axis {vertical_axis}): {floor_level:.3f}")
        print(f"Raster grid {H}x{W} at {raster_resolution} m/px")
        approx_floor = float(np.count_nonzero(floor_occ)) * (width_m * height_m) / max(1, H * W)
        print(f"Approx floor area: {approx_floor:.2f} m²")
        if walkable_geom is not None:
            print(f"Walkable area: {walkable_geom.area:.2f} m²")

    return walkable_geom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_walkable_polygon(
    mesh_path: str,
    info_semantic_path: str,
    visualize: bool = False,
    debug: bool = False,
    *,
    raster_resolution: float = 0.03,
    raster_margin: float = 0.2,
    raster_workers: Optional[int] = None,
) -> Polygon:
    """
    Load a Replica-format semantic mesh and return a 2D walkable-area polygon.

    Rasterises floor triangles to a binary grid, subtracts obstacle footprints,
    and recovers a Shapely polygon with interior holes where obstacles sit.
    Raises RuntimeError if no walkable polygon can be built.
    """
    mesh = trimesh.load(mesh_path)
    object_ids = mesh.metadata["_ply_raw"]["face"]["data"]["object_id"]

    with open(info_semantic_path) as f:
        info_semantic = json.load(f)

    object_to_class = {obj["id"]: obj["class_id"] for obj in info_semantic["objects"]}
    class_ids = np.array([object_to_class.get(oid, -1) for oid in object_ids])

    walkable_mask = (class_ids == _FLOOR_CLASS) | (class_ids == _RUG_CLASS)
    floor_face_indices = np.where(walkable_mask)[0]

    if debug:
        print(f"Walkable face classes: {_FLOOR_CLASS, _RUG_CLASS}")
        print(f"Walkable faces: {len(floor_face_indices)}")

    if len(floor_face_indices) == 0:
        raise RuntimeError("No walkable faces (class_id 40/98) found in mesh.")

    floor_mesh = mesh.submesh([floor_face_indices], append=True)
    floor_vertices = floor_mesh.vertices

    extents = floor_vertices.max(axis=0) - floor_vertices.min(axis=0)
    axis_order = np.argsort(extents)
    proj_axes = axis_order[1:]

    if debug:
        print(f"Floor extents (x,y,z): {extents}")
        print(f"Projection axes: {proj_axes}")

    walkable_geom = _walkable_from_raster(
        mesh=mesh, floor_mesh=floor_mesh, floor_vertices=floor_vertices,
        object_ids=object_ids, info_semantic=info_semantic,
        proj_axes=proj_axes, axis_order=axis_order, debug=debug,
        raster_resolution=raster_resolution,
        raster_margin=raster_margin,
        raster_workers=raster_workers,
    )

    if walkable_geom is None or walkable_geom.is_empty:
        raise RuntimeError("Could not build walkable polygon from raster path.")

    if visualize:
        plt.figure()
        geoms = walkable_geom.geoms if isinstance(walkable_geom, MultiPolygon) else [walkable_geom]
        for poly in geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color="blue")
        plt.gca().set_aspect("equal")
        plt.title("Walkable Area")
        plt.show()

    return walkable_geom
