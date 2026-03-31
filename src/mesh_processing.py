import os
import json
from typing import Optional
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import cv2
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union


def _walkable_from_triangle_union(
    mesh,
    floor_mesh,
    floor_vertices,
    object_ids,
    info_semantic,
    proj_axes,
    axis_order,
    debug,
):
    """Original path: one Shapely polygon per triangle + unary_union (accurate but slow)."""
    floor_polys = []
    for face in floor_mesh.faces:
        tri_verts = floor_vertices[face]
        tri_2d = tri_verts[:, proj_axes].copy()
        tri_2d[:, 0] *= -1
        poly = Polygon(tri_2d)
        if poly.is_valid and poly.area > 1e-8:
            floor_polys.append(poly)

    floor_union = unary_union(floor_polys)
    floor_geom = floor_union.buffer(0)

    IGNORED_OBSTACLE_CLASSES = {"floor", "wall", "ceiling", "rug"}
    obstacle_polys = []
    vertical_axis = axis_order[0]
    floor_level = np.median(floor_vertices[:, vertical_axis])

    if debug:
        print(
            "Estimated floor level (vertical axis {}): {}".format(
                vertical_axis, floor_level
            )
        )

    for obj in info_semantic["objects"]:
        class_name = obj.get("class_name")
        if class_name in IGNORED_OBSTACLE_CLASSES:
            continue

        obj_id = obj["id"]
        face_idx = np.where(object_ids == obj_id)[0]
        if len(face_idx) == 0:
            continue

        vert_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        verts3d = mesh.vertices[vert_idx]

        heights = verts3d[:, vertical_axis]
        if heights.min() > floor_level + 0.2:
            continue

        pts2d = verts3d[:, proj_axes].copy()
        pts2d[:, 0] *= -1

        hull = MultiPoint(pts2d).convex_hull

        if not hull.is_empty and hull.area > 0.05:
            obstacle_polys.append(hull.buffer(0.02))

    if debug:
        print("Number of obstacle objects:", len(obstacle_polys))

    obstacle_geom = unary_union(obstacle_polys) if obstacle_polys else None

    if obstacle_geom is not None and not obstacle_geom.is_empty:
        walkable_geom = floor_geom.difference(obstacle_geom)
    else:
        walkable_geom = floor_geom

    if debug:
        print("Floor area (m²):", floor_geom.area)
        print("Walkable area (m²):", walkable_geom.area)
        print("Walkable bounds:", walkable_geom.bounds)

    return walkable_geom


def _faces_per_object(object_ids):
    out = {}
    o = np.asarray(object_ids)
    for fi in range(len(o)):
        oid = int(o[fi])
        out.setdefault(oid, []).append(fi)
    return out


def _rasterize_triangles_chunked(tris_xy, minx, miny, maxx, maxy, H, W, max_workers: Optional[int]):
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

    if max_workers is None:
        workers = min(8, (os.cpu_count() or 4))
    else:
        workers = max(1, max_workers)
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
    """
    occ: HxW uint8/bool foreground mask (walkable = 1).
    bbox: (minx, miny, maxx, maxy) world coords covering the grid (same convention as rasterize.points_to_occupancy).
    """
    minx, miny, maxx, maxy = bbox
    H, W = occ.shape
    width_m = maxx - minx
    height_m = maxy - miny

    def pix_to_world(pt):
        u, v = float(pt[0]), float(pt[1])
        x = minx + (u / max(1, (W - 1))) * width_m
        y = maxy - (v / max(1, (H - 1))) * height_m
        return (x, y)

    img = ((occ > 0).astype(np.uint8) * 255)
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
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)


def _walkable_from_raster(
    mesh,
    floor_mesh,
    floor_vertices,
    object_ids,
    info_semantic,
    proj_axes,
    axis_order,
    debug,
    raster_resolution,
    raster_margin,
    raster_workers,
):
    tris_xy = floor_mesh.triangles[:, :, proj_axes].astype(np.float64, copy=True)
    tris_xy[:, :, 0] *= -1

    flat = tris_xy.reshape(-1, 2)
    rminx, rminy = flat.min(axis=0)
    rmaxx, rmaxy = flat.max(axis=0)
    minx = float(rminx - raster_margin)
    miny = float(rminy - raster_margin)
    maxx = float(rmaxx + raster_margin)
    maxy = float(rmaxy + raster_margin)
    width_m = maxx - minx
    height_m = maxy - miny

    W = max(3, int(np.ceil(width_m / raster_resolution)))
    H = max(3, int(np.ceil(height_m / raster_resolution)))

    floor_occ = _rasterize_triangles_chunked(
        tris_xy, minx, miny, maxx, maxy, H, W, raster_workers
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    floor_occ = cv2.morphologyEx(floor_occ, cv2.MORPH_CLOSE, kernel, iterations=1)

    IGNORED_OBSTACLE_CLASSES = {"floor", "wall", "ceiling", "rug"}
    vertical_axis = axis_order[0]
    floor_level = np.median(floor_vertices[:, vertical_axis])
    obs_occ = np.zeros((H, W), dtype=np.uint8)

    face_map = _faces_per_object(object_ids)
    dilate_px = max(1, int(round(0.02 / raster_resolution)))

    for obj in info_semantic["objects"]:
        class_name = obj.get("class_name")
        if class_name in IGNORED_OBSTACLE_CLASSES:
            continue

        obj_id = int(obj["id"])
        face_idx = face_map.get(obj_id)
        if not face_idx or len(face_idx) == 0:
            continue

        face_idx = np.asarray(face_idx, dtype=np.intp)
        vert_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        verts3d = mesh.vertices[vert_idx]

        heights = verts3d[:, vertical_axis]
        if heights.min() > floor_level + 0.2:
            continue

        pts2d = verts3d[:, proj_axes].copy().astype(np.float64)
        pts2d[:, 0] *= -1
        if pts2d.shape[0] < 3:
            continue

        hull = cv2.convexHull(pts2d.astype(np.float32).reshape(-1, 1, 2))
        hp = hull.reshape(-1, 2)
        if hp.shape[0] < 3:
            continue
        if cv2.contourArea(hp.astype(np.float32)) < 0.05:
            continue

        us, vs = _world_xy_to_pixel(hp, minx, maxy, W, H, width_m, height_m)
        pts = np.stack([us, vs], axis=1).reshape(-1, 1, 2)
        cv2.fillConvexPoly(obs_occ, pts, 1)

    if dilate_px > 1:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        obs_occ = cv2.dilate(obs_occ, dk, iterations=1)

    walk_occ = ((floor_occ > 0) & (obs_occ == 0)).astype(np.uint8)
    walkable_geom = _binary_mask_to_shapely(walk_occ, (minx, miny, maxx, maxy))

    if debug:
        print(
            "Estimated floor level (vertical axis {}): {}".format(
                vertical_axis, floor_level
            )
        )
        print(
            "Raster grid HxW:",
            H,
            "x",
            W,
            "resolution (m/px):",
            raster_resolution,
        )
        print(
            "Approx floor area from mask (m²):",
            float(np.count_nonzero(floor_occ)) * (width_m * height_m) / max(1, H * W),
        )
        if walkable_geom is not None:
            print("Walkable area (m²):", walkable_geom.area)
            print("Walkable bounds:", walkable_geom.bounds)

    return walkable_geom


def compute_walkable_polygon(
    mesh_path: str,
    info_semantic_path: str,
    visualize: bool = False,
    debug: bool = False,
    *,
    raster_resolution: float = 0.03,
    raster_margin: float = 0.2,
    raster_workers: Optional[int] = None,
    use_triangle_union: bool = False,
):
    """
    Load a semantic mesh + metadata and compute a 2D walkable-area polygon.

    By default uses a **raster + threaded triangle fill** path (much faster than
    unary_union over every floor triangle). Pass ``use_triangle_union=True`` to
    use the original Shapely-per-triangle method.
    """
    mesh = trimesh.load(mesh_path)

    raw = mesh.metadata["_ply_raw"]["face"]["data"]
    object_ids = raw["object_id"]

    with open(info_semantic_path) as f:
        info_semantic = json.load(f)

    object_to_class = {obj["id"]: obj["class_id"] for obj in info_semantic["objects"]}
    class_ids = np.array([object_to_class.get(obj_id, -1) for obj_id in object_ids])

    FLOOR_CLASS = 40
    RUG_CLASS = 98

    walkable_mask = (class_ids == FLOOR_CLASS) | (class_ids == RUG_CLASS)
    floor_face_indices = np.where(walkable_mask)[0]

    if debug:
        print("Walkable class ids (treated as floor):", {FLOOR_CLASS, RUG_CLASS})
        print("Number of walkable faces:", len(floor_face_indices))

    if len(floor_face_indices) == 0:
        raise RuntimeError("No walkable faces (class_id 40/98) found in mesh.")

    floor_mesh = mesh.submesh([floor_face_indices], append=True)
    floor_vertices = floor_mesh.vertices

    mins = floor_vertices.min(axis=0)
    maxs = floor_vertices.max(axis=0)
    extents = maxs - mins
    axis_order = np.argsort(extents)
    proj_axes = axis_order[1:]

    if debug:
        print("Floor vertex mins (x, y, z):", mins)
        print("Floor vertex maxs (x, y, z):", maxs)
        print("Floor vertex extents (x, y, z):", extents)
        print("Using projection axes:", proj_axes)

    if use_triangle_union:
        walkable_geom = _walkable_from_triangle_union(
            mesh,
            floor_mesh,
            floor_vertices,
            object_ids,
            info_semantic,
            proj_axes,
            axis_order,
            debug,
        )
    else:
        walkable_geom = _walkable_from_raster(
            mesh,
            floor_mesh,
            floor_vertices,
            object_ids,
            info_semantic,
            proj_axes,
            axis_order,
            debug,
            raster_resolution,
            raster_margin,
            raster_workers,
        )

    if walkable_geom is None or walkable_geom.is_empty:
        if not use_triangle_union:
            if debug:
                print("Fast raster path failed; falling back to triangle union.")
            walkable_geom = _walkable_from_triangle_union(
                mesh,
                floor_mesh,
                floor_vertices,
                object_ids,
                info_semantic,
                proj_axes,
                axis_order,
                debug,
            )

    if walkable_geom is None or walkable_geom.is_empty:
        raise RuntimeError("Could not build walkable polygon.")

    if visualize:
        plt.figure()
        if isinstance(walkable_geom, MultiPolygon):
            for poly in walkable_geom.geoms:
                x, y = poly.exterior.xy
                plt.plot(x, y, color="blue")
        else:
            x, y = walkable_geom.exterior.xy
            plt.plot(x, y, color="blue")

        plt.gca().set_aspect("equal")
        plt.title("Walkable Area")
        plt.show()

    return walkable_geom


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    candidate_meshes = [
        (
            os.path.join(data_dir, "mesh_semantic_room1.ply"),
            os.path.join(data_dir, "info_semantic_room1.json"),
        ),
        (
            os.path.join(base_dir, "mesh_semantic.ply"),
            os.path.join(base_dir, "info_semantic.json"),
        ),
    ]

    for m_path, i_path in candidate_meshes:
        if os.path.exists(m_path) and os.path.exists(i_path):
            mesh_path = m_path
            info_semantic_path = i_path
            break
    else:
        raise RuntimeError(
            "Could not find a default mesh + info_semantic pair to load."
        )

    print("Using mesh:", mesh_path)
    print("Using semantics:", info_semantic_path)
    compute_walkable_polygon(mesh_path, info_semantic_path, visualize=True, debug=True)
