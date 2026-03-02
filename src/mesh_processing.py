import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union


def compute_walkable_polygon(
    mesh_path: str,
    info_semantic_path: str,
    visualize: bool = False,
    debug: bool = False,
):
    """
    Load a semantic mesh + metadata and compute a 2D walkable-area polygon.

    Returns a Shapely Polygon or MultiPolygon in a floor-aligned 2D coordinate frame.
    """
    # --------------------------------------------------
    # Load mesh and semantic metadata
    # --------------------------------------------------
    mesh = trimesh.load(mesh_path)

    # Face-level instance ids
    raw = mesh.metadata["_ply_raw"]["face"]["data"]
    object_ids = raw["object_id"]

    with open(info_semantic_path) as f:
        info_semantic = json.load(f)

    # Map instance id → class_id
    object_to_class = {obj["id"]: obj["class_id"] for obj in info_semantic["objects"]}

    # Convert face object_ids → class_ids
    class_ids = np.array([object_to_class.get(obj_id, -1) for obj_id in object_ids])

    # --------------------------------------------------
    # Extract walkable-surface faces (floor + rug)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Build 2D floor polygon (project to dominant plane)
    # --------------------------------------------------
    floor_vertices = floor_mesh.vertices
    floor_polys = []

    # Determine best projection plane: use the two axes
    # with largest extent (ignore the "thickness" axis).
    mins = floor_vertices.min(axis=0)
    maxs = floor_vertices.max(axis=0)
    extents = maxs - mins
    axis_order = np.argsort(extents)  # ascending
    proj_axes = axis_order[1:]        # two largest extents

    if debug:
        print("Floor vertex mins (x, y, z):", mins)
        print("Floor vertex maxs (x, y, z):", maxs)
        print("Floor vertex extents (x, y, z):", extents)
        print("Using projection axes:", proj_axes)

    for face in floor_mesh.faces:
        tri_verts = floor_vertices[face]
        tri_2d = tri_verts[:, proj_axes].copy()
        tri_2d[:, 0] *= -1  # flip one axis for a nicer orientation
        poly = Polygon(tri_2d)

        if poly.is_valid and poly.area > 1e-8:
            floor_polys.append(poly)

    floor_union = unary_union(floor_polys)
    floor_geom = floor_union.buffer(0)  # clean geometry, may be Polygon or MultiPolygon

    # --------------------------------------------------
    # Build obstacle footprint from mesh geometry per object
    # --------------------------------------------------
    # Any object with these classes is *not* treated as an obstacle.
    # We explicitly treat "rug" as walkable, same as floor.
    IGNORED_OBSTACLE_CLASSES = {"floor", "wall", "ceiling", "rug"}

    obstacle_polys = []

    # Use same notion of "up" as when picking projection axes
    vertical_axis = axis_order[0]
    floor_level = np.median(floor_vertices[:, vertical_axis])

    if debug:
        print(
            "Estimated floor level (vertical axis {}):".format(vertical_axis),
            floor_level,
        )

    for obj in info_semantic["objects"]:
        class_name = obj.get("class_name")
        if class_name in IGNORED_OBSTACLE_CLASSES:
            continue

        obj_id = obj["id"]
        face_idx = np.where(object_ids == obj_id)[0]
        if len(face_idx) == 0:
            continue

        # Get unique vertices for this object
        vert_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        verts3d = mesh.vertices[vert_idx]

        # Filter out objects that float significantly above the floor
        heights = verts3d[:, vertical_axis]
        if heights.min() > floor_level + 0.2:  # e.g. lamps, ceiling fixtures
            continue

        # Project to 2D floor frame
        pts2d = verts3d[:, proj_axes].copy()
        pts2d[:, 0] *= -1

        # Use convex hull of all projected points as obstacle footprint
        hull = MultiPoint(pts2d).convex_hull

        # Skip tiny objects and add a small buffer for safety
        if not hull.is_empty and hull.area > 0.05:
            obstacle_polys.append(hull.buffer(0.02))

    if debug:
        print("Number of obstacle objects:", len(obstacle_polys))

    obstacle_geom = None
    if obstacle_polys:
        obstacle_geom = unary_union(obstacle_polys)

    if obstacle_geom is not None and not obstacle_geom.is_empty:
        walkable_geom = floor_geom.difference(obstacle_geom)
    else:
        walkable_geom = floor_geom

    if debug:
        print("Floor area (m²):", floor_geom.area)
        print("Walkable area (m²):", walkable_geom.area)
        print("Walkable bounds:", walkable_geom.bounds)

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

    # Try to pick a reasonable default scene:
    # prefer room1 in data/, fall back to legacy top-level files.
    candidate_meshes = [
        (os.path.join(data_dir, "mesh_semantic_room1.ply"),
         os.path.join(data_dir, "info_semantic_room1.json")),
        (os.path.join(base_dir, "mesh_semantic.ply"),
         os.path.join(base_dir, "info_semantic.json")),
    ]

    for m_path, i_path in candidate_meshes:
        if os.path.exists(m_path) and os.path.exists(i_path):
            mesh_path = m_path
            info_semantic_path = i_path
            break
    else:
        raise RuntimeError("Could not find a default mesh + info_semantic pair to load.")

    print("Using mesh:", mesh_path)
    print("Using semantics:", info_semantic_path)
    compute_walkable_polygon(mesh_path, info_semantic_path, visualize=True, debug=True)