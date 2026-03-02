import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union

# --------------------------------------------------
# Paths
# --------------------------------------------------

base_dir = os.path.dirname(os.path.dirname(__file__))

mesh_path = os.path.join(base_dir, "mesh_semantic.ply")
info_semantic_path = os.path.join(base_dir, "info_semantic.json")

# --------------------------------------------------
# Load mesh
# --------------------------------------------------

mesh = trimesh.load(mesh_path)

# Face-level instance ids
raw = mesh.metadata["_ply_raw"]["face"]["data"]
object_ids = raw["object_id"]

# --------------------------------------------------
# Load semantic mapping
# --------------------------------------------------

with open(info_semantic_path) as f:
    info_semantic = json.load(f)

# Map instance id → class_id
object_to_class = {
    obj["id"]: obj["class_id"]
    for obj in info_semantic["objects"]
}

# Convert face object_ids → class_ids
class_ids = np.array([
    object_to_class.get(obj_id, -1)
    for obj_id in object_ids
])

# --------------------------------------------------
# Extract floor faces (class 40 = floor)
# --------------------------------------------------

FLOOR_CLASS = 40

floor_face_indices = np.where(class_ids == FLOOR_CLASS)[0]

print("Number of floor faces:", len(floor_face_indices))

floor_mesh = mesh.submesh([floor_face_indices], append=True)

# --------------------------------------------------
# Build 2D floor polygon (project to XY)
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
print("Floor vertex mins (x, y, z):", mins)
print("Floor vertex maxs (x, y, z):", maxs)
print("Floor vertex extents (x, y, z):", extents)
print("Using projection axes:", proj_axes)

for face in floor_mesh.faces:
    tri_verts = floor_vertices[face]
    tri_2d = tri_verts[:, proj_axes].copy()
    tri_2d[:, 0] *= -1  # project to dominant plane
    poly = Polygon(tri_2d)

    if poly.is_valid and poly.area > 1e-8:
        floor_polys.append(poly)

floor_union = unary_union(floor_polys)
floor_geom = floor_union.buffer(0)  # clean geometry, may be Polygon or MultiPolygon

# --------------------------------------------------
# Build obstacle footprint from mesh geometry per object
# --------------------------------------------------

IGNORED_OBSTACLE_CLASSES = {"floor", "wall", "ceiling"}

obstacle_polys = []

# Use same notion of "up" as when picking projection axes
vertical_axis = axis_order[0]
floor_level = np.median(floor_vertices[:, vertical_axis])
print("Estimated floor level (vertical axis {}):".format(vertical_axis), floor_level)

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

print("Number of obstacle objects:", len(obstacle_polys))

obstacle_geom = None
if obstacle_polys:
    obstacle_geom = unary_union(obstacle_polys)

if obstacle_geom is not None and not obstacle_geom.is_empty:
    walkable_geom = floor_geom.difference(obstacle_geom)
else:
    walkable_geom = floor_geom

# --------------------------------------------------
# Print stats
# --------------------------------------------------

print("Floor area (m²):", floor_geom.area)
print("Walkable area (m²):", walkable_geom.area)
print("Walkable bounds:", walkable_geom.bounds)

# --------------------------------------------------
# Visualize: walkable area only
# --------------------------------------------------

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