import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
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
# Print stats
# --------------------------------------------------

print("Floor area (m²):", floor_geom.area)
print("Floor bounds:", floor_geom.bounds)

# --------------------------------------------------
# Visualize
# --------------------------------------------------

plt.figure()

if isinstance(floor_geom, MultiPolygon):
    for poly in floor_geom.geoms:
        x, y = poly.exterior.xy
        plt.plot(x, y)
else:
    x, y = floor_geom.exterior.xy
    plt.plot(x, y)

plt.gca().set_aspect("equal")
plt.title("Floor Footprint")
plt.show()