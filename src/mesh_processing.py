import trimesh
import os
import numpy as np
import matplotlib.pyplot as plt
import alphashape
import shapely
from shapely.geometry import MultiPolygon

base_dir = os.path.dirname(os.path.dirname(__file__))
mesh_path = os.path.join(base_dir, "mesh_semantic.ply")

mesh = trimesh.load(mesh_path)
#mesh.show()

FLOOR_ID = 40

raw = mesh.metadata["_ply_raw"]["face"]["data"]
object_ids = raw["object_id"]

floor_face_indices = np.where(object_ids == FLOOR_ID)[0]
print("Number of floor faces:", len(floor_face_indices))

floor_mesh = mesh.submesh([floor_face_indices], append=True)
floor_mesh.visual.vertex_colors = [255, 0, 0, 255]
#floor_mesh.show()

floor_vertices = floor_mesh.vertices
floor_2d = floor_vertices[:, :2]

plt.scatter(floor_2d[:, 0], floor_2d[:, 1], s=1)
plt.gca().set_aspect("equal")
#plt.show()

floor_poly = alphashape.alphashape(floor_2d, alpha=0.2)

x, y = floor_poly.exterior.xy
plt.plot(x, y)
plt.gca().set_aspect("equal")
#plt.show()

non_floor_face_indices = np.where(object_ids != FLOOR_ID)[0]

non_floor_mesh = mesh.submesh([non_floor_face_indices], append=True)
non_floor_mesh.visual.vertex_colors = [0, 0, 255, 255]
#non_floor_mesh.show()

floor_z = np.mean(floor_mesh.vertices[:, 2])
print("Floor Z:", floor_z)

unique_ids = np.unique(object_ids)

furniture_polys = []

for obj_id in unique_ids:
    if obj_id==FLOOR_ID:
        continue

    face_idx = np.where(object_ids == obj_id)[0]
    obj_mesh = mesh.submesh([face_idx], append=True)

    verts = obj_mesh.vertices
    z_vals = verts[:, 2]

    FLOOR_TOL = 0.03  # 3 cm

    if np.min(z_vals) < floor_z + FLOOR_TOL:
        bottom_mask = np.abs(z_vals - floor_z) < FLOOR_TOL
        bottom_verts = verts[bottom_mask]

        if len(bottom_verts) > 20:
            footprint = shapely.geometry.MultiPoint(bottom_verts[:, :2]).convex_hull
            furniture_polys.append(footprint)

from shapely.ops import unary_union

furniture_union = unary_union(furniture_polys)

walkable = floor_poly.difference(furniture_union)

if isinstance(walkable, MultiPolygon):
    walkable = max(walkable.geoms, key=lambda p: p.area)

x, y = walkable.exterior.xy
plt.plot(x, y)
plt.gca().set_aspect("equal")
plt.title("Final Walkable Region")
#plt.show()

print("Number of furniture polys:", len(furniture_polys))
print("Furniture union area:", furniture_union.area if not furniture_union.is_empty else 0)
print("Intersection area:",
      floor_poly.intersection(furniture_union).area)