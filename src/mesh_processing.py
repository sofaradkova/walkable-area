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
