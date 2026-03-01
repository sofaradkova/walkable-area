import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "scenenn_seg_005.hdf5"

with h5py.File(DATA_FILE, "r") as f:
    points = f["data"][0][:, :3]
    labels = f["label"][0]

# 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

# Downsample for speed
sample_idx = np.random.choice(len(points), 2000, replace=False)

ax.scatter(
    points[sample_idx, 0],
    points[sample_idx, 1],
    points[sample_idx, 2],
    c=labels[sample_idx],
    s=2
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()