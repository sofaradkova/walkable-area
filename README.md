## Real-Time Walkable Space Alignment for Remote VR

[Actively working on this project]

This repository contains a Python pipeline for computing, visualizing, and validating geometric overlap between 3D indoor scenes (rooms) given semantic meshes. It projects 3D floor meshes to 2D walkable-area polygons, samples synthetic floor points, rasterizes them to occupancy grids, and uses image-based features plus geometric search to align pairs of rooms and measure their intersection.

### Installation

1. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   From the repository root:

   ```bash
   pip install -r requirements.txt
   ```

   The pipeline depends on common scientific and geometry libraries such as `numpy`, `shapely`, `matplotlib`, `opencv-python`, and `trimesh` (the latter may be part of your environment even if not explicitly listed).

### Data Requirements

The demo expects semantic meshes and metadata in the `data/` directory:

- **Mesh files**: `mesh_semantic_<room-name>.ply`
- **Info files**: `info_semantic_<room-name>.json`

### Running the Demo

From the repository root, run:

```bash
PYTHONPATH=. python demo/demo_multi_room_tests.py
```

This will:

- Discover all room meshes in `data/` that match the `mesh_semantic_*.ply` / `info_semantic_*.json` pattern.
- Build a set of unordered room pairs.
- For each pair:
  - Compute walkable polygons from both meshes.
  - Sample floor points and rasterize to occupancy grids.
  - Generate thumbnails and detect ORB features.
  - Estimate an alignment transform (feature-based, with rotation search fallbacks).
  - Compute intersection metrics and run verification checks.
  - Save:
    - A comprehensive visualization PNG of the occupancy grids, thumbnails, feature matches, and aligned polygons.
    - A JSON file with transform parameters, areas, IoU, and verification flags.
    - GeoJSON files for the original polygons, aligned polygons, and (if verified) the intersection polygon.

Outputs are written into the `demo_out/` directory. Existing files may be overwritten on subsequent runs.

