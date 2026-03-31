# Real-Time Walkable Space Alignment for Remote VR

[Actively working on this project]

This repository contains a Python pipeline that computes, aligns, and validates walkable overlap between pairs of 3D indoor scenes from semantic meshes. Each room is converted to a 2D walkable polygon (with obstacle holes), aligned with a rigid transform, and evaluated by physically traversable overlap.

## Pipeline

For each pair of rooms:

1. **Build walkable polygons**  
   Load semantic mesh + metadata (`.ply` + `.json`) and compute a 2D walkable Shapely polygon:
   - Floor classes: `40` (floor) and `98` (rug)
   - Obstacles are detected from semantic objects that touch floor level
   - Obstacles are subtracted, producing polygons with interior holes where needed

2. **Search alignment (rotation + translation)**  
   Use PCA-seeded rotation search to align room B to room A:
   - 4 PCA candidate angles
   - Fine search around each PCA angle (±20 degrees in 5-degree steps)
   - Additional coarse sweep (15-degree grid)
   - Candidate scoring maximizes the largest contiguous walkable overlap

3. **Keep only traversable intersection**  
   Apply morphological opening to the raw intersection:
   - Erode by `min_passage_width / 2`
   - Keep the largest connected component
   - Dilate back

   This removes thin slivers and narrow passages that are geometrically overlapping but not practically walkable.

4. **Verify overlap quality**  
   Mark results as verified only if all checks pass:
   - `IoU >= 0.1`
   - `intersection_area >= 0.5 m²`
   - overlap of either room `>= 5%`

5. **Export outputs**  
   For each room pair:
   - `test_XX_<pair>.png` polygon-only visualization (before, per-room intersection views, and after-alignment overlay)
   - `test_XX_<pair>_results.json` (affine transform, metrics, verification)
   - Verified-only GeoJSON exports:
     - `_room_a.geojson`
     - `_room_b_original.geojson`
     - `_room_b_aligned.geojson`
     - `_intersection_verified.geojson`

## Source Modules

| File | Purpose |
|---|---|
| `src/walkable.py` | Semantic mesh -> 2D walkable polygon (fast raster path + triangle-union fallback) |
| `src/alignment.py` | PCA candidate rotations + rigid alignment search using walkable-area scoring |
| `src/intersection.py` | Intersection filtering (largest walkable subpolygon), metrics, and verification |

## Demo

### Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Data

Place semantic meshes and metadata in `data/` using:

- `mesh_semantic_<room-name>.ply`
- `info_semantic_<room-name>.json`

Compatible meshes can be downloaded from the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset). The demo automatically tests all unordered room pairs found in `data/`.

### Run

```bash
PYTHONPATH=. python demo/demo_multi_room_tests.py
```

Results are written to `demo_out/`.
