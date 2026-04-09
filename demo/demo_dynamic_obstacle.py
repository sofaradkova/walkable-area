# demo/demo_dynamic_obstacle.py
"""
Synthetic dynamic obstacle simulation.

Loads two rooms, aligns them once, then moves a circular blob (cat-sized) along a
straight-line trajectory through the intersection area of Room A. Shows how the
walkable polygon and intersection polygon change at each timestep as a grid of subplots.

Run:
    PYTHONPATH=. python demo/demo_dynamic_obstacle.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon

from src.walkable import compute_walkable_polygon
from src.alignment import (
    find_best_alignment_by_rotation,
    apply_affine_to_polygon,
    pca_candidate_rotations,
)
from src.dynamic import make_blob, apply_dynamic_obstacle, make_trajectory

OUT = "demo_out"
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

N_FRAMES = 12       # timesteps to simulate
COLS = 4            # columns in the output grid


def _discover_rooms():
    rooms = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("mesh_semantic_") and f.endswith(".ply"):
            stem = f.replace("mesh_semantic_", "").replace(".ply", "")
            info_name = f"info_semantic_{stem}.json"
            if os.path.isfile(os.path.join(DATA_DIR, info_name)):
                rooms.append((f, info_name, stem))
    return rooms




def _plot_poly(ax, poly, color, alpha=0.3, linewidth=1.5, linestyle="-", zorder=2):
    if poly is None or poly.is_empty:
        return
    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for g in geoms:
        if g.exterior is None:
            continue
        x, y = g.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, zorder=zorder)
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder + 1)
        for interior in g.interiors:
            xi, yi = zip(*interior.coords)
            ax.fill(xi, yi, color="white", alpha=1.0, zorder=zorder + 1)
            ax.plot(xi, yi, color=color, linewidth=0.8, linestyle=":", zorder=zorder + 2)


def main():
    rooms = _discover_rooms()
    if len(rooms) < 2:
        print("Need at least 2 rooms (mesh_semantic_*.ply + info_semantic_*.json) in data/")
        return

    room_a_name, room_b_name = rooms[0][2], rooms[1][2]
    print(f"Room A: {room_a_name}")
    print(f"Room B: {room_b_name}")

    print("\nLoading Room A...")
    poly_a = compute_walkable_polygon(
        os.path.join(DATA_DIR, rooms[0][0]),
        os.path.join(DATA_DIR, rooms[0][1]),
    )
    print("Loading Room B...")
    poly_b = compute_walkable_polygon(
        os.path.join(DATA_DIR, rooms[1][0]),
        os.path.join(DATA_DIR, rooms[1][1]),
    )

    # --- Align once ---
    print("\nRunning one-time alignment search...")
    pca_angles = pca_candidate_rotations(poly_a, poly_b)
    seen, rotation_angles = set(), []
    for base in pca_angles:
        for delta in range(-20, 25, 5):
            a = int(round(base + delta)) % 360
            if a not in seen:
                seen.add(a)
                rotation_angles.append(a)
    for a in range(0, 360, 15):
        if a not in seen:
            seen.add(a)
            rotation_angles.append(a)

    best_affine, _, _ = find_best_alignment_by_rotation(
        poly_a, poly_b, rotation_angles=rotation_angles, use_centroids=True
    )
    poly_b_aligned = apply_affine_to_polygon(poly_b, best_affine)

    # Static baseline intersection (no obstacle)
    static_inter_raw = poly_a.intersection(poly_b_aligned)
    static_walkable = largest_walkable_subpolygon(static_inter_raw)
    static_area = static_walkable.area if static_walkable else 0.0
    print(f"Static walkable intersection: {static_area:.2f} m²")

    # --- Trajectory ---
    positions = make_trajectory(poly_a, static_walkable, N_FRAMES, angle_deg=20.0)

    # --- Per-frame computation ---
    print(f"\nSimulating {N_FRAMES} frames...")
    frames = []
    for i, pos in enumerate(positions):
        blob = make_blob(pos[0], pos[1])
        dynamic_a = apply_dynamic_obstacle(poly_a, blob)
        dynamic_walkable = apply_dynamic_obstacle(static_walkable, blob)
        area = dynamic_walkable.area if dynamic_walkable else 0.0
        frames.append({
            "pos": pos,
            "blob": blob,
            "dynamic_a": dynamic_a,
            "dynamic_walkable": dynamic_walkable,
            "area": area,
        })
        print(f"  Frame {i+1:2d}: obstacle at ({pos[0]:.2f}, {pos[1]:.2f})  "
              f"intersection={area:.2f} m²  (Δ={area - static_area:+.2f})")

    # --- Layout: trajectory panel on the left, frame grid on the right ---
    rows = (N_FRAMES + COLS - 1) // COLS
    fig = plt.figure(figsize=(4 + COLS * 3.6, rows * 3.8))

    # Left column: trajectory overview
    ax_traj = fig.add_axes([0.02, 0.08, 0.20, 0.84])

    # Right area: grid of per-frame subplots
    grid_left = 0.25
    grid_width = 0.73
    grid_bottom = 0.08
    grid_height = 0.84
    cell_w = grid_width / COLS
    cell_h = grid_height / rows
    frame_axes = []
    for r in range(rows):
        for c in range(COLS):
            left = grid_left + c * cell_w + 0.01
            bottom = grid_bottom + (rows - 1 - r) * cell_h + 0.01
            ax = fig.add_axes([left, bottom, cell_w - 0.015, cell_h - 0.02])
            frame_axes.append(ax)

    # Shared axis limits from the union of both rooms
    b = poly_a.union(poly_b_aligned).bounds
    pad = 0.4
    xlim = (b[0] - pad, b[2] + pad)
    ylim = (b[1] - pad, b[3] + pad)

    # --- Trajectory panel ---
    _plot_poly(ax_traj, poly_a, color="tab:blue", alpha=0.18, zorder=1)
    _plot_poly(ax_traj, poly_b_aligned, color="tab:green", alpha=0.10, linestyle="--", zorder=1)
    _plot_poly(ax_traj, static_walkable, color="purple", alpha=0.30, zorder=2)
    if static_walkable and not static_walkable.is_empty:
        geoms = static_walkable.geoms if isinstance(static_walkable, MultiPolygon) else [static_walkable]
        for g in geoms:
            x, y = g.exterior.xy
            ax_traj.plot(x, y, color="purple", linewidth=1.0, zorder=3)

    traj_xs = [f["pos"][0] for f in frames]
    traj_ys = [f["pos"][1] for f in frames]
    ax_traj.plot(traj_xs, traj_ys, color="red", linewidth=1.0, zorder=4, alpha=0.5, linestyle="--")
    for i, frame in enumerate(frames):
        _plot_poly(ax_traj, frame["blob"], color="red", alpha=0.25, linewidth=0.8, zorder=5)
        ax_traj.annotate(str(i + 1), xy=(frame["pos"][0], frame["pos"][1]),
                         xytext=(3, 3), textcoords="offset points", fontsize=5, color="darkred",
                         zorder=6)
    ax_traj.set_xlim(xlim)
    ax_traj.set_ylim(ylim)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.2)
    ax_traj.set_title("Trajectory", fontsize=8, fontweight="bold")
    ax_traj.tick_params(labelsize=5)

    # --- Per-frame grid ---
    for i, frame in enumerate(frames):
        ax = frame_axes[i]

        # Room A with obstacle subtracted (blue)
        _plot_poly(ax, frame["dynamic_a"], color="tab:blue", alpha=0.22, zorder=1)
        # Room B aligned (green, dashed outline)
        _plot_poly(ax, poly_b_aligned, color="tab:green", alpha=0.12, linestyle="--", zorder=1)
        # Dynamic walkable intersection (purple)
        _plot_poly(ax, frame["dynamic_walkable"], color="purple", alpha=0.55, zorder=3)
        # Static intersection outline for reference (grey dashed)
        if static_walkable and not static_walkable.is_empty:
            geoms = static_walkable.geoms if isinstance(static_walkable, MultiPolygon) else [static_walkable]
            for g in geoms:
                x, y = g.exterior.xy
                ax.plot(x, y, color="grey", linewidth=0.8, linestyle="--", zorder=2, alpha=0.5)

        delta = frame["area"] - static_area
        ax.set_title(
            f"t={i + 1}   {frame['area']:.2f} m²  (Δ{delta:+.2f})",
            fontsize=7,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=5)

    for i in range(len(frames), len(frame_axes)):
        frame_axes[i].set_visible(False)

    legend_elements = [
        Patch(facecolor="tab:blue",  alpha=0.5, label="Room A (obstacle subtracted)"),
        Patch(facecolor="tab:green", alpha=0.4, label="Room B aligned"),
        Patch(facecolor="purple",    alpha=0.7, label="Walkable intersection"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3,
        fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Dynamic Obstacle Simulation — {room_a_name} vs {room_b_name}   "
        f"(static baseline: {static_area:.2f} m²,  grey dashed = static intersection)",
        fontsize=10, fontweight="bold", y=0.98,
    )

    outpath = os.path.join(OUT, "dynamic_obstacle_simulation.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
