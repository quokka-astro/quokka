#!/usr/bin/env python3
"""Plot pre/post-restart magnetic field lines for the FCQuantities test.

This script expects two plotfiles (or directories containing plotfiles) and
uses the face-centered outputs in fc_vars to build a midplane streamplot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def find_plotfile(path: Path) -> Path:
    if path.is_dir() and (path / "Header").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Plotfile path not found: {path}")
    plotfiles = sorted([p for p in path.iterdir() if p.is_dir() and (p / "Header").is_file()])
    if not plotfiles:
        raise FileNotFoundError(f"No plotfiles found in: {path}")
    return plotfiles[-1]


def find_fc_plotfile(plotfile: Path, dim: str) -> Path:
    fc_dir = plotfile / "fc_vars"
    if not fc_dir.is_dir():
        raise FileNotFoundError(f"No fc_vars directory in plotfile: {plotfile}")
    candidates = sorted([p for p in fc_dir.iterdir() if p.is_dir() and p.name.startswith(dim)])
    if not candidates:
        raise FileNotFoundError(f"No face-centered plotfile for dim '{dim}' in {fc_dir}")
    return candidates[-1]


def load_face_field(fc_plotfile: Path, dim: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import yt

    ds = yt.load(str(fc_plotfile))
    field = next((f for f in ds.field_list if f[1].endswith("BField")), None)
    if field is None:
        raise ValueError(f"No BField found in face-centered plotfile: {fc_plotfile}")

    axis = {"x": 0, "y": 1, "z": 2}.get(dim)
    if axis is None:
        raise ValueError(f"Unknown dim '{dim}'")

    left_edge = np.asarray(ds.domain_left_edge)
    right_edge = np.asarray(ds.domain_right_edge)
    domain_dims = np.asarray(ds.domain_dimensions, dtype=int)
    face_dims = domain_dims.copy()
    face_dims[axis] += 1
    data = np.zeros(tuple(face_dims), dtype=float)

    for grid in ds.index.grids:
        if grid.Level != 0:
            continue
        start = np.asarray(grid.get_global_startindex(), dtype=int)
        grid_data = grid[field].to_ndarray()
        end = start + np.asarray(grid_data.shape, dtype=int)
        data[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = grid_data

    return data, left_edge, right_edge


def plot_field_lines(ax, bx_cc: np.ndarray, by_cc: np.ndarray, left_edge: np.ndarray, right_edge: np.ndarray, title: str) -> None:
    nx, ny, nz = bx_cc.shape
    dx = (right_edge[0] - left_edge[0]) / nx
    dy = (right_edge[1] - left_edge[1]) / ny
    x = left_edge[0] + (np.arange(nx) + 0.5) * dx
    y = left_edge[1] + (np.arange(ny) + 0.5) * dy
    k = nz // 2

    bx = bx_cc[:, :, k]
    by = by_cc[:, :, k]
    ax.streamplot(x, y, bx.T, by.T, color="k", density=1.2, linewidth=0.7, arrowsize=0.7)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def load_bfield_faces(
    plotfile: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fc_x = find_fc_plotfile(plotfile, "x")
    fc_y = find_fc_plotfile(plotfile, "y")
    fc_z = find_fc_plotfile(plotfile, "z")

    bx_face, left_edge, right_edge = load_face_field(fc_x, "x")
    by_face, _, _ = load_face_field(fc_y, "y")
    bz_face, _, _ = load_face_field(fc_z, "z")
    return bx_face, by_face, bz_face, left_edge, right_edge


def infer_cell_dimensions_from_faces(bx_face: np.ndarray, by_face: np.ndarray, bz_face: np.ndarray) -> tuple[int, int, int]:
    nx = bx_face.shape[0] - 1
    ny = by_face.shape[1] - 1
    nz = bz_face.shape[2] - 1
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(f"Face arrays too small for cell dimensions: bx={bx_face.shape}, by={by_face.shape}, bz={bz_face.shape}")

    expected_bx = (nx + 1, ny, nz)
    expected_by = (nx, ny + 1, nz)
    expected_bz = (nx, ny, nz + 1)
    if bx_face.shape != expected_bx or by_face.shape != expected_by or bz_face.shape != expected_bz:
        raise ValueError(
            "Expected N+1 faces per axis: "
            f"bx={bx_face.shape} (expected {expected_bx}), "
            f"by={by_face.shape} (expected {expected_by}), "
            f"bz={bz_face.shape} (expected {expected_bz})"
        )

    return nx, ny, nz


def faces_to_cell_center(
    bx_face: np.ndarray, by_face: np.ndarray, bz_face: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    infer_cell_dimensions_from_faces(bx_face, by_face, bz_face)
    bx_cc = 0.5 * (bx_face[:-1, :, :] + bx_face[1:, :, :])
    by_cc = 0.5 * (by_face[:, :-1, :] + by_face[:, 1:, :])
    bz_cc = 0.5 * (bz_face[:, :, :-1] + bz_face[:, :, 1:])
    return bx_cc, by_cc, bz_cc


def compute_divergence_from_faces(
    bx_face: np.ndarray,
    by_face: np.ndarray,
    bz_face: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = infer_cell_dimensions_from_faces(bx_face, by_face, bz_face)
    dx = (right_edge[0] - left_edge[0]) / nx
    dy = (right_edge[1] - left_edge[1]) / ny
    dz = (right_edge[2] - left_edge[2]) / nz

    dbx_dx = (bx_face[1:, :, :] - bx_face[:-1, :, :]) / dx
    dby_dy = (by_face[:, 1:, :] - by_face[:, :-1, :]) / dy
    dbz_dz = (bz_face[:, :, 1:] - bz_face[:, :, :-1]) / dz
    return dbx_dx + dby_dy + dbz_dz


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pre/post-restart magnetic field lines for FCQuantities.")
    parser.add_argument("--pre", required=True, type=Path, help="Pre-restart plotfile or directory containing plotfiles.")
    parser.add_argument("--post", required=True, type=Path, help="Post-restart plotfile or directory containing plotfiles.")
    parser.add_argument("--output", type=Path, default=Path("tests/fcquantities_bfield_lines.png"), help="Output image path.")
    parser.add_argument("--output-divb", type=Path, default=Path("tests/fcquantities_divB.png"), help="Output div B image path.")
    args = parser.parse_args()

    pre_plotfile = find_plotfile(args.pre)
    post_plotfile = find_plotfile(args.post)

    bx_face_pre, by_face_pre, bz_face_pre, left_edge, right_edge = load_bfield_faces(pre_plotfile)
    bx_face_post, by_face_post, bz_face_post, _, _ = load_bfield_faces(post_plotfile)
    bx_pre, by_pre, bz_pre = faces_to_cell_center(bx_face_pre, by_face_pre, bz_face_pre)
    bx_post, by_post, bz_post = faces_to_cell_center(bx_face_post, by_face_post, bz_face_post)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    plot_field_lines(axes[0], bx_pre, by_pre, left_edge, right_edge, "Pre-restart")
    plot_field_lines(axes[1], bx_post, by_post, left_edge, right_edge, "Post-restart")
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")

    div_pre = compute_divergence_from_faces(bx_face_pre, by_face_pre, bz_face_pre, left_edge, right_edge)
    div_post = compute_divergence_from_faces(bx_face_post, by_face_post, bz_face_post, left_edge, right_edge)
    k = div_pre.shape[2] // 2
    vmax = max(np.abs(div_pre[:, :, k]).max(), np.abs(div_post[:, :, k]).max())
    vmax = vmax if vmax > 0.0 else 1.0

    nx = div_pre.shape[0]
    dx = (right_edge[0] - left_edge[0]) / nx
    bmag_pre = np.sqrt(bx_pre**2 + by_pre**2 + bz_pre**2)
    bmag_post = np.sqrt(bx_post**2 + by_post**2 + bz_post**2)
    mask_pre = bmag_pre > 0.0
    mask_post = bmag_post > 0.0
    ratio_pre = np.zeros_like(div_pre)
    ratio_post = np.zeros_like(div_post)
    ratio_pre[mask_pre] = dx * np.abs(div_pre[mask_pre] / bmag_pre[mask_pre])
    ratio_post[mask_post] = dx * np.abs(div_post[mask_post] / bmag_post[mask_post])
    max_ratio_pre = float(ratio_pre.max())
    max_ratio_post = float(ratio_post.max())
    if max_ratio_post > 3.0 * max_ratio_pre:
        raise ValueError(
            f"Post-restart max dx |divB/B| too large: {max_ratio_post:.3e} > 3x pre {max_ratio_pre:.3e}"
        )
    print(f"max dx |divB/B| pre  = {max_ratio_pre:.6e}")
    print(f"max dx |divB/B| post = {max_ratio_post:.6e}")

    fig_div, axes_div = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    extent = (left_edge[0], right_edge[0], left_edge[1], right_edge[1])
    axes_div[0].imshow(div_pre[:, :, k].T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes_div[0].set_title("Pre-restart div B")
    axes_div[0].set_xlabel("x")
    axes_div[0].set_ylabel("y")
    im1 = axes_div[1].imshow(div_post[:, :, k].T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes_div[1].set_title("Post-restart div B")
    axes_div[1].set_xlabel("x")
    axes_div[1].set_ylabel("y")
    fig_div.colorbar(im1, ax=axes_div, shrink=0.85, label="div B")
    fig_div.savefig(args.output_divb, dpi=200)
    print(f"Wrote {args.output_divb}")


if __name__ == "__main__":
    main()
