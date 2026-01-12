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


def load_face_field(fc_plotfile: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import yt
    ds = yt.load(str(fc_plotfile))
    field = next((f for f in ds.field_list if f[1].endswith("BField")), None)
    if field is None:
        raise ValueError(f"No BField found in face-centered plotfile: {fc_plotfile}")
    
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    data = cg[field].to_ndarray()
    left_edge = np.asarray(ds.domain_left_edge)
    right_edge = np.asarray(ds.domain_right_edge)
    domain_dims = np.asarray(ds.domain_dimensions)
    return data, left_edge, right_edge, domain_dims


def face_to_cell_center(data: np.ndarray, dim: str, domain_dims: np.ndarray) -> np.ndarray:
    if tuple(data.shape) == tuple(domain_dims):
        return data
    expected = domain_dims.copy()
    if dim == "x":
        expected[0] += 1
        if tuple(data.shape) == tuple(expected):
            return 0.5 * (data[:-1, :, :] + data[1:, :, :])
    elif dim == "y":
        expected[1] += 1
        if tuple(data.shape) == tuple(expected):
            return 0.5 * (data[:, :-1, :] + data[:, 1:, :])
    elif dim == "z":
        expected[2] += 1
        if tuple(data.shape) == tuple(expected):
            return 0.5 * (data[:, :, :-1] + data[:, :, 1:])
    else:
        raise ValueError(f"Unknown dim '{dim}'")
    raise ValueError(f"Unexpected face data shape {data.shape} for dim '{dim}' with domain {tuple(domain_dims)}")


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


def load_bfield_components(plotfile: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fc_x = find_fc_plotfile(plotfile, "x")
    fc_y = find_fc_plotfile(plotfile, "y")
    fc_z = find_fc_plotfile(plotfile, "z")

    bx_face, left_edge, right_edge, dims_x = load_face_field(fc_x)
    by_face, _, _, dims_y = load_face_field(fc_y)
    bz_face, _, _, dims_z = load_face_field(fc_z)

    bx_cc = face_to_cell_center(bx_face, "x", dims_x)
    by_cc = face_to_cell_center(by_face, "y", dims_y)
    bz_cc = face_to_cell_center(bz_face, "z", dims_z)
    return bx_cc, by_cc, bz_cc, left_edge, right_edge


def compute_divergence(bx: np.ndarray, by: np.ndarray, bz: np.ndarray, left_edge: np.ndarray, right_edge: np.ndarray) -> np.ndarray:
    nx, ny, nz = bx.shape
    dx = (right_edge[0] - left_edge[0]) / nx
    dy = (right_edge[1] - left_edge[1]) / ny
    dz = (right_edge[2] - left_edge[2]) / nz

    dbx_dx = (np.roll(bx, -1, axis=0) - np.roll(bx, 1, axis=0)) / (2.0 * dx)
    dby_dy = (np.roll(by, -1, axis=1) - np.roll(by, 1, axis=1)) / (2.0 * dy)
    dbz_dz = (np.roll(bz, -1, axis=2) - np.roll(bz, 1, axis=2)) / (2.0 * dz)
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

    bx_pre, by_pre, bz_pre, left_edge, right_edge = load_bfield_components(pre_plotfile)
    bx_post, by_post, bz_post, _, _ = load_bfield_components(post_plotfile)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    plot_field_lines(axes[0], bx_pre, by_pre, left_edge, right_edge, "Pre-restart")
    plot_field_lines(axes[1], bx_post, by_post, left_edge, right_edge, "Post-restart")
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")

    div_pre = compute_divergence(bx_pre, by_pre, bz_pre, left_edge, right_edge)
    div_post = compute_divergence(bx_post, by_post, bz_post, left_edge, right_edge)
    k = div_pre.shape[2] // 2
    vmax = max(np.abs(div_pre[:, :, k]).max(), np.abs(div_post[:, :, k]).max())
    vmax = vmax if vmax > 0.0 else 1.0

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
