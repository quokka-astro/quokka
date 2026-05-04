#!/usr/bin/env python3

"""Post-process DustMagnetizedRDI CSV diagnostics into figure PDFs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "quokka-matplotlib-cache"
_mpl_config_dir = _cache_root / "mplconfig"
_xdg_cache_dir = _cache_root / "xdg-cache"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_dir))

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np


SNAPSHOT_TAGS = ("t6p2ts0", "t8p3ts0", "t17p0ts0")
FACE_TAGS = ("xface", "yface", "zface")


def read_csv_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def read_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary[str(row["key"])] = str(row["value"])
    return summary


def reshape_face(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uvals = np.array(sorted({float(row["u"]) for row in rows}))
    vvals = np.array(sorted({float(row["v"]) for row in rows}))
    nu = uvals.size
    nv = vvals.size
    bmag_minus_b0 = np.empty((nv, nu))
    dust = np.empty((nv, nu))
    u_index = {value: idx for idx, value in enumerate(uvals)}
    v_index = {value: idx for idx, value in enumerate(vvals)}
    for row in rows:
        iu = u_index[float(row["u"])]
        iv = v_index[float(row["v"])]
        bmag_minus_b0[iv, iu] = float(row["bmag_minus_b0"])
        dust[iv, iu] = float(row["dust_overdensity"])
    return uvals, vvals, bmag_minus_b0, dust


def build_growth_guide(
    t_code: np.ndarray,
    t_over_ts0: np.ndarray,
    series: np.ndarray,
    growth_rate: float,
    fit_window_over_ts0: tuple[float, float],
) -> np.ndarray:
    lo, hi = fit_window_over_ts0
    positive = np.isfinite(series) & (series > 0.0)
    fit_mask = positive & (t_over_ts0 >= lo) & (t_over_ts0 <= hi)
    if np.count_nonzero(fit_mask) == 0:
        fit_mask = positive
    if np.count_nonzero(fit_mask) == 0:
        return np.full_like(t_code, 1.0e-12, dtype=float)
    intercept = float(np.median(np.log(series[fit_mask]) - growth_rate * t_code[fit_mask]))
    return np.exp(intercept + growth_rate * t_code)


def get_summary_float(summary: dict[str, str], key: str) -> float:
    return float(summary[key])


def make_fig8(data_dir: Path, output_dir: Path) -> Path:
    growth_rows = read_csv_rows(data_dir / "dust_magnetized_rdi_growth.csv")
    summary = read_summary(data_dir / "dust_magnetized_rdi_summary.csv")
    ts0 = get_summary_float(summary, "equilibrium_stop_time")
    cs0 = get_summary_float(summary, "cs0")
    b0 = np.sqrt(
        get_summary_float(summary, "Bx0") ** 2
        + get_summary_float(summary, "By0") ** 2
        + get_summary_float(summary, "Bz0") ** 2
    )

    t_code = np.array([float(row["t"]) for row in growth_rows])
    t_over_ts0 = t_code / ts0

    sig_log_rho_g = np.array([float(row["sigma_log_rho_g"]) for row in growth_rows])
    sig_log_rho_d = np.array([float(row["sigma_log_rho_d"]) for row in growth_rows])
    sig_vgx = np.array([float(row["sigma_vgx"]) for row in growth_rows]) / cs0
    sig_vgy = np.array([float(row["sigma_vgy"]) for row in growth_rows]) / cs0
    sig_vgz = np.array([float(row["sigma_vgz"]) for row in growth_rows]) / cs0
    sig_vdx = np.array([float(row["sigma_vdx"]) for row in growth_rows]) / cs0
    sig_vdy = np.array([float(row["sigma_vdy"]) for row in growth_rows]) / cs0
    sig_vdz = np.array([float(row["sigma_vdz"]) for row in growth_rows]) / cs0
    sig_bx = np.array([float(row["sigma_bx"]) for row in growth_rows]) / b0
    sig_by = np.array([float(row["sigma_by"]) for row in growth_rows]) / b0
    sig_bz = np.array([float(row["sigma_bz"]) for row in growth_rows]) / b0

    guide_fast = build_growth_guide(t_code, t_over_ts0, sig_log_rho_d, growth_rate=1.0, fit_window_over_ts0=(8.0, 12.0))
    guide_slow = build_growth_guide(t_code, t_over_ts0, sig_log_rho_g, growth_rate=0.1, fit_window_over_ts0=(8.0, 12.0))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True, constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(t_over_ts0, sig_log_rho_g, color="red", linewidth=1.2, label=r"$\rho_g$")
    ax.semilogy(t_over_ts0, sig_log_rho_d, color="black", linewidth=1.2, label=r"$\rho_d$")
    ax.semilogy(t_over_ts0, guide_fast, color="0.55", linestyle="--", linewidth=1.0, label=r"$\propto e^{t}$")
    ax.semilogy(t_over_ts0, guide_slow, color="0.65", linestyle=":", linewidth=1.0, label=r"$\propto e^{0.1t}$")
    ax.set_ylabel(r"$\sigma(\log(\rho/\rho_0))$")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    ax.semilogy(t_over_ts0, sig_vgx, color="tab:red", linewidth=1.15, label=r"$u_{g,x}$")
    ax.semilogy(t_over_ts0, sig_vgy, color="tab:orange", linewidth=1.15, label=r"$u_{g,y}$")
    ax.semilogy(t_over_ts0, sig_vgz, color="tab:brown", linewidth=1.15, label=r"$u_{g,z}$")
    ax.semilogy(t_over_ts0, guide_fast, color="0.55", linestyle="--", linewidth=1.0)
    ax.semilogy(t_over_ts0, guide_slow, color="0.65", linestyle=":", linewidth=1.0)
    ax.set_ylabel(r"$\sigma(u_g/c_s)$")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    ax.semilogy(t_over_ts0, sig_vdx, color="black", linewidth=1.15, label=r"$v_{d,x}$")
    ax.semilogy(t_over_ts0, sig_vdy, color="0.35", linewidth=1.15, label=r"$v_{d,y}$")
    ax.semilogy(t_over_ts0, sig_vdz, color="0.6", linewidth=1.15, label=r"$v_{d,z}$")
    ax.semilogy(t_over_ts0, guide_fast, color="0.55", linestyle="--", linewidth=1.0)
    ax.semilogy(t_over_ts0, guide_slow, color="0.65", linestyle=":", linewidth=1.0)
    ax.set_ylabel(r"$\sigma(v_d/c_s)$")
    ax.set_xlabel(r"$t/t_s^0$")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    ax.semilogy(t_over_ts0, sig_bx, color="tab:blue", linewidth=1.15, label=r"$B_x$")
    ax.semilogy(t_over_ts0, sig_by, color="tab:cyan", linewidth=1.15, label=r"$B_y$")
    ax.semilogy(t_over_ts0, sig_bz, color="tab:green", linewidth=1.15, label=r"$B_z$")
    ax.semilogy(t_over_ts0, guide_fast, color="0.55", linestyle="--", linewidth=1.0)
    ax.semilogy(t_over_ts0, guide_slow, color="0.65", linestyle=":", linewidth=1.0)
    ax.set_ylabel(r"$\sigma(B/B_0)$")
    ax.set_xlabel(r"$t/t_s^0$")
    ax.legend(frameon=False, loc="upper left")

    output_path = output_dir / "dust_magnetized_rdi_fig8_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def draw_cube_edges(ax: plt.Axes) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for start, end in edges:
        xs = [vertices[start, 0], vertices[end, 0]]
        ys = [vertices[start, 1], vertices[end, 1]]
        zs = [vertices[start, 2], vertices[end, 2]]
        ax.plot(xs, ys, zs, color="0.15", linewidth=0.9, alpha=0.9)


def make_face_coordinates(face_tag: str, uvals: np.ndarray, vvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(uvals, vvals, indexing="xy")
    if face_tag == "xface":
        xx = np.zeros_like(uu)
        yy = uu
        zz = vv
    elif face_tag == "yface":
        xx = uu
        yy = np.zeros_like(uu)
        zz = vv
    else:
        xx = uu
        yy = vv
        zz = np.zeros_like(uu)
    return xx, yy, zz


def plot_face_dust_contours(
    ax: plt.Axes,
    face_tag: str,
    uvals: np.ndarray,
    vvals: np.ndarray,
    dust: np.ndarray,
) -> None:
    dust_levels = [1.2, 1.5, 2.0]
    valid_levels = [level for level in dust_levels if np.nanmax(dust) >= level]
    if not valid_levels:
        return

    uu, vv = np.meshgrid(uvals, vvals, indexing="xy")
    contour_offset = -0.02
    if face_tag == "xface":
        xdata = uu
        ydata = vv
        zdir = "x"
    elif face_tag == "yface":
        xdata = uu
        ydata = vv
        zdir = "y"
    else:
        xdata = uu
        ydata = vv
        zdir = "z"

    ax.contour(xdata, ydata, dust, zdir=zdir, offset=contour_offset, levels=valid_levels, colors="black", linewidths=0.75, linestyles="-")


def plot_cube_face(
    ax: plt.Axes,
    face_tag: str,
    uvals: np.ndarray,
    vvals: np.ndarray,
    bfield_color_norm: np.ndarray,
    dust: np.ndarray,
    mapper: cm.ScalarMappable,
) -> None:
    xx, yy, zz = make_face_coordinates(face_tag, uvals, vvals)
    facecolors = mapper.to_rgba(bfield_color_norm)
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=facecolors, shade=False, linewidth=0.0, antialiased=False)
    plot_face_dust_contours(ax, face_tag, uvals, vvals, dust)


def make_fig9(data_dir: Path, output_dir: Path) -> Path:
    summary = read_summary(data_dir / "dust_magnetized_rdi_summary.csv")
    rho_g0 = get_summary_float(summary, "rho_g0")
    cs0 = get_summary_float(summary, "cs0")
    p0 = rho_g0 * cs0 * cs0
    magnetic_scale = np.sqrt(4.0 * np.pi * p0)
    face_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    mappers: dict[str, cm.ScalarMappable] = {}
    for snapshot_tag in SNAPSHOT_TAGS:
        max_abs_bfield = 0.0
        for face_tag in FACE_TAGS:
            rows = read_csv_rows(data_dir / f"dust_magnetized_rdi_{snapshot_tag}_{face_tag}.csv")
            payload = reshape_face(rows)
            face_payload[(snapshot_tag, face_tag)] = payload
            max_abs_bfield = max(max_abs_bfield, float(np.max(np.abs(payload[2])) / magnetic_scale))
        vmax = max(max_abs_bfield, 1.0e-12)
        mappers[snapshot_tag] = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=vmax), cmap="viridis")

    fig = plt.figure(figsize=(7.2, 14.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 0.055))

    for idx, snapshot_tag in enumerate(SNAPSHOT_TAGS):
        ax = fig.add_subplot(grid[idx, 0], projection="3d")
        cax = fig.add_subplot(grid[idx, 1])
        mapper = mappers[snapshot_tag]
        for face_tag in FACE_TAGS:
            uvals, vvals, bmag_minus_b0, dust = face_payload[(snapshot_tag, face_tag)]
            plot_cube_face(ax, face_tag, uvals, vvals, np.abs(bmag_minus_b0) / magnetic_scale, dust, mapper)

        draw_cube_edges(ax)
        snapshot_time_over_ts0 = get_summary_float(summary, f"snapshot_{snapshot_tag}_time_ts0")
        ax.set_title(rf"$t/t_s^0 = {snapshot_time_over_ts0:.1f}$", pad=6.0)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlim(1.0, -0.03)
        ax.set_ylim(1.0, -0.03)
        ax.set_zlim(1.0, -0.03)
        ax.view_init(elev=22.0, azim=35.0)
        ax.set_axis_off()
        ax.text(1.05, 0.0, 0.0, "x", fontsize=10)
        ax.text(0.0, 1.05, 0.0, "y", fontsize=10)
        ax.text(0.0, 0.0, 1.05, "z", fontsize=10)
        cbar = fig.colorbar(mapper, cax=cax)
        cbar.set_label(r"$\bigl||B|-B_0\bigr|/\sqrt{4\pi P_0}$")

    output_path = output_dir / "dust_magnetized_rdi_fig9_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dust_magnetized_rdi_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig8 = make_fig8(data_dir, output_dir)
    fig9 = make_fig9(data_dir, output_dir)
    print(fig8)
    print(fig9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
