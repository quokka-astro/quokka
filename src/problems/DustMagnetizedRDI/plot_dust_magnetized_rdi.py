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
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


SNAPSHOT_TAGS = ("t6p2ts0", "t8p3ts0", "t17p0ts0")
FACE_TAGS = ("xface", "yface", "zface")
VISIBLE_FACE_ORDER = ("yface", "xface", "zface")
PROJ_X = np.array([1.0, -0.36])
PROJ_Y = np.array([0.72, 0.42])
PROJ_Z = np.array([0.0, 1.0])
PROJECTION_ORIGIN = np.array([0.0, 0.0, 0.0])


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
    bvec_minus_b0_norm = np.empty((nv, nu))
    dust_overdensity = np.empty((nv, nu))
    u_index = {value: idx for idx, value in enumerate(uvals)}
    v_index = {value: idx for idx, value in enumerate(vvals)}
    for row in rows:
        iu = u_index[float(row["u"])]
        iv = v_index[float(row["v"])]
        bvec_minus_b0_norm[iv, iu] = float(row["bvec_minus_b0_norm"])
        dust_overdensity[iv, iu] = float(row["dust_overdensity"])
    return uvals, vvals, bvec_minus_b0_norm, dust_overdensity


def compute_cell_edges(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        width = 1.0
        return np.array([centers[0] - 0.5 * width, centers[0] + 0.5 * width], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def project_cube_coordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = x - PROJECTION_ORIGIN[0]
    dy = y - PROJECTION_ORIGIN[1]
    dz = z - PROJECTION_ORIGIN[2]
    x2d = (PROJ_X[0] * dx) + (PROJ_Y[0] * dy) + (PROJ_Z[0] * dz)
    y2d = (PROJ_X[1] * dx) + (PROJ_Y[1] * dy) + (PROJ_Z[1] * dz)
    return x2d, y2d


def project_point(x: float, y: float, z: float) -> np.ndarray:
    px, py = project_cube_coordinates(np.array(x), np.array(y), np.array(z))
    return np.array([float(px), float(py)], dtype=float)


def make_projected_face_grid(face_tag: str, uvals: np.ndarray, vvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uedges = compute_cell_edges(uvals)
    vedges = compute_cell_edges(vvals)
    uu, vv = np.meshgrid(uedges, vedges, indexing="xy")

    # The CSVs contain the three visible outer faces of the analog cube:
    # y = y_min (front), x = x_max (right), and z = z_max (top).
    if face_tag == "yface":
        xx = uu
        yy = np.zeros_like(uu)
        zz = vv
    elif face_tag == "xface":
        xx = np.ones_like(uu)
        yy = uu
        zz = vv
    elif face_tag == "zface":
        xx = uu
        yy = vv
        zz = np.ones_like(uu)
    else:
        raise ValueError(f"Unknown face tag '{face_tag}'.")
    return project_cube_coordinates(xx, yy, zz)


def projected_cube_vertices() -> dict[str, np.ndarray]:
    return {
        "000": project_point(0.0, 0.0, 0.0),
        "100": project_point(1.0, 0.0, 0.0),
        "010": project_point(0.0, 1.0, 0.0),
        "110": project_point(1.0, 1.0, 0.0),
        "001": project_point(0.0, 0.0, 1.0),
        "101": project_point(1.0, 0.0, 1.0),
        "011": project_point(0.0, 1.0, 1.0),
        "111": project_point(1.0, 1.0, 1.0),
    }


def draw_projected_cube_outlines(ax: plt.Axes) -> None:
    vertices = projected_cube_vertices()
    visible_faces = (
        ("000", "100", "101", "001", "000"),
        ("100", "110", "111", "101", "100"),
        ("001", "101", "111", "011", "001"),
    )
    for face in visible_faces:
        polygon = np.array([vertices[name] for name in face])
        ax.plot(polygon[:, 0], polygon[:, 1], color="0.12", linewidth=0.85, solid_capstyle="round", solid_joinstyle="round")


def draw_projected_axis_triad(ax: plt.Axes) -> None:
    vertices = projected_cube_vertices()
    axis_specs = (
        (vertices["000"], vertices["100"], "+x", np.array([0.06, -0.06])),
        (vertices["100"], vertices["110"], "+y", np.array([0.06, 0.03])),
        (vertices["000"], vertices["001"], "+z", np.array([-0.06, 0.05])),
    )
    for start, end, label, offset in axis_specs:
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "0.12",
                "linewidth": 0.95,
                "mutation_scale": 10.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        text_position = end + offset
        ax.text(text_position[0], text_position[1], label, fontsize=10, color="0.12")


def configure_projected_cube_axes(ax: plt.Axes) -> None:
    vertices = np.stack(tuple(projected_cube_vertices().values()), axis=0)
    ax.set_xlim(float(np.min(vertices[:, 0]) - 0.14), float(np.max(vertices[:, 0]) + 0.18))
    ax.set_ylim(float(np.min(vertices[:, 1]) - 0.16), float(np.max(vertices[:, 1]) + 0.14))
    ax.set_aspect("equal")
    ax.set_axis_off()


def load_face_payload(data_dir: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for snapshot_tag in SNAPSHOT_TAGS:
        for face_tag in FACE_TAGS:
            rows = read_csv_rows(data_dir / f"dust_magnetized_rdi_{snapshot_tag}_{face_tag}.csv")
            payload[(snapshot_tag, face_tag)] = reshape_face(rows)
    return payload


def magnetic_field_values(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return payload[2]


def dust_density_values(payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return payload[3]


def make_snapshot_norms(
    face_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    value_getter,
    *,
    lower_bound: float | None = None,
) -> dict[str, colors.Normalize]:
    norms: dict[str, colors.Normalize] = {}
    for snapshot_tag in SNAPSHOT_TAGS:
        values = [value_getter(face_payload[(snapshot_tag, face_tag)]) for face_tag in FACE_TAGS]
        vmin = lower_bound if lower_bound is not None else float(min(np.min(arr) for arr in values))
        vmax = float(max(np.max(arr) for arr in values))
        if vmax <= vmin:
            vmax = vmin + max(abs(vmin), 1.0) * 1.0e-12
        norms[snapshot_tag] = colors.Normalize(vmin=vmin, vmax=vmax)
    return norms


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


def render_projected_cube(
    ax: plt.Axes,
    snapshot_faces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    values_getter,
    cmap: str,
    norm: colors.Normalize,
):
    mesh = None
    for face_tag in VISIBLE_FACE_ORDER:
        uvals, vvals, _, _ = snapshot_faces[face_tag]
        xgrid, ygrid = make_projected_face_grid(face_tag, uvals, vvals)
        values = values_getter(snapshot_faces[face_tag])
        mesh = ax.pcolormesh(
            xgrid,
            ygrid,
            values,
            shading="flat",
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
        )
    draw_projected_cube_outlines(ax)
    draw_projected_axis_triad(ax)
    configure_projected_cube_axes(ax)
    return mesh


def make_projected_cube_figure(
    summary: dict[str, str],
    face_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    values_getter,
    norms: dict[str, colors.Normalize],
    cmap: str,
    colorbar_label: str,
    output_name: str,
) -> Path:
    fig = plt.figure(figsize=(7.4, 14.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 0.055))

    for idx, snapshot_tag in enumerate(SNAPSHOT_TAGS):
        ax = fig.add_subplot(grid[idx, 0])
        cax = fig.add_subplot(grid[idx, 1])
        snapshot_faces = {face_tag: face_payload[(snapshot_tag, face_tag)] for face_tag in FACE_TAGS}
        mesh = render_projected_cube(ax, snapshot_faces, values_getter, cmap, norms[snapshot_tag])
        snapshot_time_over_ts0 = get_summary_float(summary, f"snapshot_{snapshot_tag}_time_ts0")
        ax.set_title(rf"$t/t_s^0 = {snapshot_time_over_ts0:.1f}$", pad=6.0)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)

    output_path = output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_fig9(output_dir: Path, summary: dict[str, str], face_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    rho_g0 = get_summary_float(summary, "rho_g0")
    cs0 = get_summary_float(summary, "cs0")
    p0 = rho_g0 * cs0 * cs0
    magnetic_scale = np.sqrt(4.0 * np.pi * p0)
    norms = make_snapshot_norms(face_payload, lambda payload: magnetic_field_values(payload) / magnetic_scale, lower_bound=0.0)
    return make_projected_cube_figure(
        summary,
        face_payload,
        values_getter=lambda payload: magnetic_field_values(payload) / magnetic_scale,
        norms=norms,
        cmap="viridis",
        colorbar_label=r"$|\vec{B}-\vec{B}_0|/\sqrt{4\pi P_0}$",
        output_name=output_dir / "dust_magnetized_rdi_fig9_analog.pdf",
    )


def make_fig9_dust(output_dir: Path, summary: dict[str, str], face_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    norms = make_snapshot_norms(face_payload, dust_density_values)
    return make_projected_cube_figure(
        summary,
        face_payload,
        values_getter=dust_density_values,
        norms=norms,
        cmap="magma",
        colorbar_label=r"$\rho_d/\rho_{d,0}$",
        output_name=output_dir / "dust_magnetized_rdi_fig9_dust_analog.pdf",
    )


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
    summary = read_summary(data_dir / "dust_magnetized_rdi_summary.csv")
    face_payload = load_face_payload(data_dir)

    fig8 = make_fig8(data_dir, output_dir)
    fig9 = make_fig9(output_dir, summary, face_payload)
    fig9_dust = make_fig9_dust(output_dir, summary, face_payload)
    print(fig8)
    print(fig9)
    print(fig9_dust)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
