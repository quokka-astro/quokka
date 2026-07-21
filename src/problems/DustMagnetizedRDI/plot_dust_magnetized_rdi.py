#!/usr/bin/env python3

"""Plot DustMagnetizedRDI standard deviations, stage cubes, and the full-volume dust PDF."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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

SINGLE_COLUMN_WIDTH = 3.4
DOUBLE_COLUMN_WIDTH = 6.9

_LATEX_AVAILABLE = shutil.which("latex") is not None

plt.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.45,
    "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.25,
    "legend.columnspacing": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    "lines.markerfacecolor": "none",
    "lines.markeredgewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.minor.size": 0.0,
    "ytick.minor.size": 0.0,
    "axes.formatter.use_mathtext": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

if _LATEX_AVAILABLE:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    })
else:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


STAGES = ("linear", "nonlinear", "saturation")
SLICE_TAGS = ("xmax_slice", "ymin_slice", "zmax_slice")
VISIBLE_SLICE_ORDER = ("ymin_slice", "xmax_slice", "zmax_slice")
COMPONENT_COLORS = {"x": "#D55E00", "y": "#009E73", "z": "#0072B2"}
DENSITY_COLORS = {"gas": "#984EA3", "dust": "#A65628"}
STAGE_FILL_COLORS = {"linear": "#F2C94C", "nonlinear": "#9B51E0", "saturation": "#F299C2"}
STAGE_TEXT_COLORS = {"linear": "#9A7200", "nonlinear": "#7132A8", "saturation": "#B44E80"}
REGIME_BOUNDARIES_OVER_TS0 = (5.8, 11.3)

PROJ_X = np.array([1.0, -0.36])
PROJ_Y = np.array([0.72, 0.42])
PROJ_Z = np.array([0.0, 1.0])


def read_table(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def read_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row["key"]): str(row["value"]) for row in csv.DictReader(handle)}


def summary_float(summary: dict[str, str], key: str) -> float:
    if key not in summary:
        raise KeyError(f"Summary CSV is missing required key '{key}'.")
    return float(summary[key])


def stage_metadata(summary: dict[str, str]) -> dict[str, dict[str, float | str]]:
    metadata: dict[str, dict[str, float | str]] = {}
    for stage in STAGES:
        reached_key = f"stage_{stage}_reached"
        if int(summary_float(summary, reached_key)) != 1:
            raise RuntimeError(f"Stage '{stage}' was not reached according to summary key '{reached_key}'.")
        plotfile_key = f"stage_{stage}_plotfile"
        if plotfile_key not in summary or not summary[plotfile_key]:
            raise KeyError(f"Summary CSV is missing a plotfile path for stage '{stage}'.")
        metadata[stage] = {
            "time_over_ts0": summary_float(summary, f"stage_{stage}_actual_time_over_ts0"),
            "plotfile": summary[plotfile_key],
        }
    return metadata


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def growth_guide(time: np.ndarray, values: tuple[np.ndarray, ...], phase_mask: np.ndarray, rate: float) -> np.ndarray:
    stacked_values = np.vstack(values)
    mask = phase_mask[np.newaxis, :] & np.isfinite(stacked_values) & (stacked_values > 0.0)
    intercept = np.median((np.log(stacked_values) - rate * time[np.newaxis, :])[mask])
    return np.exp(intercept + rate * time)


def make_sigma_evolution(
    data_dir: Path,
    output_dir: Path,
    summary: dict[str, str],
) -> Path:
    history = read_table(data_dir / "dust_magnetized_rdi_growth.csv")
    ts0 = summary_float(summary, "equilibrium_stop_time")
    cs0 = summary_float(summary, "cs0")
    b0 = summary_float(summary, "B0")
    crossing_time = summary_float(summary, "box_length_x") / cs0
    x_time = history["t"] / ts0
    code_time = history["t"] / crossing_time

    series = {
        "rho_g": positive(history["sigma_log_rho_g"]),
        "rho_d": positive(history["sigma_log_rho_d"]),
        "vgx": positive(history["sigma_vgx"] / cs0),
        "vgy": positive(history["sigma_vgy"] / cs0),
        "vgz": positive(history["sigma_vgz"] / cs0),
        "vdx": positive(history["sigma_vdx"] / cs0),
        "vdy": positive(history["sigma_vdy"] / cs0),
        "vdz": positive(history["sigma_vdz"] / cs0),
        "bx": positive(history["sigma_bx"] / b0),
        "by": positive(history["sigma_by"] / b0),
        "bz": positive(history["sigma_bz"] / b0),
    }

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.75))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.84)

    handles = {}
    handles["rho_g"], = ax.semilogy(
        x_time,
        series["rho_g"],
        color=DENSITY_COLORS["gas"],
        label=r"$\ln[\rho_{\rm g}/\rho_{{\rm g},0}]$",
    )
    handles["rho_d"], = ax.semilogy(
        x_time,
        series["rho_d"],
        color=DENSITY_COLORS["dust"],
        label=r"$\ln[\rho_{\rm d}/\rho_{{\rm d},0}]$",
    )
    for component in ("x", "y", "z"):
        color = COMPONENT_COLORS[component]
        handles[f"vg{component}"], = ax.semilogy(
            x_time,
            series[f"vg{component}"],
            color=color,
            linestyle="-",
            label=rf"$v_{{g,{component}}}/c_s$",
        )
        handles[f"vd{component}"], = ax.semilogy(
            x_time,
            series[f"vd{component}"],
            color=color,
            linestyle="--",
            label=rf"$v_{{d,{component}}}/c_s$",
        )
        handles[f"b{component}"], = ax.semilogy(
            x_time,
            series[f"b{component}"],
            color=color,
            linestyle=":",
            label=rf"$B_{component}/B_0$",
        )

    boundaries = (
        0.0,
        *REGIME_BOUNDARIES_OVER_TS0,
        float(np.max(x_time)),
    )
    for index, stage in enumerate(STAGES):
        lower, upper = boundaries[index], boundaries[index + 1]
        ax.axvspan(lower, upper, color=STAGE_FILL_COLORS[stage], alpha=0.16, linewidth=0.0, zorder=0)
        ax.text(
            0.5 * (lower + upper),
            1.015,
            stage,
            color=STAGE_TEXT_COLORS[stage],
            fontsize=8.5,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )

    all_series = tuple(series.values())
    # The t in these reference growth laws is code time, in units of L_box / c_s.
    for fit_limits, draw_limits, rate, label, label_height in (
        ((2.0, boundaries[1]), (3.4, 5.4), 1.0, r"$e^t$", 9.0),
        ((boundaries[1], boundaries[2]), (6.5, 9.5), 0.1, r"$e^{0.1t}$", 3.0),
    ):
        fit_mask = (x_time >= fit_limits[0]) & (x_time < fit_limits[1])
        guide = 1.0e-2 * growth_guide(code_time, all_series, fit_mask, rate)
        indices = np.flatnonzero((x_time >= draw_limits[0]) & (x_time <= draw_limits[1]))
        ax.semilogy(x_time[indices], guide[indices], color="black", linestyle="-", linewidth=1.0)
        label_index = indices[len(indices) // 2]
        ax.annotate(
            label,
            xy=(x_time[label_index], guide[label_index]),
            xytext=(3.0, label_height),
            textcoords="offset points",
            fontsize=8.0,
            color="black",
        )

    legend_order = (
        "vdx",
        "vdy",
        "vdz",
        "rho_d",
        "vgx",
        "vgy",
        "vgz",
        "rho_g",
        "bx",
        "by",
        "bz",
    )
    ax.legend(handles=[handles[key] for key in legend_order], loc="lower right", ncol=3, fontsize=7.2)
    ax.set_xlabel(r"$t/t_s^0$")
    ax.set_ylabel("standard deviation")
    ax.set_xlim(boundaries[0], boundaries[-1])

    output = output_dir / "dust_magnetized_rdi_sigma_evolution.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def reshape_slice(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = read_table(path)
    uvals = np.unique(table["u"])
    vvals = np.unique(table["v"])
    iu = np.searchsorted(uvals, table["u"])
    iv = np.searchsorted(vvals, table["v"])
    shape = (vvals.size, uvals.size)
    bdelta = np.empty(shape)
    dust = np.empty(shape)
    bdelta[iv, iu] = table["magnetic_perturbation_magnitude"]
    dust[iv, iu] = table["dust_density_ratio"]
    return uvals, vvals, bdelta, dust


def load_slices(data_dir: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return {
        (stage, slice_tag): reshape_slice(data_dir / f"dust_magnetized_rdi_{stage}_{slice_tag}.csv")
        for stage in STAGES
        for slice_tag in SLICE_TAGS
    }


def cell_edges(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def project_coordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return PROJ_X[0] * x + PROJ_Y[0] * y + PROJ_Z[0] * z, PROJ_X[1] * x + PROJ_Y[1] * y + PROJ_Z[1] * z


def normalize_coordinate(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (values - lo) / (hi - lo)


def slice_coordinate_grid(slice_tag: str, uvals: np.ndarray, vvals: np.ndarray, summary: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(cell_edges(uvals), cell_edges(vvals), indexing="xy")
    xlo, xhi = summary_float(summary, "box_xlo"), summary_float(summary, "box_xhi")
    ylo, yhi = summary_float(summary, "box_ylo"), summary_float(summary, "box_yhi")
    zlo, zhi = summary_float(summary, "box_zlo"), summary_float(summary, "box_zhi")

    if slice_tag == "ymin_slice":
        x = normalize_coordinate(u, xlo, xhi)
        y = np.zeros_like(u)
        z = normalize_coordinate(v, zlo, zhi)
    elif slice_tag == "xmax_slice":
        x = np.ones_like(u)
        y = normalize_coordinate(u, ylo, yhi)
        z = normalize_coordinate(v, zlo, zhi)
    else:
        x = normalize_coordinate(u, xlo, xhi)
        y = normalize_coordinate(v, ylo, yhi)
        z = np.ones_like(u)
    return project_coordinates(x, y, z)


def cube_vertices() -> dict[str, np.ndarray]:
    vertices = {}
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                px, py = project_coordinates(np.array(x), np.array(y), np.array(z))
                vertices[f"{x}{y}{z}"] = np.array([float(px), float(py)])
    return vertices


def decorate_cube(ax: plt.Axes) -> None:
    vertices = cube_vertices()
    for outline in (
        ("000", "100", "101", "001", "000"),
        ("100", "110", "111", "101", "100"),
        ("001", "101", "111", "011", "001"),
    ):
        polygon = np.array([vertices[name] for name in outline])
        ax.plot(polygon[:, 0], polygon[:, 1], color="0.12", linewidth=0.8)

    for start, end, label, offset in (
        ("000", "100", r"$+x$", np.array([0.05, -0.06])),
        ("100", "110", r"$+y$", np.array([0.05, 0.03])),
        ("000", "001", r"$+z$", np.array([-0.06, 0.05])),
    ):
        ax.annotate("", xy=vertices[end], xytext=vertices[start], arrowprops={"arrowstyle": "-|>", "color": "0.12", "linewidth": 0.9})
        position = vertices[end] + offset
        ax.text(position[0], position[1], label)

    points = np.stack(tuple(vertices.values()))
    ax.set_xlim(np.min(points[:, 0]) - 0.14, np.max(points[:, 0]) + 0.18)
    ax.set_ylim(np.min(points[:, 1]) - 0.16, np.max(points[:, 1]) + 0.14)
    ax.set_aspect("equal")
    ax.set_axis_off()


def stage_norms(
    slices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    value_index: int,
    *,
    logarithmic: bool,
) -> dict[str, colors.Normalize]:
    norms = {}
    for stage in STAGES:
        values = [slices[(stage, slice_tag)][value_index] for slice_tag in SLICE_TAGS]
        vmin = min(float(np.min(value)) for value in values)
        vmax = max(float(np.max(value)) for value in values)
        if vmax == vmin:
            vmax = vmin + max(abs(vmin), 1.0) * 1.0e-12
        norms[stage] = colors.LogNorm(vmin=vmin, vmax=vmax) if logarithmic else colors.Normalize(vmin=0.0, vmax=vmax)
    return norms


def draw_cube(
    ax: plt.Axes,
    summary: dict[str, str],
    stage: str,
    slices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    value_index: int,
    cmap: str,
    norm: colors.Normalize,
):
    mesh = None
    for slice_tag in VISIBLE_SLICE_ORDER:
        uvals, vvals, _, _ = slices[(stage, slice_tag)]
        xgrid, ygrid = slice_coordinate_grid(slice_tag, uvals, vvals, summary)
        mesh = ax.pcolormesh(
            xgrid,
            ygrid,
            slices[(stage, slice_tag)][value_index],
            shading="flat",
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
        )
    decorate_cube(ax)
    return mesh


def make_stage_cubes(
    output_dir: Path,
    summary: dict[str, str],
    stages: dict[str, dict[str, float | str]],
    slices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    magnetic_norms = stage_norms(slices, 2, logarithmic=False)
    dust_norms = stage_norms(slices, 3, logarithmic=True)
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 5.0))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=(1.0, 0.018, 1.0, 0.018, 1.0, 0.018),
        left=0.01,
        right=0.90,
        bottom=0.04,
        top=0.88,
        hspace=0.08,
        wspace=0.20,
    )

    for column, stage in enumerate(STAGES):
        top_ax = fig.add_subplot(grid[0, 2 * column])
        bottom_ax = fig.add_subplot(grid[1, 2 * column])
        top_cax = fig.add_subplot(grid[0, 2 * column + 1])
        bottom_cax = fig.add_subplot(grid[1, 2 * column + 1])
        for cax in (top_cax, bottom_cax):
            box = cax.get_position()
            cax.set_position([box.x0 - 0.012, box.y0 + 0.18 * box.height, box.width, 0.64 * box.height])

        magnetic_mesh = draw_cube(top_ax, summary, stage, slices, 2, "viridis", magnetic_norms[stage])
        top_ax.set_title(
            rf"$t={float(stages[stage]['time_over_ts0']):.4g}\,t_s^0$",
            fontsize=8.2,
            y=0.98,
        )
        magnetic_cbar = fig.colorbar(magnetic_mesh, cax=top_cax)
        magnetic_cbar.ax.tick_params(labelsize=6.5)
        if column == len(STAGES) - 1:
            magnetic_cbar.set_label(r"$|\mathbf{B}-\mathbf{B}_0|$")

        dust_mesh = draw_cube(bottom_ax, summary, stage, slices, 3, "turbo", dust_norms[stage])
        dust_cbar = fig.colorbar(dust_mesh, cax=bottom_cax)
        dust_cbar.ax.tick_params(labelsize=6.5)
        if column == len(STAGES) - 1:
            dust_cbar.set_label(r"$\rho_{\rm d}/\rho_{{\rm d},0}$")

    output = output_dir / "dust_magnetized_rdi_stage_cubes.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def resolve_plotfile(data_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def load_dust_density(plotfile: Path) -> np.ndarray:
    if not (plotfile / "Header").is_file():
        raise FileNotFoundError(f"Missing Quokka plotfile: {plotfile}")

    import yt

    yt.set_log_level(40)
    dataset = yt.load(str(plotfile))
    field = next((candidate for candidate in dataset.field_list if candidate[1] == "dustDensity-Group0"), None)
    if field is None:
        raise KeyError(f"Plotfile '{plotfile}' does not contain dustDensity-Group0.")
    grid = dataset.covering_grid(level=0, left_edge=dataset.domain_left_edge, dims=dataset.domain_dimensions)
    return np.asarray(grid[field], dtype=float).ravel()


def compute_dust_pdfs(
    dust_density: dict[str, np.ndarray],
    dust_floor: float,
    bins: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    log_density = {stage: np.log(values / np.mean(values)) for stage, values in dust_density.items()}
    lower = min(float(np.min(values)) for values in log_density.values())
    upper = max(float(np.max(values)) for values in log_density.values())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    edges = np.linspace(lower, upper, bins + 1)
    widths = np.diff(edges)

    pdfs = {}
    integrals = {}
    floor_fractions = {}
    for stage, values in log_density.items():
        counts, _ = np.histogram(values, bins=edges)
        pdfs[stage] = counts / (counts.sum() * widths)
        integrals[stage] = float(np.sum(pdfs[stage] * widths))
        floor_fractions[stage] = float(np.mean(dust_density[stage] <= dust_floor * (1.0 + 1.0e-12)))
    return edges, pdfs, integrals, floor_fractions


def make_dust_density_pdf(
    data_dir: Path,
    output_dir: Path,
    summary: dict[str, str],
    stages: dict[str, dict[str, float | str]],
    bins: int,
) -> Path:
    dust_density = {
        stage: load_dust_density(resolve_plotfile(data_dir, str(stages[stage]["plotfile"])))
        for stage in STAGES
    }
    edges, pdfs, integrals, floor_fractions = compute_dust_pdfs(
        dust_density,
        summary_float(summary, "dust_density_floor"),
        bins,
    )
    pdf_baseline = 0.5 * min(float(np.min(values[values > 0.0])) for values in pdfs.values())
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.7))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.97)
    for stage in STAGES:
        ax.stairs(
            pdfs[stage],
            edges,
            baseline=pdf_baseline,
            fill=True,
            facecolor=STAGE_FILL_COLORS[stage],
            edgecolor="none",
            alpha=0.35,
            linewidth=0.0,
            label=stage,
        )
        print(f"{stage}: PDF integral = {integrals[stage]:.16f}")
        print(f"{stage}: dust-floor cell volume fraction = {floor_fractions[stage]:.16e}")
    ax.set_xlabel(r"$\ln(\rho_{\rm d}/\langle\rho_{\rm d}\rangle)$")
    ax.set_ylabel(r"$\mathrm{PDF}\left(\ln(\rho_{\rm d}/\langle\rho_{\rm d}\rangle)\right)$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=pdf_baseline)
    ax.legend(loc="upper right")

    output = output_dir / "dust_magnetized_rdi_dust_density_pdf.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing RDI CSV files and stage plotfiles.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    parser.add_argument("--pdf-bins", type=int, default=80, help="Number of common bins in the dust-density PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(data_dir / "dust_magnetized_rdi_summary.csv")
    stages = stage_metadata(summary)
    slices = load_slices(data_dir)
    outputs = [
        make_sigma_evolution(data_dir, output_dir, summary),
        make_stage_cubes(output_dir, summary, stages, slices),
        make_dust_density_pdf(data_dir, output_dir, summary, stages, args.pdf_bins),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
