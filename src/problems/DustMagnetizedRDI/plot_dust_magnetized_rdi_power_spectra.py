#!/usr/bin/env python3

"""Plot dust-density and MHD-mode power spectra for DustMagnetizedRDI."""

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
import matplotlib.pyplot as plt
import numpy as np

SINGLE_COLUMN_WIDTH = 3.4
STAGES = ("linear", "nonlinear", "saturation")
STAGE_COLORS = {"linear": "#9A7200", "nonlinear": "#7132A8", "saturation": "#B44E80"}

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


def read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row["key"]): str(row["value"]) for row in csv.DictReader(handle)}


def load_stage_fields(data_dir: Path, summary: dict[str, str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import yt

    yt.set_log_level(40)
    dust_density = {}
    for stage in STAGES:
        plotfile = data_dir / summary[f"stage_{stage}_plotfile"]
        dataset = yt.load(str(plotfile))
        grid = dataset.covering_grid(level=0, left_edge=dataset.domain_left_edge, dims=dataset.domain_dimensions)
        dust_density[stage] = np.asarray(grid[("boxlib", "dustDensity-Group0")], dtype=float)
        if stage == "saturation":
            gas_density = np.asarray(grid[("boxlib", "gasDensity")], dtype=float)
            gas_velocity = np.stack(
                [
                    np.asarray(grid[("boxlib", f"{component}-GasMomentum")], dtype=float) / gas_density
                    for component in ("x", "y", "z")
                ]
            )
    return dust_density, gas_velocity


def wavevectors(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    components = np.meshgrid(
        *(np.fft.fftfreq(size) * size for size in shape),
        indexing="ij",
    )
    vectors = np.stack(components)
    magnitude = np.sqrt(np.sum(vectors * vectors, axis=0))
    return vectors, magnitude


def shell_average(power: np.ndarray, wavenumber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = min(power.shape) // 2
    shells = np.rint(wavenumber).astype(int)
    counts = np.bincount(shells.ravel(), minlength=maximum + 1)
    summed_power = np.bincount(shells.ravel(), weights=power.ravel(), minlength=maximum + 1)
    k = np.arange(1, maximum + 1)
    return k, summed_power[1 : maximum + 1] / counts[1 : maximum + 1]


def scalar_power_spectrum(field: np.ndarray, wavenumber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.fft.fftn(field)
    power = np.abs(transform) ** 2 / (2.0 * np.pi * field.size)
    return shell_average(power, wavenumber)


def normalize_mode(mode: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.sum(mode * mode, axis=0))
    return mode / np.where(magnitude > 0.0, magnitude, 1.0)


def mhd_mode_power_spectra(
    gas_velocity: np.ndarray,
    k_vector: np.ndarray,
    wavenumber: np.ndarray,
    summary: dict[str, str],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    background_field = np.array([float(summary[key]) for key in ("Bx0", "By0", "Bz0")])
    b_hat = background_field / np.linalg.norm(background_field)
    k_parallel_scalar = np.einsum("i...,i->...", k_vector, b_hat)
    k_parallel = b_hat[:, np.newaxis, np.newaxis, np.newaxis] * k_parallel_scalar
    k_perpendicular = k_vector - k_parallel

    alpha = 0.5 * float(summary["beta"]) * float(summary["gamma"])
    k_squared = wavenumber * wavenumber
    discriminant = (1.0 + alpha) ** 2 - 4.0 * alpha * k_parallel_scalar**2 / np.where(k_squared > 0.0, k_squared, 1.0)
    root = np.sqrt(discriminant)

    alfven = normalize_mode(np.cross(k_vector, b_hat, axisa=0, axisb=0, axisc=0))
    slow = (1.0 + alpha - root) * k_perpendicular + (-1.0 + alpha - root) * k_parallel
    fast = (1.0 + alpha + root) * k_perpendicular + (-1.0 + alpha + root) * k_parallel
    slow = normalize_mode(slow)
    fast = normalize_mode(fast)

    degenerate_slow = (np.sum(slow * slow, axis=0) == 0.0) & (wavenumber > 0.0)
    k_hat = k_vector / np.where(wavenumber > 0.0, wavenumber, 1.0)
    slow_limit = np.cross(alfven, k_hat, axisa=0, axisb=0, axisc=0)
    slow[:, degenerate_slow] = slow_limit[:, degenerate_slow]

    velocity_transform = np.stack([np.fft.fftn(component - np.mean(component)) for component in gas_velocity])
    normalization = 2.0 * np.pi * gas_velocity[0].size
    mode_power = {
        "alfven": np.abs(np.sum(velocity_transform * alfven, axis=0)) ** 2 / normalization,
        "slow": np.abs(np.sum(velocity_transform * slow, axis=0)) ** 2 / normalization,
        "fast": np.abs(np.sum(velocity_transform * fast, axis=0)) ** 2 / normalization,
    }
    total_power = np.sum(np.abs(velocity_transform) ** 2, axis=0) / normalization
    nonzero = wavenumber > 0.0
    closure_error = abs(
        sum(np.sum(power[nonzero]) for power in mode_power.values()) / np.sum(total_power[nonzero]) - 1.0
    )
    k, alfven_spectrum = shell_average(mode_power["alfven"], wavenumber)
    _, slow_spectrum = shell_average(mode_power["slow"], wavenumber)
    _, fast_spectrum = shell_average(mode_power["fast"], wavenumber)
    spectra = {"alfven": alfven_spectrum, "slow": slow_spectrum, "fast": fast_spectrum}
    return k, spectra, closure_error


def add_resolution_axis(ax: plt.Axes, resolution: int) -> None:
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    ticks = np.array([1, 2, 4, 8, 16, 32, 64])
    ticks = ticks[ticks <= resolution // 2]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{resolution // tick:g}" for tick in ticks])
    top.set_xlabel("cells per wavelength")


def power_law_guide(ax: plt.Axes, k: np.ndarray, power: np.ndarray, start: int, stop: int, slope: float, label: str) -> None:
    x = np.array([start, stop], dtype=float)
    anchor = power[np.argmin(np.abs(k - start))]
    y = anchor * (x / start) ** slope
    ax.loglog(x, y, color="black", linewidth=1.0)
    midpoint = np.sqrt(start * stop)
    ax.annotate(
        label,
        xy=(midpoint, anchor * (midpoint / start) ** slope),
        xytext=(2.5, 2.5),
        textcoords="offset points",
        fontsize=8.0,
    )


def make_dust_density_spectrum(
    output_dir: Path,
    dust_density: dict[str, np.ndarray],
    wavenumber: np.ndarray,
) -> Path:
    spectra = {}
    for stage in STAGES:
        k, spectra[stage] = scalar_power_spectrum(dust_density[stage], wavenumber)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.75))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.82)
    for stage in STAGES:
        ax.loglog(k, spectra[stage], color=STAGE_COLORS[stage], label=stage)
    power_law_guide(ax, k, spectra["saturation"], 4, 16, -1.0, r"$k^{-1}$")
    power_law_guide(ax, k, spectra["saturation"], 16, 48, -2.0, r"$k^{-2}$")
    ax.set_xlabel(r"$kL_{\rm box}/(2\pi)$")
    ax.set_ylabel(r"$E_{\rho}(k)$")
    ax.set_xlim(1.0, k[-1])
    ax.legend(loc="best")
    add_resolution_axis(ax, dust_density["linear"].shape[0])

    output = output_dir / "dust_magnetized_rdi_dust_density_spectrum.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def make_mhd_mode_spectra(
    output_dir: Path,
    k: np.ndarray,
    spectra: dict[str, np.ndarray],
    resolution: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.75))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.82)
    ax.loglog(k, spectra["alfven"], color="#0072B2", label=r"$\mathrm{Alfv\acute{e}n}$")
    ax.loglog(k, spectra["slow"], color="#009E73", label="slow")
    ax.loglog(k, spectra["fast"], color="#D55E00", label="fast")
    power_law_guide(ax, k, spectra["alfven"], 8, 32, -2.0, r"$k^{-2}$")
    ax.set_xlabel(r"$kL_{\rm box}/(2\pi)$")
    ax.set_ylabel(r"$E_v(k)$")
    ax.set_xlim(1.0, k[-1])
    ax.legend(loc="best")
    add_resolution_axis(ax, resolution)

    output = output_dir / "dust_magnetized_rdi_mhd_mode_spectra.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing the RDI summary and stage plotfiles.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(data_dir / "dust_magnetized_rdi_summary.csv")
    dust_density, gas_velocity = load_stage_fields(data_dir, summary)
    k_vector, wavenumber = wavevectors(dust_density["linear"].shape)
    density_output = make_dust_density_spectrum(output_dir, dust_density, wavenumber)
    k, mode_spectra, closure_error = mhd_mode_power_spectra(gas_velocity, k_vector, wavenumber, summary)
    mode_output = make_mhd_mode_spectra(output_dir, k, mode_spectra, dust_density["linear"].shape[0])

    print(f"MHD mode projection closure error = {closure_error:.3e}")
    print(density_output)
    print(mode_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
