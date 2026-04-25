#!/usr/bin/env python3
"""Generate volume-render colormap arrays from Gaussian bumps."""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def _parse_floats(values: Iterable[str]) -> List[float]:
    return [float(value) for value in values]


def _gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    arg = (x - mu) / sigma
    return math.exp(-0.5 * arg * arg)


def _log_gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0 or x <= 0.0 or mu <= 0.0:
        return 0.0
    log_x = math.log10(x)
    log_mu = math.log10(mu)
    arg = (log_x - log_mu) / sigma
    return math.exp(-0.5 * arg * arg)


def _compute_bumps(values: List[float], centers: List[float], sigma: float, log_spaced: bool) -> Tuple[List[List[float]], List[float]]:
    per_bump: List[List[float]] = []
    total = [0.0 for _ in values]
    for mu in centers:
        bump = []
        for idx, x in enumerate(values):
            if log_spaced:
                value = _log_gaussian(x, mu, sigma)
            else:
                value = _gaussian(x, mu, sigma)
            bump.append(value)
            total[idx] += value
        per_bump.append(bump)
    return per_bump, total


def _compute_alpha(total: List[float], alpha_max: float, alpha_floor: float) -> List[float]:
    if not total:
        return []
    peak = max(total)
    if peak <= 0.0:
        return [alpha_floor for _ in total]
    scaled = []
    for val in total:
        alpha = (val / peak) * alpha_max
        alpha = max(alpha_floor, min(1.0, alpha))
        scaled.append(alpha)
    return scaled


def _format_line(prefix: str, values: Iterable[float]) -> str:
    formatted = " ".join(f"{value:.6g}" for value in values)
    return f"{prefix} = {formatted}"


def _generate(values: List[float], cmap_name: str, vmin: float, vmax: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin.")
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    reds: List[float] = []
    greens: List[float] = []
    blues: List[float] = []
    alphas: List[float] = []
    for x in values:
        r, g, b, _ = cmap(norm(x))
        reds.append(r)
        greens.append(g)
        blues.append(b)
        alphas.append(1.0)
    return reds, greens, blues, alphas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", nargs="+", type=float, required=True, help="Gaussian centers in field units.")
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Standard deviation for Gaussian bumps (field units; log10 units when --log-spaced).",
    )
    parser.add_argument("--vmin", type=float, required=True, help="Lower bound for colormap normalization.")
    parser.add_argument("--vmax", type=float, required=True, help="Upper bound for colormap normalization.")
    parser.add_argument("--num-points", type=int, default=64, help="Number of control points to sample.")
    parser.add_argument("--alpha-max", type=float, default=0.8, help="Maximum alpha value after normalization.")
    parser.add_argument("--alpha-floor", type=float, default=0.0, help="Minimum alpha value.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument("--prefix", default="quokka.volrender", help="Input prefix to print.")
    parser.add_argument("--values", nargs="*", help="Explicit values to sample instead of linspace.")
    parser.add_argument(
        "--log-spaced",
        action="store_true",
        help="Sample values logarithmically; bumps become Gaussians in log10(value).",
    )
    parser.add_argument("--plot-file", default="colormap_bumps.png", help="PNG filename for Gaussian bump plot.")
    args = parser.parse_args()

    if args.values:
        values = _parse_floats(args.values)
    else:
        if args.num_points < 2:
            raise ValueError("num-points must be at least 2.")
        if args.log_spaced:
            if args.vmin <= 0.0 or args.vmax <= 0.0:
                raise ValueError("log-spaced sampling requires vmin and vmax to be positive.")
            log_min = math.log10(args.vmin)
            log_max = math.log10(args.vmax)
            step = (log_max - log_min) / (args.num_points - 1)
            values = [10.0 ** (log_min + i * step) for i in range(args.num_points)]
        else:
            step = (args.vmax - args.vmin) / (args.num_points - 1)
            values = [args.vmin + i * step for i in range(args.num_points)]

    reds, greens, blues, _ = _generate(values, args.cmap, args.vmin, args.vmax)
    per_bump, total = _compute_bumps(values, args.centers, args.sigma, args.log_spaced)
    alphas = _compute_alpha(total, args.alpha_max, args.alpha_floor)

    prefix = args.prefix.rstrip(".")
    print(_format_line(f"{prefix}.color_map_values", values))
    print(_format_line(f"{prefix}.color_map_r", reds))
    print(_format_line(f"{prefix}.color_map_g", greens))
    print(_format_line(f"{prefix}.color_map_b", blues))
    print(_format_line(f"{prefix}.color_map_a", alphas))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, bump in enumerate(per_bump):
        ax.plot(values, bump, linestyle="--", alpha=0.6, label=f"bump {idx + 1}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Gaussian amplitude")
    ax.set_title("Gaussian bumps")
    if args.log_spaced:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")

    ax_alpha = ax.twinx()
    ax_alpha.plot(values, alphas, color="tab:orange", linewidth=2.0, label="alpha")
    ax_alpha.set_ylabel("Alpha")
    ax_alpha.set_ylim(0.0, 1.05)
    ax_alpha.tick_params(axis="y", labelcolor="tab:orange")
    ax_alpha.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    fig.savefig(args.plot_file, dpi=150)


if __name__ == "__main__":
    main()
