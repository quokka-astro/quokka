#!/usr/bin/env python3
"""
Analyze wave convergence data (hydro or MHD).

This utility reads the CSV produced by convergence tests (e.g.,
`test_hydro_wave_convergence` or `test_alfven_wave_convergence`),
computes observed convergence orders, fits a power-law-with-floor model
in terms of cells per wavelength, and generates a log-log plot of the
error norm versus resolution.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

@dataclass
class ConvergenceDatum:
    nx: int
    dx: float
    error: float


@dataclass
class FitResult:
    amplitude: float
    order: float
    floor: float
    sse: float


def read_convergence_csv(path: Path) -> List[ConvergenceDatum]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find convergence file: {path}")

    data: List[ConvergenceDatum] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data.append(
                ConvergenceDatum(
                    nx=int(float(row["nx"])),
                    dx=float(row["dx"]),
                    error=float(row["error"]),
                )
            )

    if not data:
        raise ValueError(f"No rows found in {path}")

    # Sort from coarsest to finest resolution (largest dx to smallest).
    data.sort(key=lambda item: item.dx, reverse=True)
    return data


def linear_least_squares(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Return slope and intercept for a simple linear least-squares fit."""
    if len(xs) != len(ys):
        raise ValueError("Input sequences must have equal length.")
    n = len(xs)
    if n < 2:
        raise ValueError("At least two points are required for linear regression.")

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denominator = n * sum_xx - sum_x * sum_x
    if math.isclose(denominator, 0.0):
        raise ValueError("Degenerate linear regression problem (denominator ~ 0).")

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def fit_power_law_with_floor(data: Sequence[ConvergenceDatum]) -> FitResult:
    nx_values = [item.nx for item in data]
    error_values = [item.error for item in data]
    min_error = min(error_values)

    best: FitResult | None = None

    def search_floor_candidates(candidates: Iterable[float], current_best: FitResult | None) -> FitResult | None:
        best_local = current_best
        for floor_guess in candidates:
            floor = max(0.0, floor_guess)
            if floor >= min_error:
                continue

            residual_logs: List[float] = []
            log_nx: List[float] = []

            for nx, err in zip(nx_values, error_values):
                diff = err * err - floor * floor
                if diff <= 0.0:
                    residual_logs = []
                    break
                residual_logs.append(0.5 * math.log(diff))
                log_nx.append(math.log(nx))

            if len(residual_logs) < 2:
                continue

            slope, intercept = linear_least_squares(log_nx, residual_logs)
            amplitude = math.exp(intercept)
            order = -slope

            sse = 0.0
            for nx, err in zip(nx_values, error_values):
                model = math.hypot(amplitude * (nx ** (-order)), floor)
                diff = err - model
                sse += diff * diff

            if best_local is None or sse < best_local.sse:
                best_local = FitResult(amplitude=amplitude, order=order, floor=floor, sse=sse)

        return best_local

    # Coarse sweep across plausible floor values.
    coarse_steps = 200
    coarse_candidates = (min_error * i / coarse_steps for i in range(coarse_steps))
    best = search_floor_candidates(coarse_candidates, best)

    if best is None:
        raise RuntimeError("Failed to find a valid power-law + floor fit.")

    # Refinement sweeps around the best floor estimate.
    span = min_error / 8.0 if best.floor > 0.0 else min_error / 16.0
    for _ in range(3):
        low = max(0.0, best.floor - span)
        high = min(min_error * 0.999, best.floor + span)
        if math.isclose(high, low):
            break
        steps = 150
        refined_candidates = (low + (high - low) * i / steps for i in range(steps + 1))
        best = search_floor_candidates(refined_candidates, best)
        span *= 0.25

    return best


def compute_observed_orders(data: Sequence[ConvergenceDatum]) -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    for prev, curr in zip(data[:-1], data[1:]):
        ratio = prev.error / curr.error
        order = math.log(prev.error / curr.error) / math.log(curr.nx / prev.nx)
        rows.append((curr.nx, ratio, order))
    return rows


def geometric_space(x_min: float, x_max: float, num: int) -> List[float]:
    if num < 2:
        return [x_min]
    log_min = math.log(x_min)
    log_max = math.log(x_max)
    return [math.exp(log_min + (log_max - log_min) * i / (num - 1)) for i in range(num)]


def make_plot(
    data: Sequence[ConvergenceDatum],
    fit: FitResult,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    nx_values = [item.nx for item in data]
    error_values = [item.error for item in data]

    nx_min = min(nx_values)
    nx_max = max(nx_values)
    nx_plot = geometric_space(nx_min, nx_max, 200)

    model_values = []
    asymptotic_values = []
    for nx in nx_plot:
        asymptotic = fit.amplitude * (nx ** (-fit.order))
        model_values.append(math.sqrt(asymptotic * asymptotic + fit.floor * fit.floor))
        asymptotic_values.append(asymptotic)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(nx_values, error_values, "o", label="Simulation data")
    ax.loglog(nx_plot, model_values, "-", label="Power law + floor fit")
    ax.loglog(nx_plot, asymptotic_values, "--", label="Asymptotic power law")

    x_lo, x_hi = ax.get_xlim()
    ax.axhline(fit.floor, color="gray", linestyle="--", label="Roundoff floor")
    ax.set_xlim(x_lo, x_hi)

    ax.set_xlabel("Cells per wavelength (N_x)")
    ax.set_ylabel("L1 error norm")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze wave convergence output (hydro or MHD).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tests") / "hydro_wave_convergence.csv",
        help="Path to the wave convergence CSV file (default: tests/hydro_wave_convergence.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the generated plot (default: <csv basename>.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: derived from CSV filename, e.g., 'Alfven Wave Convergence').",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation (useful if matplotlib is unavailable).",
    )
    args = parser.parse_args()

    data = read_convergence_csv(args.csv)
    fit = fit_power_law_with_floor(data)
    orders = compute_observed_orders(data)

    title = args.title
    if title is None:
        # Turn e.g., alfven_wave_convergence.csv into "Alfven Wave Convergence"
        title = args.csv.stem.replace("_", " ").title()
    output_path = args.output or args.csv.with_suffix(".png")

    print(f"{title} statistics:\n")
    print(f"{'nx':>8} {'dx':>14} {'error':>16}")
    for row in data:
        print(f"{row.nx:8d} {row.dx:14.6e} {row.error:16.6e}")

    print("\nObserved refinement ratios and orders (successive levels):")
    print(f"{'nx':>8} {'ratio':>10} {'order':>10}")
    for nx, ratio, order in orders:
        print(f"{nx:8d} {ratio:10.3f} {order:10.3f}")

    print("\nPower-law + floor fit parameters:")
    print(f"  amplitude (A): {fit.amplitude:.6e}")
    print(f"  order (p):     {fit.order:.3f}")
    print(f"  floor (f):     {fit.floor:.6e}")

    if args.no_plot:
        print("\nSkipping plot generation (per --no-plot).")
    else:
        make_plot(data, fit, output_path, title)
        print(f"\nSaved log-log plot to {output_path}")


if __name__ == "__main__":
    main()
