#!/usr/bin/env python3
"""Plot cloud mass histories for multiple temperature thresholds."""
from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THRESHOLD_RE = re.compile(r"^cloud_mass_([0-9]+(?:e[0-9]+)?)$")


@dataclass
class Dataset:
    path: Path
    time: np.ndarray
    thresholds: Dict[str, np.ndarray]

    @property
    def label(self) -> str:
        # Use the immediate parent directory when available to keep legend concise.
        parent = self.path.parent.name
        if parent and parent != ".":
            return parent
        return self.path.stem


def _format_threshold(threshold_key: str) -> str:
    match = _THRESHOLD_RE.match(threshold_key)
    if not match:
        return threshold_key
    raw = match.group(1)
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value.is_integer():
        return f"{value:,.0f} K"
    return f"{value:g} K"


def load_history(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(path)

    header: List[str] | None = None
    data_lines: List[str] = []
    with path.open() as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("# ").split()
                continue
            data_lines.append(line)

    if header is None:
        raise ValueError(f"Could not find a header line in {path}.")
    if not data_lines:
        raise ValueError(f"No data rows found in {path}.")

    buffer = io.StringIO("\n".join(data_lines))
    data = np.genfromtxt(buffer, names=header)
    if data.size == 0:
        raise ValueError(f"No data rows found in {path}.")
    if data.shape == ():
        data = np.array([data])

    if data.dtype.names is None:
        raise ValueError(f"Failed to parse column headers in {path}.")

    names = list(data.dtype.names)
    time_column = "time" if "time" in names else "cycle"
    if time_column not in names:
        raise ValueError(f"Neither 'time' nor 'cycle' column found in {path}.")

    thresholds: Dict[str, np.ndarray] = {}
    for name in names:
        if _THRESHOLD_RE.match(name):
            thresholds[name] = np.asarray(data[name])

    if not thresholds:
        raise ValueError(f"No cloud mass threshold columns found in {path}.")

    return Dataset(path=path, time=np.asarray(data[time_column]), thresholds=thresholds)


def gather_thresholds(datasets: Iterable[Dataset]) -> List[str]:
    threshold_sets = [set(ds.thresholds.keys()) for ds in datasets]
    common = set.intersection(*threshold_sets)
    if not common:
        raise ValueError("Datasets do not share any common cloud mass thresholds.")

    def sort_key(name: str) -> float:
        match = _THRESHOLD_RE.match(name)
        if not match:
            return float("inf")
        raw = match.group(1)
        try:
            return float(raw)
        except ValueError:
            return float("inf")

    return sorted(common, key=sort_key)


def plot_cloud_mass(datasets: List[Dataset], thresholds: List[str], output: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))
    style_cycle = cycle(["-", "--", ":", "-."])

    for dataset, linestyle in zip(datasets, style_cycle):
        for color, threshold in zip(colors, thresholds):
            ax.plot(
                dataset.time,
                dataset.thresholds[threshold],
                linestyle=linestyle,
#                color=color,
                label=f"{dataset.label}: {_format_threshold(threshold)}",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Cloud Mass")
    ax.set_title("Cloud Mass History by Temperature Threshold")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper left", fontsize="small", ncol=2, frameon=False)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests/history.txt", "tests/noshock/history.txt"],
        type=Path,
        help="History files to include in the plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure instead of displaying it.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        help="Specify a temperature threshold (matching column suffix) to include. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.paths]
    datasets = [load_history(path) for path in paths]

    thresholds = gather_thresholds(datasets)
    if args.thresholds:
        requested = {f"cloud_mass_{value}" for value in args.thresholds}
        thresholds = [thr for thr in thresholds if thr in requested]
        if not thresholds:
            raise ValueError("None of the requested thresholds were found in all datasets.")

    plot_cloud_mass(datasets, thresholds, args.output)


if __name__ == "__main__":
    main()
