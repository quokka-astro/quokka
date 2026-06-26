#!/usr/bin/env python3
"""Generate a DTypeFront performance table from benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


METHOD_ORDER = {
    "ROS2S": 0,
    "Rodas3P": 1,
    "Rodas4P": 2,
    "VODE": 3,
    "Rodas5P": 4,
}


@dataclass
class Metrics:
    method: str
    fom_us: float | None = None
    mupdates: float | None = None
    tp_wall_s: float | None = None
    chem_s: float | None = None
    chem_pct: float | None = None
    adv_s: float | None = None
    adv_pct: float | None = None
    hydro_s: float | None = None
    hydro_pct: float | None = None
    subcycles: float | None = None
    failed: str | None = None

    @property
    def rad_no_ode_pct(self) -> float | None:
        if self.adv_pct is None or self.hydro_pct is None or self.chem_pct is None:
            return None
        return self.adv_pct - self.hydro_pct - self.chem_pct

    @property
    def hydro_only_speedup(self) -> float | None:
        if self.hydro_pct is None or self.hydro_pct == 0.0:
            return None
        return 100.0 / self.hydro_pct

    @property
    def rad_substep_cost(self) -> float | None:
        if self.adv_s is None or self.hydro_s is None or self.subcycles is None or self.hydro_s == 0.0:
            return None
        return (self.adv_s - self.hydro_s) / (self.hydro_s * self.subcycles)


def discover_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        if path.is_dir():
            logs.extend(sorted(path.glob("*.log")))
        else:
            logs.append(path)
    return logs


def method_from_log(text: str, path: Path) -> str:
    match = re.search(r"DTypeFront microphysics integrator: VODE", text)
    if match:
        return "VODE"

    match = re.search(r"DTypeFront microphysics integrator: Rosenbrock \(Rosenbrock tableau \d+: ([^)]+)\)", text)
    if match:
        return match.group(1)

    name_map = {
        "ros2s": "ROS2S",
        "rodas3p": "Rodas3P",
        "rodas4p": "Rodas4P",
        "rodas5p": "Rodas5P",
        "vode": "VODE",
    }
    return name_map.get(path.stem.lower(), path.stem)


def parse_profiler_line(text: str, name: str) -> tuple[float, float] | None:
    # Example:
    # PhotoChemistry::computePhotoChemistry() 7734 6.474 6.474 6.474 33.65%
    pattern = re.compile(rf"^{re.escape(name)}\s+\d+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)%", re.MULTILINE)
    matches = [(float(match.group(3)), float(match.group(4))) for match in pattern.finditer(text)]
    if not matches:
        return None
    # The same profiler name can appear as exclusive and inclusive entries.  The
    # inclusive entry is the one with the largest walltime/percentage.
    return max(matches, key=lambda value: value[1])


def parse_log(path: Path) -> Metrics:
    text = path.read_text(errors="replace")
    metrics = Metrics(method=method_from_log(text, path))

    match = re.search(r"Performance figure-of-merit:\s+([0-9.eE+-]+)\s+.*?\[([0-9.eE+-]+)\s+Mupdates/s\]", text)
    if match:
        metrics.fom_us = float(match.group(1))
        metrics.mupdates = float(match.group(2))

    match = re.search(r"TinyProfiler total time across processes \[min\.\.\.avg\.\.\.max\]:\s+([0-9.eE+-]+)\s+\.\.\.\s+([0-9.eE+-]+)\s+\.\.\.\s+([0-9.eE+-]+)", text)
    if match:
        metrics.tp_wall_s = float(match.group(3))

    match = re.search(r"avg\. num\. of radiation subcycles\s*=\s*([0-9.eE+-]+)", text)
    if match:
        metrics.subcycles = float(match.group(1))

    chem = parse_profiler_line(text, "PhotoChemistry::computePhotoChemistry()")
    if chem is not None:
        metrics.chem_s, metrics.chem_pct = chem

    advance = parse_profiler_line(text, "QuokkaSimulation::advanceSingleTimestepAtLevel()")
    if advance is not None:
        metrics.adv_s, metrics.adv_pct = advance

    hydro = parse_profiler_line(text, "REG::HydroSolver")
    if hydro is not None:
        metrics.hydro_s, metrics.hydro_pct = hydro

    match = re.search(r"Photochemistry burn failed.*", text)
    if match:
        metrics.failed = match.group(0)

    return metrics


def fmt(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def fmt_fom(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def make_table(metrics: list[Metrics]) -> str:
    rows = sorted(metrics, key=lambda row: (METHOD_ORDER.get(row.method, 99), row.method))
    lines = [
        "| Method | FoM us/zone | Mupdate/s | TP wall s | Chem s | Chem % | Rad-noODE % | Hydro % | Hydro-only x | RadSub/HydroStep | Subcyc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    fmt_fom(row.fom_us),
                    fmt(row.mupdates, 2),
                    fmt(row.tp_wall_s, 2),
                    fmt(row.chem_s, 2),
                    fmt(row.chem_pct, 2),
                    fmt(row.rad_no_ode_pct, 2),
                    fmt(row.hydro_pct, 2),
                    fmt(row.hydro_only_speedup, 2),
                    fmt(row.rad_substep_cost, 2),
                    fmt(row.subcycles, 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="DTypeFront log files or directories containing *.log files")
    parser.add_argument("--show-failures", action="store_true", help="print failed-run diagnostics after the table")
    args = parser.parse_args()

    logs = discover_logs(args.logs)
    if not logs:
        raise SystemExit("no log files found")

    metrics = [parse_log(path) for path in logs]
    print(make_table(metrics))

    if args.show_failures:
        failures = [(path, row.failed) for path, row in zip(logs, metrics, strict=True) if row.failed is not None]
        if failures:
            print("\nFailures:")
            for path, failure in failures:
                print(f"- {path}: {failure}")


if __name__ == "__main__":
    main()
