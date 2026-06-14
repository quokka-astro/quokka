#!/usr/bin/env python3
"""Preprocess raw stellar yield files into Quokka 1D DataTable CSV files."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

MSUN_CGS = 1.9884e33
MAX_MASS_POINTS = 4096

CHANNEL_FILES = {
    "SNII": "SNII_yield_table.csv",
    "WR": "WR_yield_table.csv",
    "AGB": "AGB_yield_table.csv",
}

TABLE_METALLICITIES = {
    "SNII": 0.02,
    "WR": 0.02,
    "AGB": 0.02,
}

ISOTOPE_NAME_RE = re.compile(r"^[A-Za-z]+[0-9]*$")
NUMBERED_ISOTOPE_NAME_RE = re.compile(r"^[A-Za-z]+[0-9]+$")

VALIDATION_ANCHOR_MASSES = {
    "SNII": 3.978e34 / MSUN_CGS,
    "WR": 5.66102368682473908e34 / MSUN_CGS,
    "AGB": 7.0,
}


def normalize_numeric(token: str) -> str:
    return token.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", "")


def add_fraction(entries: dict[tuple[int, int], dict[str, float]], mass: float, metallicity: float, isotope: str, value: float) -> None:
    if mass <= 0.0 or metallicity <= 0.0:
        return
    entries[(round(mass * 1000.0), round(metallicity * 1.0e6))][isotope.lower()] += max(value / mass, 0.0)


def load_sukhbold(root: Path, channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    sukhbold_root = root / "SNII_Sukhbold16"
    if not sukhbold_root.is_dir():
        return

    mass_re = re.compile(r"s([0-9]+(?:\.[0-9]+)?)\.yield_table")
    for table in sukhbold_root.iterdir():
        match = mass_re.fullmatch(table.name)
        if not (table.is_file() and match):
            continue

        mass = float(match.group(1))
        has_ejecta_col = False
        has_wind_col = False
        for line in table.read_text(errors="ignore").splitlines():
            if not line or line[0] == "#":
                continue
            if line[0] == "[":
                has_ejecta_col = "[ejecta]" in line
                has_wind_col = "[wind]" in line
                continue

            parts = line.split()
            if not parts:
                continue
            isotope = parts[0].lower()
            col = 1
            if has_ejecta_col:
                col += 1
            if has_wind_col and len(parts) > col:
                add_fraction(channel_entries["WR"], mass, 0.014, isotope, float(normalize_numeric(parts[col])))


def load_kobayashi_snii(root: Path, channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    kobayashi_root = root / "SNII_Kobayashi0611" / "z02models"
    if not kobayashi_root.is_dir():
        return

    mass_re = re.compile(r"s([0-9]+(?:\.[0-9]+)?)\.yield_table")
    for table in kobayashi_root.iterdir():
        match = mass_re.fullmatch(table.name)
        if not (table.is_file() and match):
            continue

        mass = float(match.group(1))
        has_ejecta_col = False
        for line in table.read_text(errors="ignore").splitlines():
            if not line or line[0] == "#":
                continue
            if line[0] == "[":
                has_ejecta_col = "[ejecta]" in line
                continue
            if not has_ejecta_col:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            add_fraction(channel_entries["SNII"], mass, TABLE_METALLICITIES["SNII"], parts[0], float(normalize_numeric(parts[1])))


def parse_karakas_mass_z(table: Path, text: str) -> tuple[float, float]:
    filename_re = re.compile(r"m([0-9]+(?:\.[0-9]+)?)z([0-9]+).*\.dat", re.IGNORECASE)
    match = filename_re.fullmatch(table.name)
    if match:
        z_digits = match.group(2)
        return float(match.group(1)), int(z_digits) / (10 ** len(z_digits))

    header_re = re.compile(
        r"#\s*Initial\s+mass\s*=\s*([0-9]+(?:\.[0-9]+)?),\s*Z\s*=\s*([0-9]+(?:\.[0-9]+)?),.*M_mix\s*=\s*([0-9eE+\-.]+)",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        header = header_re.search(line)
        if header:
            return float(header.group(1)), float(header.group(2))
    return -1.0, -1.0


def is_isotope_yield_file(text: str) -> bool:
    for line in text.splitlines():
        if line.strip():
            lowered = line.lower()
            return "species" in lowered and "yield" in lowered
    return False


def load_karakas(root: Path, channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    agb_root = root / "AGB_Karakas16"
    if not agb_root.is_dir():
        return

    for table in agb_root.rglob("*.dat"):
        text = table.read_text(errors="ignore")
        if not is_isotope_yield_file(text):
            continue

        mass, metallicity = parse_karakas_mass_z(table, text)
        if mass <= 0.0 or metallicity <= 0.0:
            continue

        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                float(parts[1])
                yield_value = float(normalize_numeric(parts[2]))
            except ValueError:
                continue
            if not ISOTOPE_NAME_RE.fullmatch(parts[0]):
                continue
            add_fraction(channel_entries["AGB"], mass, metallicity, parts[0], yield_value)


def load_doherty(root: Path, channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    doherty_root = root / "superAGB_Doherty14"
    if not doherty_root.is_dir():
        return

    header_re = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)M\s+Z=([0-9eE+\-.]+).*", re.IGNORECASE)
    for table in doherty_root.iterdir():
        if not table.is_file():
            continue

        mass = -1.0
        metallicity = -1.0
        for line in table.read_text(errors="ignore").splitlines():
            header = header_re.fullmatch(line)
            if header:
                mass = float(header.group(1))
                metallicity = float(header.group(2))
                continue
            if mass <= 0.0 or metallicity <= 0.0 or not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if not NUMBERED_ISOTOPE_NAME_RE.fullmatch(parts[0]):
                continue
            try:
                yield_value = float(normalize_numeric(parts[1]))
            except ValueError:
                continue
            add_fraction(channel_entries["AGB"], mass, metallicity, parts[0], yield_value)


def load_channel_entries(root: Path) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]] = {
        name: defaultdict(float_dict) for name in CHANNEL_FILES
    }
    load_kobayashi_snii(root, channel_entries)
    load_sukhbold(root, channel_entries)
    load_karakas(root, channel_entries)
    load_doherty(root, channel_entries)
    return channel_entries


def float_dict() -> dict[str, float]:
    return defaultdict(float)


def collect_isotopes(channel_entries: dict[str, dict[tuple[int, int], dict[str, float]]]) -> list[str]:
    isotopes: set[str] = set()
    for entries in channel_entries.values():
        for values in entries.values():
            isotopes.update(values)
    return sorted(isotopes)


def query_fraction(entries: dict[tuple[int, int], dict[str, float]], isotope: str, mass: float, metallicity: float) -> float:
    log_mass = math.log(max(mass, 1.0e-12))
    log_z = math.log(max(metallicity, 1.0e-12))
    weighted_sum = 0.0
    weight_total = 0.0

    for key, values in entries.items():
        frac = values.get(isotope, 0.0)
        if frac <= 0.0:
            continue
        entry_mass = max(key[0] / 1000.0, 1.0e-12)
        entry_z = max(key[1] / 1.0e6, 1.0e-12)
        dist2 = (math.log(entry_mass) - log_mass) ** 2 + (math.log(entry_z) - log_z) ** 2
        if dist2 < 1.0e-10:
            return frac
        weight = 1.0 / (dist2 + 1.0e-20)
        weighted_sum += weight * frac
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0.0 else 0.0


def build_log_mass_grid(entries: dict[tuple[int, int], dict[str, float]], channel: str) -> tuple[list[float], float, float]:
    masses = sorted({key[0] / 1000.0 for key, values in entries.items() if any(value > 0.0 for value in values.values())})
    if len(masses) < 2:
        raise RuntimeError(f"{channel} yield table needs at least two positive-yield mass points")

    xlo = masses[0]
    raw_xhi = masses[-1]
    anchor = VALIDATION_ANCHOR_MASSES[channel]
    if not (xlo < anchor < raw_xhi):
        n = min(1024, MAX_MASS_POINTS)
        log_xlo = math.log(xlo)
        dlog = (math.log(raw_xhi) - log_xlo) / (n - 1)
        return [math.exp(log_xlo + i * dlog) for i in range(n)], xlo, raw_xhi

    anchor_index = 512
    dlog = (math.log(anchor) - math.log(xlo)) / anchor_index
    n = math.ceil((math.log(raw_xhi) - math.log(xlo)) / dlog) + 1
    if n > MAX_MASS_POINTS:
        anchor_index = max(1, math.floor((MAX_MASS_POINTS - 1) * (math.log(anchor) - math.log(xlo)) / (math.log(raw_xhi) - math.log(xlo))))
        dlog = (math.log(anchor) - math.log(xlo)) / anchor_index
        n = math.ceil((math.log(raw_xhi) - math.log(xlo)) / dlog) + 1

    log_xlo = math.log(xlo)
    masses = [math.exp(log_xlo + i * dlog) for i in range(n)]
    masses[anchor_index] = anchor
    return masses, xlo, masses[-1]


def write_datatable(path: Path, entries: dict[tuple[int, int], dict[str, float]], isotopes: list[str], channel: str) -> None:
    masses, xlo, xhi = build_log_mass_grid(entries, channel)
    metallicity = TABLE_METALLICITIES[channel]
    with path.open("w", encoding="utf-8") as file:
        file.write("1\n")
        file.write(f"{len(masses)}\n")
        file.write(f"{len(isotopes)}\n")
        file.write("mass\n")
        file.write(",".join(isotopes) + "\n")
        file.write("Msun\n")
        file.write(",".join("fraction" for _ in isotopes) + "\n")
        file.write(f"{xlo:.17e}\n")
        file.write(f"{xhi:.17e}\n")
        file.write("log\n")

        for isotope in isotopes:
            values = [query_fraction(entries, isotope, mass, metallicity) for mass in masses]
            file.write(",".join(f"{value:.17e}" for value in values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yield-root", type=Path, default=Path("extern/yields"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.yield_root
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_entries = load_channel_entries(args.yield_root)
    isotopes = collect_isotopes(channel_entries)
    if not isotopes:
        raise SystemExit(f"no isotope yields found under {args.yield_root}")

    for channel, filename in CHANNEL_FILES.items():
        write_datatable(output_dir / filename, channel_entries[channel], isotopes, channel)

    print(f"wrote {len(CHANNEL_FILES)} chemical yield tables with {len(isotopes)} isotopes to {output_dir}")


if __name__ == "__main__":
    main()
