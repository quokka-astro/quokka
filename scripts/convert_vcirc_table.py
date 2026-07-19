#!/usr/bin/env python3
"""Convert a 5-column vcirc table to quokka::DataTable CSVReader format."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_header_units(header_line: str) -> tuple[str, list[str], list[str]]:
    header = header_line.lstrip("#").strip()
    parts = header.split()
    if len(parts) != 5:
        return "R", ["vcirc", "rho", "velr", "T"], ["", "", "", ""]

    input_name_raw = parts[0]
    output_names_raw = parts[1:]

    def split_name_unit(token: str) -> tuple[str, str]:
        if "_" in token:
            name, unit = token.rsplit("_", 1)
            return name, unit
        return token, ""

    input_name, input_unit = split_name_unit(input_name_raw)
    output_names = []
    output_units = []
    for tok in output_names_raw:
        name, unit = split_name_unit(tok)
        output_names.append(name)
        output_units.append(unit)

    return input_name, output_names, [input_unit] + output_units


def read_table(path: Path) -> tuple[list[float], list[list[float]], str, list[str], list[str], list[str]]:
    input_name = "R"
    output_names = ["vcirc", "rho", "velr", "T"]
    input_units = [""]
    output_units = ["", "", "", ""]

    data = [[], [], [], []]
    r_values = []
    header_line = ""

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header_line = stripped
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 columns, got {len(parts)} in line: {line}")
        values = [float(x) for x in parts]
        r_values.append(values[0])
        data[0].append(values[1])
        data[1].append(values[2])
        data[2].append(values[3])
        data[3].append(values[4])

    if header_line:
        input_name, output_names, units = parse_header_units(header_line)
        input_units = [units[0]]
        output_units = units[1:]

    return r_values, data, input_name, output_names, input_units, output_units


def write_datatable_csv(
    output_path: Path,
    r_values: list[float],
    data: list[list[float]],
    input_name: str,
    output_names: list[str],
    input_units: list[str],
    output_units: list[str],
) -> None:
    n = len(r_values)
    if n == 0:
        raise ValueError("No data rows found")

    xlo = min(r_values)
    xhi = max(r_values)

    def join_csv(values) -> str:
        return ",".join(values)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"{n}\n")
        f.write("4\n")
        f.write(f"{input_name}\n")
        f.write(join_csv(output_names) + "\n")
        f.write(join_csv(input_units) + "\n")
        f.write(join_csv(output_units) + "\n")
        f.write(f"{xlo:.17e}\n")
        f.write(f"{xhi:.17e}\n")
        f.write("linear\n")

        # Data: Nout rows, each with N values
        for out_idx in range(4):
            row = ",".join(f"{val:.17e}" for val in data[out_idx])
            f.write(row + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert vcirc table to DataTable CSVReader format")
    parser.add_argument("input", type=Path, help="Input table (space-separated, 5 columns)")
    parser.add_argument("output", type=Path, help="Output CSVReader file")
    args = parser.parse_args()

    r_values, data, input_name, output_names, input_units, output_units = read_table(args.input)
    write_datatable_csv(args.output, r_values, data, input_name, output_names, input_units, output_units)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
