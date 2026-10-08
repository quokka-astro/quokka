#!/usr/bin/env python3
"""Regression tests for ffieldlines.

Each case writes a synthetic plotfile with analytic fields, runs ffieldlines,
and checks the output with an independent VTK reader (numpy only).

Usage:
  check_fieldlines.py --ffieldlines BIN --make-plotfile BIN --workdir DIR [--mpiexec CMD] CASE
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

STATUS = {
    "active": 0,
    "max_length": 1,
    "max_steps": 2,
    "weak_field": 3,
    "domain_exit": 4,
    "closed_loop": 5,
    "max_sweeps": 6,
    "bad_sample": 7,
}

VTK_DTYPES = {"Float64": "<f8", "Int64": "<i8", "Int32": "<i4", "UInt64": "<u8"}


# --------------------------------------------------------------------------- readers


def read_vtp(path: Path) -> dict:
    """Read a VTK XML PolyData file written with raw appended data (UInt64 headers)."""
    raw = path.read_bytes()
    marker = b'<AppendedData encoding="raw">\n_'
    start = raw.index(marker)
    header = raw[:start].decode() + "</VTKFile>"
    data = raw[start + len(marker) :]
    root = ET.fromstring(header)
    assert root.get("type") == "PolyData"
    assert root.get("byte_order") == "LittleEndian"
    assert root.get("header_type") == "UInt64"

    def load(array: ET.Element) -> np.ndarray:
        offset = int(array.get("offset"))
        nbytes = int(np.frombuffer(data, dtype="<u8", count=1, offset=offset)[0])
        dtype = np.dtype(VTK_DTYPES[array.get("type")])
        values = np.frombuffer(data, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset + 8)
        ncomp = int(array.get("NumberOfComponents", "1"))
        return values.reshape(-1, ncomp) if ncomp > 1 else values

    for array in root.iter("DataArray"):
        # VTK's XML reader needs NumberOfTuples to size FieldData arrays
        assert array.get("NumberOfTuples") is not None, f"DataArray {array.get('Name')} lacks NumberOfTuples"
    piece = root.find("PolyData/Piece")
    out = {
        "field": {a.get("Name"): load(a) for a in root.findall("PolyData/FieldData/DataArray")},
        "point": {a.get("Name"): load(a) for a in piece.findall("PointData/DataArray")},
        "cell": {a.get("Name"): load(a) for a in piece.findall("CellData/DataArray")},
        "points": load(piece.find("Points/DataArray")),
    }
    lines = {a.get("Name"): load(a) for a in piece.findall("Lines/DataArray")}
    npts = int(piece.get("NumberOfPoints"))
    nlines = int(piece.get("NumberOfLines"))
    assert out["points"].shape == (npts, 3), out["points"].shape
    assert lines["offsets"].shape == (nlines,)
    assert np.array_equal(lines["connectivity"], np.arange(npts))
    out["offsets"] = np.concatenate([[0], lines["offsets"]])
    for name, values in {**out["point"]}.items():
        assert values.shape == (npts,), name
    for name, values in out["cell"].items():
        assert values.shape == (nlines,), name
    return out


def read_legacy_vtk(path: Path) -> dict:
    """Read the legacy binary POLYDATA subset that ffieldlines writes."""
    raw = path.read_bytes()
    pos = 0

    def line() -> str:
        nonlocal pos
        end = raw.index(b"\n", pos)
        text = raw[pos:end].decode()
        pos = end + 1
        return text

    def block(count: int, dtype: str) -> np.ndarray:
        nonlocal pos
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
        pos += values.nbytes + 1  # trailing newline
        return values.astype(dtype.replace(">", "="))

    types = {"double": ">f8", "int": ">i4"}
    assert line().startswith("# vtk DataFile Version 3.0")
    line()
    assert line() == "BINARY"
    assert line() == "DATASET POLYDATA"
    out: dict = {"field": {}, "point": {}, "cell": {}}
    section = "field"
    while pos < len(raw):
        words = line().split()
        if not words:
            continue
        if words[0] == "FIELD":
            for _ in range(int(words[2])):
                name, ncomp, ntuples, vtype = line().split()
                out[section][name] = block(int(ncomp) * int(ntuples), types[vtype])
        elif words[0] == "POINTS":
            out["points"] = block(3 * int(words[1]), types[words[2]]).reshape(-1, 3)
        elif words[0] == "LINES":
            cells = block(int(words[2]), ">i4")
            offsets = [0]
            k = 0
            while k < len(cells):
                n = int(cells[k])
                assert np.array_equal(cells[k + 1 : k + 1 + n], np.arange(offsets[-1], offsets[-1] + n))
                offsets.append(offsets[-1] + n)
                k += n + 1
            out["offsets"] = np.array(offsets)
        elif words[0] == "CELL_DATA":
            section = "cell"
        elif words[0] == "POINT_DATA":
            section = "point"
        else:
            raise AssertionError(f"unexpected legacy VTK keyword {words[0]}")
    return out


def pieces(data: dict) -> list[slice]:
    off = data["offsets"]
    return [slice(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1)]


# --------------------------------------------------------------------------- helpers


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.ffieldlines = args.ffieldlines
        self.make_plotfile = args.make_plotfile
        self.mpiexec = shlex.split(args.mpiexec) if args.mpiexec else []
        self.workdir = Path(args.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def plotfile(self, case: str, ncell: int, max_grid: int, nlevels: int) -> Path:
        path = self.workdir / f"plt_{case}_{ncell}_{max_grid}_{nlevels}"
        if not (path / "Header").exists():
            subprocess.run([self.make_plotfile, case, str(path), str(ncell), str(max_grid), str(nlevels)], check=True, capture_output=True)
        return path

    def trace(self, plotfile: Path, out: str, *options: str, mpi: bool = False, expect_fail: bool = False) -> str:
        cmd = [*(self.mpiexec if mpi else []), self.ffieldlines, *options, "-o", str(self.workdir / out), str(plotfile)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        if expect_fail:
            assert proc.returncode != 0, f"expected failure:\n{' '.join(cmd)}\n{output}"
        else:
            assert proc.returncode == 0, f"ffieldlines failed:\n{' '.join(cmd)}\n{output}"
        return output


def check_close(name: str, actual, expected, atol: float) -> None:
    err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))) if np.size(actual) else 0.0
    assert err <= atol, f"{name}: max error {err:.3e} > {atol:.1e}"


def assert_same_output(a: dict, b: dict) -> None:
    assert np.array_equal(a["offsets"], b["offsets"]), "piece offsets differ"
    assert np.array_equal(a["points"], b["points"]), "points differ"
    for section in ("point", "cell"):
        assert a[section].keys() == b[section].keys()
        for name in a[section]:
            assert np.array_equal(a[section][name], b[section][name], equal_nan=True), f"{section} array {name} differs"


# --------------------------------------------------------------------------- cases

CIRCLE_SEEDS = ("--seed-line", "0.6", "0.5", "0.4", "0.85", "0.5", "0.6", "4")


def check_circle_lines(data: dict, dx_fine: float) -> None:
    """Checks for the circle field B = (-(y-1/2), x-1/2, 0) with rho = 1+x, v = (1,0,0), T = 100+10z."""
    assert len(pieces(data)) == 4, f"expected 4 closed lines, got {len(pieces(data))}"
    assert np.all(data["cell"]["status_forward"] == STATUS["closed_loop"])
    pts = data["points"]
    for idx, sl in enumerate(pieces(data)):
        p = pts[sl]
        r = np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5)
        check_close("radius conservation", r, r[0], 1.0e-7)
        check_close("z conservation", p[:, 2], p[0, 2], 1.0e-12)
        assert np.allclose(p[0], p[-1]), "closed loop should end at its seed"
        length = data["cell"]["length_forward"][idx]
        assert abs(length - 2.0 * math.pi * r[0]) <= 2.0 * dx_fine, f"loop length {length} vs 2 pi r = {2 * math.pi * r[0]}"
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rxy = np.hypot(x - 0.5, y - 0.5)
    check_close("density", data["point"]["density"], 1.0 + x, 1.0e-12)
    check_close("temperature", data["point"]["temperature"], 100.0 + 10.0 * z, 1.0e-10)
    check_close("B_mag", data["point"]["B_mag"], rxy, 1.0e-12)
    check_close("v_mag", data["point"]["v_mag"], 1.0, 1.0e-12)
    check_close("v_dot_B", data["point"]["v_dot_B"], -(y - 0.5), 1.0e-12)
    check_close("cos_vB", data["point"]["cos_vB"], -(y - 0.5) / rxy, 1.0e-10)


def case_circle(run: Runner) -> None:
    plt = run.plotfile("circle", 32, 8, 1)
    run.trace(plt, "circle.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS)
    check_circle_lines(read_vtp(run.workdir / "circle.vtp"), 1.0 / 32)


def case_decomposition(run: Runner) -> None:
    """Identical output for 1 grid vs 64 grids, and for different steps per sweep."""
    one = run.plotfile("circle", 32, 32, 1)
    many = run.plotfile("circle", 32, 8, 1)
    run.trace(one, "one_grid.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS)
    run.trace(many, "many_grids.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS)
    run.trace(many, "short_sweeps.vtp", "--periodic", "0", "0", "0", "--steps-per-sweep", "3", *CIRCLE_SEEDS)
    reference = read_vtp(run.workdir / "one_grid.vtp")
    assert_same_output(reference, read_vtp(run.workdir / "many_grids.vtp"))
    assert_same_output(reference, read_vtp(run.workdir / "short_sweeps.vtp"))


def case_mpi(run: Runner) -> None:
    """Identical output on 1 and 4 MPI ranks."""
    if not run.mpiexec:
        print("no MPI launcher configured; skipping")
        return
    plt = run.plotfile("circle", 32, 8, 2)
    run.trace(plt, "serial.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS)
    run.trace(plt, "parallel.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS, mpi=True)
    assert_same_output(read_vtp(run.workdir / "serial.vtp"), read_vtp(run.workdir / "parallel.vtp"))


def case_twolevel(run: Runner) -> None:
    """Lines crossing a refined patch stay exact and are integrated with the fine step there."""
    plt = run.plotfile("circle", 32, 8, 2)
    run.trace(plt, "twolevel.vtp", "--periodic", "0", "0", "0", "--output-every", "1", *CIRCLE_SEEDS)
    data = read_vtp(run.workdir / "twolevel.vtp")
    dx_c, dx_f = 1.0 / 32, 1.0 / 64
    check_circle_lines(data, dx_f)
    spacing = np.concatenate([np.diff(data["point"]["arc_length"][sl]) for sl in pieces(data)])
    spacing = np.abs(spacing[spacing != 0.0])
    assert np.any(np.isclose(spacing, 0.25 * dx_f)), "no fine-level steps recorded"
    assert np.any(np.isclose(spacing, 0.25 * dx_c)), "no coarse-level steps recorded"
    fine = np.isclose(spacing, 0.25 * dx_f)
    assert fine.sum() > 20, "too few fine-level steps"


def case_straight(run: Runner) -> None:
    """Uniform diagonal B: straight lines of exact length, exact samples, max_length on both halves."""
    plt = run.plotfile("diagonal", 16, 8, 1)
    length = 0.2
    run.trace(plt, "straight.vtp", "--periodic", "0", "0", "0", "--max-length", str(length), "--seed-line", "0.3", "0.4", "0.5", "0.6", "0.5", "0.5", "3")
    data = read_vtp(run.workdir / "straight.vtp")
    assert len(pieces(data)) == 3
    assert np.all(data["cell"]["status_forward"] == STATUS["max_length"])
    assert np.all(data["cell"]["status_backward"] == STATUS["max_length"])
    check_close("forward length", data["cell"]["length_forward"], length, 1.0e-12)
    check_close("backward length", data["cell"]["length_backward"], length, 1.0e-12)
    direction = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    for sl in pieces(data):
        p = data["points"][sl]
        s = data["point"]["arc_length"][sl]
        check_close("arc length range", [s[0], s[-1]], [-length, length], 1.0e-12)
        seed = p[np.argmin(np.abs(s))]
        check_close("straight line", p, seed + np.outer(s, direction), 1.0e-12)
    check_close("cos_vB", data["point"]["cos_vB"], 1.0 / math.sqrt(2.0), 1.0e-12)
    check_close("density", data["point"]["density"], 2.0, 1.0e-12)


def case_periodic(run: Runner) -> None:
    """A line crossing a periodic face is split into two pieces without a cross-box segment."""
    plt = run.plotfile("uniform", 16, 8, 1)
    run.trace(plt, "periodic.vtp", "--periodic", "1", "0", "0", "--direction", "forward", "--max-length", "0.5", "--seed-line", "0.9", "0.5", "0.5", "0.9", "0.5", "0.5", "1")
    data = read_vtp(run.workdir / "periodic.vtp")
    assert len(pieces(data)) == 2, f"expected 2 pieces, got {len(pieces(data))}"
    assert np.all(data["cell"]["piece_index"] == [0, 1])
    assert np.all(data["cell"]["status_forward"] == STATUS["max_length"])
    assert np.all(data["cell"]["status_backward"] == -1)
    for sl in pieces(data):
        assert np.all(np.abs(np.diff(data["points"][sl][:, 0])) < 0.5)
    check_close("total length", data["cell"]["length_forward"], 0.5, 1.0e-12)
    check_close("cos_vB", data["point"]["cos_vB"], 1.0 / math.sqrt(2.0), 1.0e-12)


def case_terminations(run: Runner) -> None:
    plt = run.plotfile("uniform", 16, 8, 1)
    run.trace(plt, "exit.vtp", "--periodic", "0", "0", "0", "--direction", "forward", "--seed-line", "0.9", "0.5", "0.5", "0.9", "0.5", "0.5", "1")
    data = read_vtp(run.workdir / "exit.vtp")
    assert np.all(data["cell"]["status_forward"] == STATUS["domain_exit"])
    assert data["points"][:, 0].max() >= 1.0

    run.trace(plt, "steps.vtp", "--periodic", "0", "0", "0", "--max-steps", "10", "--seed-line", "0.5", "0.5", "0.5", "0.5", "0.5", "0.5", "1")
    data = read_vtp(run.workdir / "steps.vtp")
    assert np.all(data["cell"]["status_forward"] == STATUS["max_steps"])
    check_close("max_steps length", data["cell"]["length_forward"], 10 * 0.25 / 16, 1.0e-12)

    circle = run.plotfile("circle", 16, 8, 1)
    output = run.trace(circle, "weak.vtp", "--periodic", "0", "0", "0", "--seed-line", "0.5", "0.5", "0.5", "0.5", "0.5", "0.5", "1")
    assert "weak_field=1" in output, output
    assert "fewer than two points" in output, output


def case_legacy(run: Runner) -> None:
    plt = run.plotfile("circle", 32, 8, 1)
    run.trace(plt, "lines.vtp", "--periodic", "0", "0", "0", *CIRCLE_SEEDS)
    run.trace(plt, "lines.vtk", "--periodic", "0", "0", "0", "--format", "vtk", *CIRCLE_SEEDS)
    xml = read_vtp(run.workdir / "lines.vtp")
    legacy = read_legacy_vtk(run.workdir / "lines.vtk")
    assert np.array_equal(xml["offsets"], legacy["offsets"])
    assert np.array_equal(xml["points"], legacy["points"])
    for name, values in xml["point"].items():
        assert np.array_equal(values, legacy["point"][name], equal_nan=True), name
    for name, values in xml["cell"].items():
        assert np.array_equal(values, legacy["cell"][name]), name
    check_close("TIME", legacy["field"]["TIME"], 1.5, 0.0)
    check_close("TimeValue", xml["field"]["TimeValue"], 1.5, 0.0)


def case_no_temperature(run: Runner) -> None:
    plt = run.plotfile("circle", 16, 8, 1)
    run.trace(plt, "notemp.vtp", "--periodic", "0", "0", "0", "--temperature", "none", "--seed-line", "0.7", "0.5", "0.5", "0.7", "0.5", "0.5", "1")
    data = read_vtp(run.workdir / "notemp.vtp")
    assert "temperature" not in data["point"]
    assert "cos_vB" in data["point"]


def case_validation(run: Runner) -> None:
    plt = run.plotfile("uniform", 16, 8, 1)
    seed = ("--seed-line", "0.5", "0.5", "0.5", "0.5", "0.5", "0.5", "1")
    checks = [
        ((*seed,), "--periodic"),
        (("--periodic", "0", "0", "0"), "exactly one of"),
        (("--periodic", "0", "0", "0", "--seed-line", "1.5", "0.5", "0.5", "1.5", "0.5", "0.5", "1"), "outside the domain"),
        (("--periodic", "0", "0", "0", "--density", "rho", *seed), "Available variables"),
        (("--periodic", "0", "0", "0", "--step-fraction", "0.75", *seed), "--step-fraction"),
        (("--periodic", "0", "0", "0", "--seed-line", "0.1", "0.5", "0.5", "0.9", "0.5", "0.5", "10001"), "point count"),
    ]
    for options, message in checks:
        output = run.trace(plt, "invalid.vtp", *options, expect_fail=True)
        assert message in output, f"expected '{message}' in output of {options}:\n{output}"


CASES = {
    name[len("case_") :]: fn for name, fn in globals().items() if name.startswith("case_") and callable(fn)
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffieldlines", required=True)
    parser.add_argument("--make-plotfile", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mpiexec", default="", help="launcher prefix for the MPI case, e.g. 'mpiexec -n 4'")
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    CASES[args.case](Runner(args))
    print(f"{args.case}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
