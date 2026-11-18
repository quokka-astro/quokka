#!/usr/bin/env python3
# SPDX-FileCopyrightText: © The Quokka Authors
# SPDX-License-Identifier: MIT
"""Convert an ORION2 (Chombo HDF5) MHD checkpoint into a native Quokka checkpoint.

ORION2 and Quokka share the same block-structured AMR layout (Berger-Colella grids with an
integer refinement ratio, cell-centred conserved variables, face-centred magnetic field), so the
conversion copies every box of every level one-to-one into AMReX VisMF files.  The only changes
are a rescaling to physical units and a re-ordering of the components.

Scaling.  The ORION2 snapshot is a scale-free driven-turbulence box, so it is rescaled with three
factors chosen from the command line:

    f_L   = L_target     / L_orion           (box length)
    f_rho = rho0_target  / <rho>_V,orion      (volume-mean density)
    f_v   = sigma_target / sigma_M,orion      (mass-weighted 3D velocity dispersion)

with sigma_M^2 = <v^2>_M - |<v>_M|^2, evaluated on the composite AMR hierarchy (every cell
counted once, at its finest available level).  All other quantities follow from dimensional
analysis: time f_L/f_v, momentum f_rho f_v, energy density f_rho f_v^2 and magnetic field
sqrt(f_rho) f_v (both codes store B such that the magnetic energy density is B^2/2).

Output layout (cell-centred), matching Physics_Indices for an MHD problem with one passive
scalar and one radiation group (e.g. src/problems/OrionRestart):

    0 density, 1-3 momentum, 4 total energy, 5 internal energy,
    [6 passive scalar (tracer1)], [Erad, Fx, Fy, Fz]

The radiation flux is set to zero (ORION2 uses flux-limited diffusion and stores no flux).

The target values are recorded in <outdir>/metadata.yaml so that a Quokka run can verify them.

Example:
    uv run --python .venv python scripts/python/orion2_to_quokka_checkpoint.py \\
        chk.0000.3d.hdf5 chk_orion --length 3.0857e18 --mean-density 1e-20 --vdisp 1e5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import h5py
import numpy as np

A_RAD = 7.5657e-15  # erg cm^-3 K^-4

# little-endian IEEE double: AMReX FPC::NativeRealDescriptor on x86/ARM (ieee_double, reverse_double_order)
REAL_DESCRIPTOR = "((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"

ORION_NAMES = {
    "rho": "density",
    "mx": "X-momentum",
    "my": "Y-momentum",
    "mz": "Z-momentum",
    "bx": "X-magnfield",
    "by": "Y-magnfield",
    "bz": "Z-magnfield",
    "E": "energy-density",
    "tracer": "tracer1",
    "Erad": "radiation-energy-density",
}


@dataclass
class Level:
    """Grid hierarchy information for one ORION2 AMR level."""

    lev: int
    lo: np.ndarray  # (nboxes, 3) inclusive lower corners (i, j, k)
    hi: np.ndarray  # (nboxes, 3) inclusive upper corners
    cell_offsets: np.ndarray
    face_offsets: np.ndarray
    dx: float
    dt: float
    ref_ratio: int
    domain_lo: np.ndarray
    domain_hi: np.ndarray

    @property
    def nboxes(self) -> int:
        return len(self.lo)

    def shape(self, ib: int) -> tuple[int, int, int]:
        """(nx, ny, nz) of box ib."""
        n = self.hi[ib] - self.lo[ib] + 1
        return int(n[0]), int(n[1]), int(n[2])


def read_hierarchy(f: h5py.File, max_level: int) -> tuple[list[Level], dict[str, int]]:
    ncomp = int(f.attrs["num_components"])
    names = [f.attrs[f"component_{n}"].decode() for n in range(ncomp)]
    comp = {}
    for key, name in ORION_NAMES.items():
        if name not in names:
            sys.exit(f"error: ORION2 component '{name}' not found; available: {names}")
        comp[key] = names.index(name)

    if int(f["Chombo_global"].attrs["SpaceDim"]) != 3:
        sys.exit("error: only 3D ORION2 checkpoints are supported")

    levels = []
    nlev = min(int(f.attrs["num_levels"]), max_level + 1)
    for lev in range(nlev):
        g = f[f"level_{lev}"]
        for name in ("cell_data", "face_data"):
            og = g[f"{name}_attributes"].attrs["outputGhost"]
            if tuple(og) != (0, 0, 0):
                sys.exit(f"error: level {lev} {name} was written with ghost cells {tuple(og)}; expected none")
        if not all(int(g.attrs[f"is_periodic_{d}"]) == 1 for d in range(3)):
            print(f"warning: level {lev} is not periodic in every direction", file=sys.stderr)
        boxes = g["boxes"][:]
        lo = np.stack([boxes["lo_i"], boxes["lo_j"], boxes["lo_k"]], axis=1).astype(np.int64)
        hi = np.stack([boxes["hi_i"], boxes["hi_j"], boxes["hi_k"]], axis=1).astype(np.int64)
        dom = g.attrs["prob_domain"]
        levels.append(
            Level(
                lev=lev,
                lo=lo,
                hi=hi,
                cell_offsets=g["cell_data:offsets=0"][:],
                face_offsets=g["face_data:offsets=0"][:],
                dx=float(g.attrs["dx"]),
                dt=float(g.attrs["dt"]),
                ref_ratio=int(g.attrs["ref_ratio"]),
                domain_lo=np.array([dom["lo_i"], dom["lo_j"], dom["lo_k"]], dtype=np.int64),
                domain_hi=np.array([dom["hi_i"], dom["hi_j"], dom["hi_k"]], dtype=np.int64),
            )
        )
    for lv in levels:
        if np.any(lv.domain_lo != 0):
            sys.exit(f"error: level {lv.lev} domain does not start at index 0")
    for lv in levels[:-1]:
        if lv.ref_ratio != 2:
            sys.exit(f"error: level {lv.lev} has refinement ratio {lv.ref_ratio}; only 2 is supported")
    return levels, comp


def read_cells(f: h5py.File, lv: Level, ib: int, ncomp: int, comps: int | None = None) -> np.ndarray:
    """Return box ib as an array of shape (ncomp, nz, ny, nx) (Fortran cell ordering).

    If comps is given, only the leading `comps` components are read."""
    nx, ny, nz = lv.shape(ib)
    ncell = nx * ny * nz
    start = int(lv.cell_offsets[ib])
    nread = ncomp if comps is None else comps
    data = f[f"level_{lv.lev}/cell_data:datatype=0"][start : start + nread * ncell]
    return data.reshape(nread, nz, ny, nx)


def read_faces(f: h5py.File, lv: Level, ib: int) -> list[np.ndarray]:
    """Return the face-centred field of box ib as [Bx(nz,ny,nx+1), By(nz,ny+1,nx), Bz(nz+1,ny,nx)]."""
    nx, ny, nz = lv.shape(ib)
    start = int(lv.face_offsets[ib])
    stop = int(lv.face_offsets[ib + 1])
    data = f[f"level_{lv.lev}/face_data:datatype=0"][start:stop]
    sx = (nx + 1) * ny * nz
    sy = nx * (ny + 1) * nz
    sz = nx * ny * (nz + 1)
    if data.size != sx + sy + sz:
        sys.exit(f"error: level {lv.lev} box {ib}: face data has {data.size} values, expected {sx + sy + sz}")
    return [
        data[:sx].reshape(nz, ny, nx + 1),
        data[sx : sx + sy].reshape(nz, ny + 1, nx),
        data[sx + sy :].reshape(nz + 1, ny, nx),
    ]


def uncovered_mask(lv: Level, finer: Level | None, ib: int) -> np.ndarray:
    """Boolean mask (nz, ny, nx): True where box ib of lv is not covered by the next finer level."""
    nx, ny, nz = lv.shape(ib)
    mask = np.ones((nz, ny, nx), dtype=bool)
    if finer is None:
        return mask
    r = lv.ref_ratio
    flo = finer.lo // r
    fhi = finer.hi // r
    ilo = np.maximum(flo, lv.lo[ib])
    ihi = np.minimum(fhi, lv.hi[ib])
    hits = np.all(ilo <= ihi, axis=1)
    for a, b in zip(ilo[hits] - lv.lo[ib], ihi[hits] - lv.lo[ib]):
        mask[a[2] : b[2] + 1, a[1] : b[1] + 1, a[0] : b[0] + 1] = False
    return mask


def composite_statistics(f: h5py.File, levels: list[Level], comp: dict[str, int]) -> tuple[float, float]:
    """Volume-mean density and mass-weighted velocity dispersion on the composite hierarchy."""
    # the density and momenta are read as one contiguous slice: they must be the leading components
    if [comp["rho"], comp["mx"], comp["my"], comp["mz"]] != [0, 1, 2, 3]:
        sys.exit("error: expected density and momenta to be ORION2 components 0-3")
    mass = 0.0
    p = np.zeros(3)
    two_ekin = 0.0
    for i, lv in enumerate(levels):
        finer = levels[i + 1] if i + 1 < len(levels) else None
        vol = lv.dx**3
        for ib in range(lv.nboxes):
            d = read_cells(f, lv, ib, ncomp=4, comps=4)
            w = uncovered_mask(lv, finer, ib)
            rho = d[0][w]
            mom = d[1:4][:, w]
            mass += vol * rho.sum()
            p += vol * mom.sum(axis=1)
            two_ekin += vol * ((mom**2).sum(axis=0) / rho).sum()
        print(f"  statistics: level {lv.lev} done ({lv.nboxes} boxes)", flush=True)
    length = (levels[0].domain_hi[0] + 1) * levels[0].dx
    mean_rho = mass / length**3
    vdisp = math.sqrt(max(0.0, two_ekin / mass - float(p @ p) / mass**2))
    return mean_rho, vdisp


def box_str(lo, hi, itype=(0, 0, 0)) -> str:
    return f"(({lo[0]},{lo[1]},{lo[2]}) ({hi[0]},{hi[1]},{hi[2]}) ({itype[0]},{itype[1]},{itype[2]}))"


class VisMFWriter:
    """Write one AMReX MultiFab in VisMF Version_v1 format (per-FAB headers, NFiles layout)."""

    def __init__(self, prefix: str, ncomp: int, nboxes: int, nfiles: int, itype=(0, 0, 0)):
        self.prefix = prefix  # e.g. chk/Level_0/Cell
        self.ncomp = ncomp
        self.itype = itype
        self.nfiles = max(1, min(nfiles, nboxes))
        self.boxes_per_file = math.ceil(nboxes / self.nfiles)
        self.fabs: list[tuple[str, int]] = []
        self.boxes: list[str] = []
        self.mins: list[np.ndarray] = []
        self.maxs: list[np.ndarray] = []
        self._fh = None
        self._ifile = -1

    def _file_for(self, ifab: int):
        ifile = ifab // self.boxes_per_file
        if ifile != self._ifile:
            if self._fh is not None:
                self._fh.close()
            self._ifile = ifile
            self._fname = f"{os.path.basename(self.prefix)}_D_{ifile:05d}"
            self._fh = open(os.path.join(os.path.dirname(self.prefix), self._fname), "wb")
        return self._fh

    def write(self, lo, hi, data: np.ndarray):
        """data has shape (ncomp, nz, ny, nx) for the (possibly nodal) box [lo, hi]."""
        assert data.shape[0] == self.ncomp
        fh = self._file_for(len(self.fabs))
        offset = fh.tell()
        bstr = box_str(lo, hi, self.itype)
        fh.write(f"FAB {REAL_DESCRIPTOR}{bstr} {self.ncomp}\n".encode())
        fh.write(np.ascontiguousarray(data, dtype="<f8").tobytes())
        self.fabs.append((self._fname, offset))
        self.boxes.append(bstr)
        flat = data.reshape(self.ncomp, -1)
        self.mins.append(flat.min(axis=1))
        self.maxs.append(flat.max(axis=1))

    def close(self):
        if self._fh is not None:
            self._fh.close()
        n = len(self.fabs)
        with open(self.prefix + "_H", "w") as h:
            h.write("1\n")  # VisMF::Header::Version_v1
            h.write("1\n")  # VisMF::NFiles
            h.write(f"{self.ncomp}\n")
            h.write("0\n")  # no ghost cells
            h.write(f"({n} 0\n")
            for b in self.boxes:
                h.write(b + "\n")
            h.write(")\n")
            h.write(f"{n}\n")
            for name, off in self.fabs:
                h.write(f"FabOnDisk: {name} {off}\n")
            h.write("\n")
            for arr in (self.mins, self.maxs):
                h.write(f"{n},{self.ncomp}\n")
                for a in arr:
                    h.write("".join(f"{x:.17e}," for x in a) + "\n")
                h.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ORION2 Chombo HDF5 checkpoint (e.g. chk.0000.3d.hdf5)")
    ap.add_argument("output", help="output Quokka checkpoint directory (must not exist)")
    ap.add_argument("--length", type=float, required=True, help="physical box length L [cm]")
    ap.add_argument("--mean-density", type=float, required=True, help="volume-mean gas density [g cm^-3]")
    ap.add_argument("--vdisp", type=float, required=True, help="mass-weighted 3D velocity dispersion [cm s^-1]")
    ap.add_argument("--max-level", type=int, default=None, help="drop AMR levels above this one (default: keep all)")
    ap.add_argument("--no-passive-scalar", action="store_true", help="do not write ORION2 tracer1 as a passive scalar")
    ap.add_argument("--no-radiation", action="store_true", help="do not write radiation variables")
    ap.add_argument(
        "--erad-temperature",
        type=float,
        default=None,
        help="set Erad = a T^4 uniformly with this temperature [K] instead of rescaling ORION2 Erad as an energy density",
    )
    ap.add_argument("--reset-time", action="store_true", help="start the Quokka run at t = 0 instead of the rescaled ORION2 time")
    ap.add_argument("--nfiles", type=int, default=64, help="maximum number of binary data files per MultiFab (default 64)")
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"error: output '{args.output}' already exists")

    f = h5py.File(args.input, "r")
    max_level = int(f.attrs["max_level"]) if args.max_level is None else args.max_level
    levels, comp = read_hierarchy(f, max_level)
    ncomp_in = int(f.attrs["num_components"])
    n0 = levels[0].domain_hi + 1
    if not (n0[0] == n0[1] == n0[2]):
        sys.exit(f"error: base grid {n0} is not cubic")
    length_orion = n0[0] * levels[0].dx
    time_orion = float(f.attrs["time"])
    print(f"ORION2 checkpoint {args.input}: {len(levels)} levels, base grid {n0.tolist()}, L = {length_orion:.6e}, t = {time_orion:.6e}")

    print("pass 1: composite statistics")
    mean_rho_orion, vdisp_orion = composite_statistics(f, levels, comp)
    print(f"  ORION2 volume-mean density = {mean_rho_orion:.10e}, mass-weighted vdisp = {vdisp_orion:.10e}")

    f_len = args.length / length_orion
    f_rho = args.mean_density / mean_rho_orion
    f_v = args.vdisp / vdisp_orion
    f_t = f_len / f_v
    f_mom = f_rho * f_v
    f_e = f_rho * f_v**2
    f_b = math.sqrt(f_rho) * f_v
    print(f"  scale factors: length {f_len:.6e}, density {f_rho:.6e}, velocity {f_v:.6e}, time {f_t:.6e}, B {f_b:.6e}")

    keep_scalar = not args.no_passive_scalar
    keep_rad = not args.no_radiation
    ncomp_out = 6 + int(keep_scalar) + 4 * int(keep_rad)
    erad_uniform = None if args.erad_temperature is None else A_RAD * args.erad_temperature**4

    os.makedirs(args.output)
    print("pass 2: writing checkpoint")
    for i, lv in enumerate(levels):
        levdir = os.path.join(args.output, f"Level_{lv.lev}")
        os.makedirs(levdir)
        cell = VisMFWriter(os.path.join(levdir, "Cell"), ncomp_out, lv.nboxes, args.nfiles)
        faces = [
            VisMFWriter(os.path.join(levdir, f"Face_{d}"), 1, lv.nboxes, args.nfiles, itype=tuple(int(e == idim) for e in range(3)))
            for idim, d in enumerate("xyz")
        ]
        for ib in range(lv.nboxes):
            d = read_cells(f, lv, ib, ncomp_in)
            bf = [b * f_b for b in read_faces(f, lv, ib)]
            nx, ny, nz = lv.shape(ib)
            out = np.zeros((ncomp_out, nz, ny, nx))
            rho = d[comp["rho"]] * f_rho
            mom = np.stack([d[comp[c]] for c in ("mx", "my", "mz")]) * f_mom
            etot = d[comp["E"]] * f_e
            # cell-centred B as the face average, exactly as Quokka computes the magnetic energy
            bcc = np.stack([0.5 * (bf[0][:, :, 1:] + bf[0][:, :, :-1]), 0.5 * (bf[1][:, 1:, :] + bf[1][:, :-1, :]), 0.5 * (bf[2][1:] + bf[2][:-1])])
            ekin = 0.5 * (mom**2).sum(axis=0) / rho
            emag = 0.5 * (bcc**2).sum(axis=0)
            out[0] = rho
            out[1:4] = mom
            out[4] = etot
            out[5] = etot - ekin - emag
            n = 6
            if keep_scalar:
                out[n] = d[comp["tracer"]] * f_rho
                n += 1
            if keep_rad:
                out[n] = erad_uniform if erad_uniform is not None else d[comp["Erad"]] * f_e
                # out[n+1:n+4] (radiation flux) stays zero
            cell.write(lv.lo[ib], lv.hi[ib], out)
            for idim in range(3):
                hi = lv.hi[ib].copy()
                hi[idim] += 1
                faces[idim].write(lv.lo[ib], hi, bf[idim][np.newaxis])
        cell.close()
        for fw in faces:
            fw.close()
        print(f"  level {lv.lev}: {lv.nboxes} boxes written", flush=True)

    time_new = 0.0 if args.reset_time else time_orion * f_t
    finest = len(levels) - 1
    with open(os.path.join(args.output, "Header"), "w") as h:
        h.write("Checkpoint file for QuokkaCode\n")
        h.write(f"{finest}\n")
        h.write(" ".join("0" for _ in levels) + " \n")
        h.write(" ".join(f"{lv.dt * f_t:.17g}" for lv in levels) + " \n")
        h.write(" ".join(f"{time_new:.17g}" for _ in levels) + " \n")
        for lv in levels:
            h.write(f"({lv.nboxes} 0\n")
            for ib in range(lv.nboxes):
                h.write(box_str(lv.lo[ib], lv.hi[ib]) + "\n")
            h.write(")\n")

    with open(os.path.join(args.output, "metadata.yaml"), "w") as m:
        m.write(f"orion2_source: \"{os.path.abspath(args.input)}\"\n")
        m.write(f"orion2_iteration: {int(f.attrs['iteration'])}\n")
        m.write(f"orion2_time: {time_orion:.17g}\n")
        m.write(f"orion2_length: {length_orion:.17g}\n")
        m.write(f"orion2_mean_density: {mean_rho_orion:.17g}\n")
        m.write(f"orion2_vdisp: {vdisp_orion:.17g}\n")
        m.write(f"orion2_target_length: {args.length:.17g}\n")
        m.write(f"orion2_target_mean_density: {args.mean_density:.17g}\n")
        m.write(f"orion2_target_vdisp: {args.vdisp:.17g}\n")
        m.write(f"orion2_scale_length: {f_len:.17g}\n")
        m.write(f"orion2_scale_density: {f_rho:.17g}\n")
        m.write(f"orion2_scale_velocity: {f_v:.17g}\n")
        m.write(f"orion2_scale_time: {f_t:.17g}\n")

    blocking = int(np.gcd.reduce(np.concatenate([(lv.hi - lv.lo + 1).ravel() for lv in levels[1:]] or [np.array([n0[0]])])))
    print(f"\nwrote {args.output} (t = {time_new:.6e} s, crossing time L/vdisp = {args.length / args.vdisp:.6e} s)")
    print("matching Quokka input parameters:")
    print("  geometry.prob_lo = [0.0, 0.0, 0.0]")
    print(f"  geometry.prob_hi = [{args.length:.17g}, {args.length:.17g}, {args.length:.17g}]")
    print("  geometry.is_periodic = [1, 1, 1]")
    print(f"  amr.n_cell = [{n0[0]}, {n0[1]}, {n0[2]}]")
    print(f"  amr.max_level = {finest}  # or larger")
    print("  amr.ref_ratio = 2")
    print(f"  amr.blocking_factor = {min(blocking, 32)}  # largest that divides every refined ORION2 box")
    print(f'  restartfile = "{args.output}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
