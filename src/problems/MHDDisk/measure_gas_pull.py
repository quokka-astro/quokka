"""Measure the radial self-gravity of the gas in an MHDDisk snapshot and write it as a table for the run.

The run subtracts the gas's own radial pull from the halo force, so that live self-gravity plus the
modified halo give exactly v_rot^2 / R (addStrangSplitSources). By default that pull is the analytic
thin-disk pull of Sigma(R). This script measures the pull the code actually computes instead: it takes
g = -grad(gpot) with the same central differences the code uses, on every AMR grid, keeps only cells
not covered by a finer level, and averages g_R over azimuth (and +-z) into an (R, |z|) table.

Usage:
    1. Write a t = 0 snapshot with the setup you will run (same toml, grid and ICs):
         mpirun -n 8 ./build/src/problems/MHDDisk/MHDDisk inputs/MHDDisk_laptop.toml max_timesteps=0 \
             plotfile_prefix=tests/plt/pull_plt checkpoint_interval=-1
    2. python src/problems/MHDDisk/measure_gas_pull.py tests/plt/pull_plt0000000 -o tests/input/gas_pull.txt
    3. In the toml: mhd_galaxy.gas_pull_file = "tests/input/gas_pull.txt"
Regenerate the table whenever the disk, grid or refinement changes.

Table format (text): two comment lines, then "nR nz Rmax_cm z0_cm smax", then nR * nz values of g_R
[cm/s^2, < 0 inward], row-major in R. Node i is at R = i * Rmax / (nR - 1); node j is at
|z| = z0 * sinh(j * smax / (nz - 1)), which is fine near the midplane and coarse far above it.
"""

import argparse

import numpy as np
import yt
from scipy.special import i0, i1, k0, k1

KPC = 3.0857e21
PC = KPC / 1e3
G = 6.674e-8


def analytic_pull(R, Sigma0, Rd, alpha=2.0, beta=0.5, terms=12):
    """Thin-disk inward pull of Sigma0 exp(-x - beta exp(-alpha x)), as surfaceDensityPull in the code."""
    pull = np.zeros_like(R)
    c_n = 1.0
    for n in range(terms):
        Rd_n = Rd / (1.0 + alpha * n)
        y = R / (2.0 * Rd_n)
        pull += np.pi * G * Sigma0 * c_n * (R / Rd_n) * (i0(y) * k0(y) - i1(y) * k1(y))
        c_n *= -beta / (n + 1)
    return pull


def fill_empty(values, counts):
    """Fill empty bins along z, then along R, by linear interpolation between filled bins."""
    filled = counts > 0
    out = values.copy()
    for axis in (1, 0):
        lines = np.moveaxis(out, axis, -1)
        ok_lines = np.moveaxis(filled, axis, -1)
        for idx in np.ndindex(lines.shape[:-1]):
            ok = ok_lines[idx]
            if ok.any() and not ok.all():
                x = np.arange(ok.size)
                lines[idx] = np.interp(x, x[ok], lines[idx][ok])
        filled = filled | np.moveaxis(np.isfinite(lines), -1, axis)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plotfile")
    parser.add_argument("-o", "--output", default="tests/input/gas_pull.txt")
    parser.add_argument("--nR", type=int, default=256, help="radial nodes (default 256)")
    parser.add_argument("--nz", type=int, default=128, help="vertical nodes (default 128)")
    parser.add_argument("--z0-pc", type=float, default=50.0, help="vertical stretch scale in pc (default 50)")
    parser.add_argument("--Sigma0", type=float, default=0.03654104674, help="for the comparison printout only [g/cm^2]")
    parser.add_argument("--Rd-kpc", type=float, default=3.0, help="for the comparison printout only")
    args = parser.parse_args()
    yt.set_log_level(40)

    ds = yt.load(args.plotfile)
    ds.force_periodicity()  # lets ghost cells be filled at the domain edge; the edge layer is discarded below
    lo = ds.domain_left_edge.d
    hi = ds.domain_right_edge.d
    Rmax = np.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))
    zmax = max(abs(lo[2]), abs(hi[2]))
    z0 = args.z0_pc * PC
    smax = np.arcsinh(zmax / z0)
    nR, nz = args.nR, args.nz
    dR = Rmax / (nR - 1)
    ds_ = smax / (nz - 1)

    sums = np.zeros((nR, nz))
    wsum = np.zeros((nR, nz))
    for g in ds.index.grids:
        phi = g.retrieve_ghost_zones(1, [("boxlib", "gpot")])[("boxlib", "gpot")].d
        dx = g.dds.d
        gx = -(phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2.0 * dx[0])
        gy = -(phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2.0 * dx[1])
        edges = [g.LeftEdge.d[d] + (np.arange(g.ActiveDimensions[d]) + 0.5) * dx[d] for d in range(3)]
        X, Y, Z = np.meshgrid(*edges, indexing="ij")
        R = np.hypot(X, Y)
        keep = g.child_mask & (R > 0.0)
        for d, C in enumerate((X, Y, Z)):  # drop the outermost cell layer of the domain (wrapped ghost cells)
            keep &= (C > lo[d] + dx[d]) & (C < hi[d] - dx[d])
        gR = (X * gx + Y * gy) / np.where(R > 0.0, R, 1.0)
        i = np.clip(np.rint(R[keep] / dR).astype(int), 0, nR - 1)
        j = np.clip(np.rint(np.arcsinh(np.abs(Z[keep]) / z0) / ds_).astype(int), 0, nz - 1)
        w = np.full(i.size, dx.prod())
        np.add.at(sums, (i, j), w * gR[keep])
        np.add.at(wsum, (i, j), w)

    table = np.where(wsum > 0, sums / np.where(wsum > 0, wsum, 1.0), np.nan)
    table[0, :] = 0.0  # no radial pull on the axis
    wsum[0, :] = 1.0
    table = fill_empty(table, wsum)
    print(f"{(wsum[1:] > 0).mean() * 100:.0f}% of bins measured directly, the rest interpolated")

    with open(args.output, "w") as f:
        f.write(f"# MHDDisk gas self-gravity radial acceleration g_R(R, |z|) [cm/s^2], measured from {args.plotfile}\n")
        f.write("# nR nz Rmax_cm z0_cm smax; then nR*nz values row-major in R; R_i = i Rmax/(nR-1), |z|_j = z0 sinh(j smax/(nz-1))\n")
        f.write(f"{nR} {nz} {Rmax:.10e} {z0:.10e} {smax:.10e}\n")
        np.savetxt(f, table.reshape(-1), fmt="%.10e")
    print(f"wrote {args.output} ({nR} x {nz}, R < {Rmax / KPC:.2f} kpc, |z| < {zmax / KPC:.2f} kpc)")

    # Compare with the analytic thin-disk pull the run uses without this table (midplane)
    print("\n  R [kpc]   measured g_R [cm/s^2]   analytic thin disk   measured / analytic")
    for R_kpc in (0.5, 1, 2, 4, 6, 8, 12):
        i = int(round(R_kpc * KPC / dR))
        meas = -table[i, 0]
        ana = analytic_pull(np.array([i * dR]), args.Sigma0, args.Rd_kpc * KPC)[0]
        print(f"  {R_kpc:6.1f}   {meas:20.4e}   {ana:18.4e}   {meas / ana:12.3f}")


if __name__ == "__main__":
    main()
