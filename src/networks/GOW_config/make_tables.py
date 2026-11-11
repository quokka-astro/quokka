#!/usr/bin/env python3
"""Write gow_tables.hdf5, the tables of the GOW network's lookups, in the layout of quokka::DataTable::H5Reader.

Inputs: jaff's GOW++.hdf5 (CO cooling and gas-dust coupling, the Gong et al. 2017 fits) and the shielding tables jaff
writes next to the generated network (shielding_<reaction>.hdf5). Each output group holds the attributes Ndim, Nout,
Nx, xlo, xhi (physical axis bounds) and spacing, and a dataset `data` of physical values with shape [Nout, Nx...].

DataTable needs grids uniform in each axis's spacing. The CO tables tabulate temperature at irregular points, so they
are resampled onto a log-uniform temperature grid. The resampling follows gow17 (Athena++): its values are -log10 L0,
-log10 L_LTE, log10 n_1/2 and alpha, interpolated linearly in T (and in log10 N for the 2-D tables). DataTable then
interpolates between the fine samples linearly in log T.
"""

import argparse
import glob
import os

import h5py
import numpy as np

# Temperature samples of the resampled CO tables, log-uniform over the gow17 range [10, 2000] K. Fine enough that
# the piecewise-linear-in-T gow17 values change by well under 1e-3 dex between neighbouring samples.
N_TCO = 400


def write_group(f, name, axes, spacing, outputs, input_names, output_names):
    """Write one DataTable group: axes are the physical axis values, outputs the physical values on the grid."""
    g = f.create_group(name)
    g.attrs["Ndim"] = np.int32(len(axes))
    g.attrs["Nout"] = np.int32(len(outputs))
    g.attrs["Nx"] = np.array([len(a) for a in axes], dtype=np.int32)
    g.attrs["xlo"] = np.array([a[0] for a in axes], dtype=np.float64)
    g.attrs["xhi"] = np.array([a[-1] for a in axes], dtype=np.float64)
    g.attrs["spacing"] = np.array(spacing, dtype="S")
    g.attrs["input_names"] = np.array(input_names, dtype="S")
    g.attrs["output_names"] = np.array(output_names, dtype="S")
    for a, s in zip(axes, spacing):
        if s == "log":
            dlog = np.diff(np.log(a))
            assert np.allclose(dlog, dlog[0], rtol=1e-6), f"{name}: axis is not log-uniform"
    g.create_dataset("data", data=np.stack(outputs).astype(np.float64))


def co_tables(f, out):
    co = f["co"]
    assert [s.decode() if isinstance(s, bytes) else s for s in co.attrs["nd_order"]] == ["TCO", "NeffCO"]
    tco = co["TCO"][()].astype(np.float64)
    lgneff = co["NeffCO"][()]
    n_t, n_n = len(tco), len(lgneff)
    t = np.geomspace(tco[0], tco[-1], N_TCO)

    # L0(T): gow17 interpolates -log10 L0 linearly in T
    l0 = 10.0 ** -np.interp(t, tco, co["L0CO"][()])
    write_group(out, "l0", [t], ["log"], [l0], ["T"], ["L0"])

    def resample(name):
        table = co[name][()].reshape(n_t, n_n)
        return np.stack([np.interp(t, tco, table[:, j]) for j in range(n_n)], axis=1)

    llte = 10.0 ** -resample("LLTECO")
    nhalf = 10.0 ** resample("nhalfCO")
    alpha = resample("alphaCO")
    write_group(out, "co", [t, 10.0**lgneff], ["log", "log"], [llte, nhalf, alpha], ["T", "Neff"], ["LLTE", "nhalf", "alpha"])


def dust_table(f, out):
    dust = f["dust"]
    lgnh = np.unique(dust["lognH"][()])
    lgt = np.unique(dust["logTg"][()])
    # flattened with n_H slowest: [n_H, T]
    assert np.allclose(dust["lognH"][()].reshape(len(lgnh), len(lgt))[:, 0], lgnh)
    assert np.allclose(dust["logTg"][()].reshape(len(lgnh), len(lgt))[0, :], lgt)
    psi = 10.0 ** -dust["logps"][()].reshape(len(lgnh), len(lgt))
    write_group(out, "dust", [10.0**lgnh, 10.0**lgt], ["log", "log"], [psi], ["nH", "T"], ["psi_gd"])


def shielding_table(out, path, name):
    with h5py.File(path, "r") as s:
        n = s["N"][()]
        # resample onto an exactly log-uniform grid (the file's is uniform to round-off)
        grid = np.geomspace(n[0], n[-1], len(n))
        by_h2 = np.exp(np.interp(np.log(grid), np.log(n), np.log(s["H2"][()])))
        self_ = np.exp(np.interp(np.log(grid), np.log(n), np.log(s["self"][()])))
    write_group(out, name, [grid], ["log"], [by_h2, self_], ["N"], ["by_H2", "self"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gow-hdf5", required=True, help="jaff's networks/GOW++/GOW++.hdf5")
    parser.add_argument("--outdir", required=True, help="directory of the generated network and its shielding_*.hdf5")
    args = parser.parse_args()

    shielding = {"c_shield": "shielding_C._PHOTON__C+.e-.hdf5", "co_shield": "shielding_CO._PHOTON__C.O.hdf5"}
    with h5py.File(os.path.join(args.outdir, "gow_tables.hdf5"), "w") as out, h5py.File(args.gow_hdf5, "r") as f:
        co_tables(f, out)
        dust_table(f, out)
        for name, file in shielding.items():
            shielding_table(out, os.path.join(args.outdir, file), name)

    # the shielding tables are now in gow_tables.hdf5
    for path in glob.glob(os.path.join(args.outdir, "shielding_*.hdf5")):
        os.remove(path)


if __name__ == "__main__":
    main()
