#!/usr/bin/env python3
"""Convert cooling HDF5 files from old format to the DataTable PRD format.

Old format:
  /data/{cooling_rates,temperatures,sound_speeds,pressures,entropies}: (n_rho, n_eint)
  /grids/{rho,eint,fast_log_rho,fast_log_eint}
  /metadata: attrs {n_rho, n_eint, rho_min, rho_max, eint_min, eint_max, include_pe, ...}

New DataTable format (see prd-HDF5-datatable-for-cooling.md):
  /data: (group)
    attrs: Ndim, Nx, Nout, input_names, output_names, input_units, output_units, xlo, xhi, spacing
    /data: shape (Nout, n_rho, n_eint) -- outputs in order: cooling_rate, temperature,
                                          sound_speed, pressure, entropy
  /grids: (optional, for irregular grids)
    /rho, /eint
  file-level attr: include_pe
"""

import sys
import os
import numpy as np
import h5py

OUTPUT_NAMES = ["cooling_rate", "temperature", "sound_speed", "pressure", "entropy"]
OUTPUT_UNITS = [
    "erg/cm^3/s/(g/cm^3)^2",
    "K",
    "cm/s",
    "dyne/cm^2",
    "erg*cm^2",
]
OLD_DATASET_NAMES = ["cooling_rates", "temperatures", "sound_speeds", "pressures", "entropies"]
INPUT_NAMES = ["rho", "eint"]
INPUT_UNITS = ["g/cm^3", "erg/g"]
SPACING = ["fast_log", "fast_log"]


def convert_file(input_path: str) -> None:
    """Convert a single cooling HDF5 file to the DataTable format in-place."""
    with h5py.File(input_path, "r") as f:
        meta = dict(f["metadata"].attrs)
        n_rho = int(meta["n_rho"])
        n_eint = int(meta["n_eint"])
        rho_min = float(meta["rho_min"])
        rho_max = float(meta["rho_max"])
        eint_min = float(meta["eint_min"])
        eint_max = float(meta["eint_max"])
        include_pe = int(meta.get("include_pe", 0))

        rho_grid = f["grids"]["rho"][:]
        eint_grid = f["grids"]["eint"][:]

        # Stack the 5 outputs into a single (5, n_rho, n_eint) array
        data_arrays = []
        for name in OLD_DATASET_NAMES:
            arr = f["data"][name][:]
            if arr.shape != (n_rho, n_eint):
                raise ValueError(f"{name}: expected ({n_rho}, {n_eint}), got {arr.shape}")
            data_arrays.append(arr)
        combined = np.stack(data_arrays, axis=0)  # shape: (5, n_rho, n_eint)

    # Write new format (overwrites old file)
    with h5py.File(input_path, "w") as f:
        f.attrs["include_pe"] = include_pe

        g = f.create_group("data")
        g.attrs.create("Ndim", 2, dtype="i4")
        g.attrs.create("Nx", np.array([n_rho, n_eint], dtype="i4"))
        g.attrs.create("Nout", 5, dtype="i4")
        g.attrs.create("input_names", np.array(INPUT_NAMES, dtype="S"))
        g.attrs.create("output_names", np.array(OUTPUT_NAMES, dtype="S"))
        g.attrs.create("input_units", np.array(INPUT_UNITS, dtype="S"))
        g.attrs.create("output_units", np.array(OUTPUT_UNITS, dtype="S"))
        g.attrs.create("xlo", np.array([rho_min, eint_min], dtype="f8"))
        g.attrs.create("xhi", np.array([rho_max, eint_max], dtype="f8"))
        g.attrs.create("spacing", np.array(SPACING, dtype="S"))
        g.create_dataset("data", data=combined)

        grids = f.create_group("grids")
        grids.create_dataset("rho", data=rho_grid)
        grids.create_dataset("eint", data=eint_grid)

    print(f"Converted {os.path.basename(input_path)}: "
          f"({n_rho} x {n_eint}), include_pe={include_pe}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        "CloudyData_UVB=HM2012_resampled.h5",
        "CloudyData_UVB=HM2012_resampled_no_PE.h5",
        "CloudyData_UVB=HM2012_resampled_noPE.h5",
        "CloudyData_UVB=HM2012_shielded_resampled.h5",
        "CloudyData_UVB=HM2012_shielded_resampled_noPE.h5",
        "isrf_1000Go_grains_resampled.h5",
    ]

    for fname in files:
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"Skipping (not found): {fname}")
            continue
        convert_file(path)


if __name__ == "__main__":
    main()
