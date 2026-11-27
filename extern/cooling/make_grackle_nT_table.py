#!/usr/bin/env python3
"""
Write a Grackle cooling table on its native (nH, T) grid as a DataTable `tab1` HDF5 file.

Outputs:
  Lambda_net  [erg cm^3 s^-1]  (Lambda_prim - Gamma_prim) + Z' (Lambda_metal - Gamma_metal), no photoelectric heating, no Compton.
  xe_CIE      [dimensionless]  electron fraction per H nucleus, 1/(X mu) - (1 + y), with y = (1 - X) / (4 X).

Example usage (downloads CloudyData_UVB=HM2012_shielded.h5). This command was used to generate the CloudyData_UVB=HM2012_shielded_nT_Z1.h5 file in this folder.
    ./make_grackle_nT_table.py --zmet 1.0 --output CloudyData_UVB=HM2012_shielded_nT_Z1.h5
Use a local Grackle file
    ./make_grackle_nT_table.py /path/to/CloudyData_UVB=HM2012_shielded.h5 --zmet 0.5 --output my_table.h5
"""

import argparse
import os
import tempfile

import h5py
import numpy as np

from grackle_tables import X, m_H, read_tables


def make_table(grackle_file, zmet, output_file):
    """Write the (nH, T) table for metallicity `zmet` (in units of Z') from `grackle_file` to `output_file`."""
    tables = read_tables(grackle_file)

    # read_tables returns log10(rate / m_H^2); undo that to get erg cm^3 s^-1.
    physical = lambda log_table: 10.0**log_table * m_H**2
    lambda_prim = physical(tables.primCooling) - physical(tables.primHeating)
    lambda_metal = physical(tables.metalCooling) - physical(tables.metalHeating)
    lambda_net = lambda_prim + zmet * lambda_metal

    y_he = (1.0 - X) / (4.0 * X)
    xe_cie = 1.0 / (X * tables.mmw) - (1.0 + y_he)

    n_nH, n_T = lambda_net.shape
    xlo = np.array([10.0**tables.log_nH[0], 10.0**tables.log_T[0]], dtype=np.float64)
    xhi = np.array([10.0**tables.log_nH[-1], 10.0**tables.log_T[-1]], dtype=np.float64)

    with h5py.File(output_file, 'w') as f:
        tab1 = f.create_group('tab1')
        tab1.attrs.create('Ndim', 2, dtype='i4')
        tab1.attrs.create('Nx', np.array([n_nH, n_T], dtype=np.int32))
        tab1.attrs.create('Nout', 2, dtype='i4')
        tab1.attrs.create('input_names', np.array(['nH', 'T'], dtype='S'))
        tab1.attrs.create('output_names', np.array(['Lambda_net', 'xe_CIE'], dtype='S'))
        tab1.attrs.create('input_units', np.array(['cm^-3', 'K'], dtype='S'))
        tab1.attrs.create('output_units', np.array(['erg cm^3/s', '1'], dtype='S'))
        tab1.attrs.create('xlo', xlo)
        tab1.attrs.create('xhi', xhi)
        tab1.attrs.create('spacing', np.array(['log', 'log'], dtype='S'))
        tab1.attrs.create('zmet', np.float64(zmet))
        tab1.attrs.create('H_mass_fraction', np.float64(X))
        tab1.attrs.create('source_file', np.bytes_(os.path.basename(grackle_file)))
        tab1.create_dataset('data', data=np.stack([lambda_net, xe_cie], axis=0))


def main():
    parser = argparse.ArgumentParser(description='Write a Grackle (nH, T) table with net cooling and CIE electron fraction')
    parser.add_argument('grackle_file', type=str, nargs='?',
                        help='Path to Grackle HDF5 cooling table. If omitted, downloads CloudyData_UVB=HM2012_shielded.h5.')
    parser.add_argument('--zmet', type=float, default=1.0, help='Gas metallicity scaled to Solar (default: 1.0)')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file name')
    args = parser.parse_args()

    if args.grackle_file:
        make_table(args.grackle_file, args.zmet, args.output)
        return

    from resample_grackle_cooling_tables import DEFAULT_GRACKLE_DATA_URL_SHIELDED, download_default_grackle_tables
    with tempfile.TemporaryDirectory() as tmpdir:
        local = os.path.join(tmpdir, 'CloudyData_UVB=HM2012_shielded.h5')
        try:
            download_default_grackle_tables(DEFAULT_GRACKLE_DATA_URL_SHIELDED, local=local)
        except RuntimeError as err:
            parser.error(str(err))
        make_table(local, args.zmet, args.output)


if __name__ == '__main__':
    main()
