#!/usr/bin/env python3
"""Convert old-format resampled cooling HDF5 files to the new tab1 group format.

Old format layout:
  /metadata  (attrs: n_rho, n_eint, rho_min, rho_max, eint_min, eint_max,
                     cloudy_H_mass_fraction, [include_pe])
  /grids/rho, /grids/eint
  /data/cooling_rates, temperatures, sound_speeds, pressures, entropies

New format layout (written to group 'tab1'):
  /tab1  (attrs: Ndim=2, Nout=5, Nx, xlo, xhi, spacing,
                 input_names, output_names, input_units, output_units,
                 include_pe, cloudy_H_mass_fraction)
  /tab1/data   shape [5, n_rho, n_eint]
  /tab1/grids/rho, /tab1/grids/eint

Usage:
    python3 convert_cooling_table_hdf5.py file.h5 [file2.h5 ...]
    python3 convert_cooling_table_hdf5.py file.h5 --output converted.h5
    python3 convert_cooling_table_hdf5.py file.h5 --include_pe 1
"""

import argparse
import os
import sys

import h5py
import numpy as np


def convert(src_path: str, dst_path: str, include_pe_override=None) -> None:
    with h5py.File(src_path, 'r') as src:
        meta = src['metadata']

        n_rho  = int(meta.attrs['n_rho'])
        n_eint = int(meta.attrs['n_eint'])
        rho_min  = float(meta.attrs['rho_min'])
        rho_max  = float(meta.attrs['rho_max'])
        eint_min = float(meta.attrs['eint_min'])
        eint_max = float(meta.attrs['eint_max'])
        cloudy_H_mass_fraction = float(meta.attrs['cloudy_H_mass_fraction'])

        if include_pe_override is not None:
            include_pe = int(include_pe_override)
        elif 'include_pe' in meta.attrs:
            include_pe = int(meta.attrs['include_pe'])
        else:
            include_pe = 0

        cooling_rates = src['data/cooling_rates'][:]
        temperatures  = src['data/temperatures'][:]
        sound_speeds  = src['data/sound_speeds'][:]
        pressures     = src['data/pressures'][:]
        entropies     = src['data/entropies'][:]

        rho_grid  = src['grids/rho'][:]
        eint_grid = src['grids/eint'][:]

    all_data = np.stack(
        [cooling_rates, temperatures, sound_speeds, pressures, entropies], axis=0
    )
    assert all_data.shape == (5, n_rho, n_eint)

    tmp = dst_path + '.tmp'
    with h5py.File(tmp, 'w') as dst:
        tab1 = dst.create_group('tab1')

        tab1.attrs.create('Ndim',  np.int32(2))
        tab1.attrs.create('Nout',  np.int32(5))
        tab1.attrs.create('Nx',    np.array([n_rho, n_eint], dtype=np.int32))
        tab1.attrs.create('xlo',   np.array([rho_min,  eint_min], dtype=np.float64))
        tab1.attrs.create('xhi',   np.array([rho_max,  eint_max], dtype=np.float64))
        tab1.attrs.create('spacing',
                          np.array(['fast_log', 'fast_log'], dtype='S'))
        tab1.attrs.create('input_names',
                          np.array(['rho', 'eint'], dtype='S'))
        tab1.attrs.create('output_names',
                          np.array(['cooling_rate', 'temperature', 'sound_speed',
                                    'pressure', 'entropy'], dtype='S'))
        tab1.attrs.create('input_units',
                          np.array(['g/cm^3', 'erg/g'], dtype='S'))
        tab1.attrs.create('output_units',
                          np.array(['erg/cm^3/s/(g/cm^3)^2', 'K', 'cm/s',
                                    'dyne/cm^2', 'erg*cm^2'], dtype='S'))
        tab1.attrs.create('include_pe',             np.int32(include_pe))
        tab1.attrs['cloudy_H_mass_fraction'] = cloudy_H_mass_fraction

        tab1.create_dataset('data', data=all_data)

        grids = tab1.create_group('grids')
        grids.create_dataset('rho',  data=rho_grid)
        grids.create_dataset('eint', data=eint_grid)

    os.replace(tmp, dst_path)
    print(f"Converted: {src_path} -> {dst_path}  "
          f"(Nx=[{n_rho},{n_eint}], include_pe={include_pe})")


def main():
    parser = argparse.ArgumentParser(
        description='Convert old-format resampled cooling HDF5 files to tab1 format.')
    parser.add_argument('files', nargs='+', help='Input HDF5 file(s)')
    parser.add_argument('--output', '-o',
                        help='Output path (only valid with a single input file; '
                             'default: in-place)')
    parser.add_argument('--include_pe', type=int, choices=[0, 1], default=None,
                        help='Override include_pe value (0 or 1); '
                             'default: read from metadata, or 0 if absent')
    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        parser.error('--output can only be used with a single input file')

    for fpath in args.files:
        dst = args.output if args.output else fpath
        convert(fpath, dst, include_pe_override=args.include_pe)


if __name__ == '__main__':
    main()
