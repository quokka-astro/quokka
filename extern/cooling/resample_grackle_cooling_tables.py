#!/usr/bin/env python3
"""
Resample Grackle cooling tables as a function of specific internal energy and mass density on a logarithmic 2D grid.
Outputs include cooling rates, temperatures, sound speeds, pressures, and entropies.

Example usage:

The CloudyData_UVB=HM2012_resampled.h5 file in this folder was generated via this command, using the default CloudyData_UVB=HM2012.h5 data (auto-downloaded via wget) at solar metallicity
    ./resample_grackle_cooling_tables.py --output CloudyData_UVB=HM2012_resampled.h5

The CloudyData_UVB=HM2012_resampled_no_PE.h5 file in this folder was generated via this command:
    ./resample_grackle_cooling_tables.py --output CloudyData_UVB=HM2012_resampled_no_PE.h5 --exclude_pe

The default CloudyData_UVB=HM2012.h5 data at 0.5 solar metallicity:
    ./resample_grackle_cooling_tables.py --zmet 0.5

User-provided tables and non-default spacing parameters
    ./resample_grackle_cooling_tables.py /path/to/CloudyData.h5 --n_rho 50 --n_eint 400 --output my_tables.h5

Validate fast_log2 <-> inverse_fast_log2
    ./resample_grackle_cooling_tables.py --test
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import h5py

from grackle_tables import (
    read_tables, cooling_rate, interpolate_mu, compute_temperature_from_nH_e,
    m_H, boltzmann_constant_cgs_, cloudy_H_mass_fraction, specific_energy_from_temperature
)


DEFAULT_GRACKLE_DATA_URL = (
    "https://github.com/grackle-project/grackle_data_files/raw/"
    "928696482fbe15d9bac4382de6134d95568f099c/input/CloudyData_UVB=HM2012.h5"
)


def fast_log2(x):
    """Fast approximation of log2(x) using the not-quite-logarithmic method.
    This implements the same algorithm as FastMath::fastlg in the C++ code.

    Args:
        x: positive number or array
    Returns:
        Approximation of log2(x)
    """
    if np.any(x <= 0):
        raise ValueError("log divergent for x <= 0")
    # For scalar or array inputs
    x = np.asarray(x)    
    # frexp returns (mantissa, exponent) where x = mantissa * 2^exponent
    # and 0.5 <= mantissa < 1.0
    mantissa, exponent = np.frexp(x)
    # The not-quite-log approximation: log2(x) ≈ 2(mantissa - 1) + exponent
    return 2 * (mantissa - 1) + exponent


def inverse_fast_log2(y):
    """Inverse of the fast_log2 function using numerical root-finding to machine precision.
    
    This function finds x such that fast_log2(x) = y by solving the equation
    f(x) = fast_log2(x) - y = 0 using scipy's root-finding methods with machine precision.

    Args:
        y: log2 value or array
    Returns:
        x such that fast_log2(x) = y (accurate to machine precision)
    """
    from scipy.optimize import brentq
    
    # Handle scalar vs array input
    y = np.asarray(y)
    scalar_input = y.ndim == 0
    y = np.atleast_1d(y)
    
    result = np.zeros_like(y)
    
    # Machine epsilon for float64
    eps = np.finfo(np.float64).eps
    
    for i, y_val in enumerate(y):
        # Define the function to find the root of
        def f(x):
            if x <= 0:
                return np.inf  # fast_log2 is undefined for x <= 0
            return fast_log2(x) - y_val
        
        # Get initial bounds for the root search
        # Since fast_log2(x) ≈ log2(x), we can use 2^y as an initial guess
        x_guess = 2.0 ** y_val
        
        # Use a bracket around the guess
        # The fast_log2 function is monotonic, so we can bracket the root
        x_lower = x_guess * 0.5
        x_upper = x_guess * 2.0
        
        # Use Brent's method for robust root finding with tight tolerances
        # scipy requires rtol >= 4*eps and xtol > 0
        # We use the tightest allowed tolerances
        x_sol = brentq(f, x_lower, x_upper, xtol=eps*x_guess, rtol=4*eps, maxiter=1000)
        
        # Verify the solution
        residual = abs(fast_log2(x_sol) - y_val)
        tolerance = eps * abs(y_val) * 20
        if residual > tolerance:
            # Try Newton-Raphson refinement
            # Derivative of fast_log2(x) ≈ 1/(x * ln(2))
            for _ in range(5):  # Multiple refinement steps
                dx = -f(x_sol) * x_sol * np.log(2)
                x_sol_new = x_sol + dx
                if abs(f(x_sol_new)) < abs(f(x_sol)):
                    x_sol = x_sol_new
                else:
                    break
            
            # Final check with adjusted tolerance
            residual = abs(fast_log2(x_sol) - y_val)
            if residual > tolerance:
                print(f"Warning: inverse_fast_log2({y_val}) has residual {residual:.2e}")
        
        result[i] = x_sol
    
    # Return scalar if input was scalar
    if scalar_input:
        return result[0]
    return result


def compute_sound_speed(rho, T, mu, gamma=5./3.):
    """Compute sound speed from density, temperature, and mean molecular weight.
    
    Args:
        rho: density (g/cm^3)
        T: temperature (K)
        mu: mean molecular weight in units of m_H
        gamma: adiabatic index
    
    Returns:
        cs: sound speed (cm/s)
    """
    # For ideal gas: cs = sqrt(gamma * k_B * T / (mu * m_H))
    cs = np.sqrt(gamma * boltzmann_constant_cgs_ * T / (mu * m_H))
    return cs


def compute_pressure(rho, T, mu):
    """Compute pressure from density, temperature, and mean molecular weight.
    
    Args:
        rho: density (g/cm^3)
        T: temperature (K)
        mu: mean molecular weight in units of m_H
    
    Returns:
        P: pressure (dyne/cm^2 = erg/cm^3)
    """
    # For ideal gas: P = rho * k_B * T / (mu * m_H)
    P = rho * boltzmann_constant_cgs_ * T / (mu * m_H)
    return P


def compute_entropy(rho, T, mu):
    """Compute entropy as K = k_B * T * n^(-2/3).
    
    Args:
        rho: density (g/cm^3)
        T: temperature (K)
        mu: mean molecular weight in units of m_H
    
    Returns:
        K: entropy K = k_B * T * n^(-2/3) (erg * cm^2)
    """
    # Compute number density n = rho / (mu * m_H)
    n = rho / (mu * m_H)
    # Compute entropy K = k_B * T * n^(-2/3)
    K = boltzmann_constant_cgs_ * T * (n ** (-2./3.))
    return K


def find_eint_range(tables):
    """Find the range of specific internal energies for a given table.

    Args:
        tables: the Grackle cooling tables
    
    Returns:
        tuple: (eint_min, eint_max) in erg/g
    """
    # Set default temperature bounds
    T_min = np.max([10.0 ** tables.log_T[0], 25.0]) # set 25 K temperature floor
    T_max = 10.0 ** tables.log_T[-1]
    mu_min = np.min(tables.mmw)
    mu_max = np.max(tables.mmw)
    
    # Check whether temperature is (obviously) out-of-bounds
    eint_min = specific_energy_from_temperature(T_min, mu_max)
    eint_max = specific_energy_from_temperature(T_max, mu_min)

    print(f"Mean molecular weight range: {mu_min:.3f} to {mu_max:.3f}")
    print(f"Specific internal energy range: {eint_min:.2e} to {eint_max:.2e} erg/g")
    
    return eint_min, eint_max


def resample_cooling_tables(grackle_file, n_rho=100, n_eint=100, zmet=1.0,
                            output_file='resampled_cooling_tables.h5', include_pe=True):
    """Resample cooling tables on a not-quite-logarithmic grid of density and specific internal energy.
    Uses the fast logarithm approximation from https://arxiv.org/pdf/2206.08957 for grid spacing.
    
    Computes and stores:
    - Cooling rates (erg/cm^3/s)
    - Temperatures (K)
    - Sound speeds (cm/s)
    - Pressures (dyne/cm^2)
    - Entropies K = k_B * T * n^(-2/3) (erg*cm^2)
    
    Args:
        grackle_file: path to Grackle HDF5 cooling tables
        n_rho: number of density points
        n_eint: number of specific internal energy points
        output_file: output HDF5 file name
    """
    # Read the original tables
    tables = read_tables(grackle_file)
    print("Table properties:")
    print(f"\tnH len = {len(tables.log_nH)}")
    print(f"\tT len = {len(tables.log_T)}")
    print(f"\tnH min: {10.**tables.log_nH[0]:e}")
    print(f"\tnH max: {10.**tables.log_nH[-1]:e}")
    print(f"\tT min: {10.**tables.log_T[0]:e}")
    print(f"\tT max: {10.**tables.log_T[-1]:e}")
    print("")
    
    rho_from_nH = lambda nH: (nH * m_H) / cloudy_H_mass_fraction
    rho_min = rho_from_nH(10.**tables.log_nH[0])
    rho_max = rho_from_nH(10.**tables.log_nH[-1])

    eint_min, eint_max = find_eint_range(tables)
    
    # Create not-quite-logarithmic grids using fast_log2
    fast_log_rho_scaled = np.linspace(fast_log2(rho_min), fast_log2(rho_max), n_rho)
    fast_log_eint_scaled = np.linspace(fast_log2(eint_min), fast_log2(eint_max), n_eint)
    
    # Convert back to linear space using inverse transform
    rho_grid = inverse_fast_log2(fast_log_rho_scaled)
    eint_grid = inverse_fast_log2(fast_log_eint_scaled)

    # Check that the grid boundaries are correct
    print("Verifying grid boundaries...")
    print(f"  rho_grid[0] = {rho_grid[0]:.15e}, rho_min = {rho_min:.15e}")
    print(f"  rho_grid[-1] = {rho_grid[-1]:.15e}, rho_max = {rho_max:.15e}")
    print(f"  eint_grid[0] = {eint_grid[0]:.15e}, eint_min = {eint_min:.15e}")
    print(f"  eint_grid[-1] = {eint_grid[-1]:.15e}, eint_max = {eint_max:.15e}")
    
    eps = np.finfo(np.float64).eps
    tolerance = eps * 20  # Allow for some numerical error
    
    assert abs(rho_grid[0] - rho_min) <= tolerance * rho_min, f"rho_grid[0]={rho_grid[0]} != rho_min={rho_min}"
    assert abs(rho_grid[-1] - rho_max) <= tolerance * rho_max, f"rho_grid[-1]={rho_grid[-1]} != rho_max={rho_max}"
    assert abs(eint_grid[0] - eint_min) <= tolerance * eint_min, f"eint_grid[0]={eint_grid[0]} != eint_min={eint_min}"
    assert abs(eint_grid[-1] - eint_max) <= tolerance * eint_max, f"eint_grid[-1]={eint_grid[-1]} != eint_max={eint_max}"
    print("✓ Grid boundaries are correct!")
    print("")
    
    # Initialize output arrays
    cooling_rates = np.zeros((n_rho, n_eint))
    temperatures = np.zeros((n_rho, n_eint))
    sound_speeds = np.zeros((n_rho, n_eint))
    pressures = np.zeros((n_rho, n_eint))
    entropies = np.zeros((n_rho, n_eint))
    
    print(f"Resampling cooling tables on {n_rho} x {n_eint} grid using not-quite-logarithmic spacing...")
    print(f"Density range: {rho_min:.2e} to {rho_max:.2e} g/cm^3")
    print(f"Specific energy range: {eint_min:.2e} to {eint_max:.2e} erg/g")
    print(("Including" if include_pe else "Not including") + " photoelectric heating.")
    print("")
    
    # Loop over the grid and compute cooling rates
    for i, rho in enumerate(rho_grid):
        if i % 10 == 0:
            print(f"Processing density point {i+1}/{n_rho}")
        
        # Convert density to hydrogen number density
        nH = rho * cloudy_H_mass_fraction / m_H
        
        for j, e_int in enumerate(eint_grid):
            try:
                T = compute_temperature_from_nH_e(nH, e_int, tables=tables)
                Edot = cooling_rate(nH, T, zmet, redshift=0., tables=tables, include_pe=include_pe)
                mu = interpolate_mu(nH, T, tables=tables)
                cs = compute_sound_speed(rho, T, mu)
                P = compute_pressure(rho, T, mu)
                K = compute_entropy(rho, T, mu)
                temperatures[i, j] = T
                cooling_rates[i, j] = Edot / rho**2
                sound_speeds[i, j] = cs
                pressures[i, j] = P
                entropies[i, j] = K
            except ValueError:
                # Handle extrapolation errors by setting to NaN
                temperatures[i, j] = np.nan
                cooling_rates[i, j] = np.nan
                sound_speeds[i, j] = np.nan
                pressures[i, j] = np.nan
                entropies[i, j] = np.nan
    
    # Save resampled tables to HDF5 file
    print(f"\nSaving resampled tables to {output_file}")
    
    # Store the actual fast_log values of the grid
    fast_log_rho = fast_log2(rho_grid)
    fast_log_eint = fast_log2(eint_grid)
    
    # Write the HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create groups to organize data
        grids_group = f.create_group('grids')
        data_group = f.create_group('data')
        metadata_group = f.create_group('metadata')
        units_group = metadata_group.create_group('units')
        
        # Store grid data
        grids_group.create_dataset('fast_log_rho', data=fast_log_rho)
        grids_group.create_dataset('fast_log_eint', data=fast_log_eint)
        grids_group.create_dataset('rho', data=rho_grid)
        grids_group.create_dataset('eint', data=eint_grid)
        
        # Store computed data
        data_group.create_dataset('cooling_rates', data=cooling_rates)
        data_group.create_dataset('temperatures', data=temperatures)
        data_group.create_dataset('sound_speeds', data=sound_speeds)
        data_group.create_dataset('pressures', data=pressures)
        data_group.create_dataset('entropies', data=entropies)
        
        # Store metadata as attributes
        metadata_group.attrs['n_rho'] = n_rho
        metadata_group.attrs['n_eint'] = n_eint
        metadata_group.attrs['rho_min'] = rho_min
        metadata_group.attrs['rho_max'] = rho_max
        metadata_group.attrs['eint_min'] = eint_min
        metadata_group.attrs['eint_max'] = eint_max
        metadata_group.attrs['cloudy_H_mass_fraction'] = cloudy_H_mass_fraction
        metadata_group.attrs['description'] = 'Cooling rates resampled on (rho, e_int) grid using not-quite-logarithmic spacing'
        metadata_group.attrs['spacing_method'] = 'not-quite-logarithmic (fast log2 approximation)'
        
        # Store units as attributes
        units_group.attrs['rho'] = 'g/cm^3'
        units_group.attrs['eint'] = 'erg/g'
        units_group.attrs['cooling_rate'] = 'erg/cm^3/s/(g/cm^3)^2'
        units_group.attrs['temperature'] = 'K'
        units_group.attrs['sound_speed'] = 'cm/s'
        units_group.attrs['pressure'] = 'dyne/cm^2'
        units_group.attrs['entropy'] = 'erg*cm^2'
    
    print("Done!")
    
    # Print some statistics
    valid_mask = ~np.isnan(cooling_rates)
    if np.any(valid_mask):
        print(f"\nStatistics:")
        print(f"Valid cooling rates: {np.sum(valid_mask)} / {cooling_rates.size}")
        print(f"Temperature range: {np.min(temperatures[valid_mask]):.2e} to {np.max(temperatures[valid_mask]):.2e} K")
        print(f"Cooling rate range: {np.min(cooling_rates[valid_mask]):.2e} to {np.max(cooling_rates[valid_mask]):.2e} erg/cm^3/s")
        print(f"Sound speed range: {np.min(sound_speeds[valid_mask]):.2e} to {np.max(sound_speeds[valid_mask]):.2e} cm/s")
        print(f"Pressure range: {np.min(pressures[valid_mask]):.2e} to {np.max(pressures[valid_mask]):.2e} dyne/cm^2")
        print(f"Entropy range: {np.min(entropies[valid_mask]):.2e} to {np.max(entropies[valid_mask]):.2e} erg*cm^2")


def test_inverse_fast_log2():
    """Test that inverse_fast_log2 correctly inverts fast_log2 to machine precision."""
    print("Testing inverse_fast_log2...")
    
    # Test various ranges of x values
    test_values = [
        1e-10, 1e-5, 0.001, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0, 
        100.0, 1e5, 1e10, 1e15, 1e20
    ]
    
    # Also test array input
    test_array = np.logspace(-10, 20, 100)
    
    max_error = 0.0
    eps = np.finfo(np.float64).eps
    
    # Test scalar values
    for x in test_values:
        y = fast_log2(x)
        x_recovered = inverse_fast_log2(y)
        error = abs(x - x_recovered) / x
        max_error = max(max_error, error)
        
        if error > eps * 10:  # Allow for some numerical error accumulation
            print(f"  x = {x:.6e}, fast_log2(x) = {y:.6e}, recovered x = {x_recovered:.6e}, rel error = {error:.2e}")
    
    # Test array input
    y_array = fast_log2(test_array)
    x_recovered_array = inverse_fast_log2(y_array)
    errors = abs(test_array - x_recovered_array) / test_array
    max_error = max(max_error, np.max(errors))
    
    print(f"Maximum relative error: {max_error:.2e}")
    print(f"Machine epsilon: {eps:.2e}")
    
    tolerance_factor = 20
    if max_error < eps * tolerance_factor:
        print(f"✓ inverse_fast_log2 successfully inverts fast_log2 to near machine precision!")
        print(f"  Maximum error is {max_error/eps:.1f}x machine epsilon")
    else:
        print(f"✗ inverse_fast_log2 error exceeds {tolerance_factor}x machine precision")
        # Show worst cases
        worst_idx = np.argmax(errors)
        print(f"  Worst case: x = {test_array[worst_idx]:.6e}, error = {errors[worst_idx]:.2e}")
    
    return max_error < eps * tolerance_factor


def download_default_grackle_tables(url: str = DEFAULT_GRACKLE_DATA_URL) -> str:
    """Download the default Grackle tables via wget and return the temporary file path."""
    if shutil.which('wget') is None:
        raise RuntimeError("wget is required to download default Grackle tables automatically.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    temp_path = temp_file.name
    temp_file.close()

    print(f"Downloading default Grackle tables from {url}")
    try:
        subprocess.run(['wget', '-O', temp_path, url], check=True)
    except subprocess.CalledProcessError as exc:
        os.unlink(temp_path)
        raise RuntimeError("Failed to download default Grackle tables via wget.") from exc

    return temp_path


def main():
    """Main function to run the resampling."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Resample Grackle cooling tables on (rho, e_int) grid')

    parser.add_argument('grackle_file', type=str, nargs='?',
                        help=('Path to Grackle HDF5 cooling table file. '
                              'If omitted, downloads the default HM2012 tables automatically.'))

    # Mode selection
    parser.add_argument('--test', action='store_true',
                        help='Test the inverse_fast_log2 function')

    # Parameters for resampling
    parser.add_argument('--n_rho', type=int, default=30,
                        help='Number of density points (default: 30)')
    parser.add_argument('--n_eint', type=int, default=200,
                        help='Number of specific energy points (default: 200)')
    parser.add_argument('--output', type=str, default='resampled_cooling_tables.h5',
                        help='Output HDF5 file name (default: resampled_cooling_tables.h5)')
    parser.add_argument('--exclude_pe', action='store_true',
                        help='Exclude photoelectric heating (default: False, i.e., include PE heating)')
    parser.add_argument('--zmet', type=float, default=1.0,
                        help='Gas Metallicity scaled to Solar (default: 1.0)')

    args = parser.parse_args()

    if args.test:
        # Run the test
        success = test_inverse_fast_log2()
        sys.exit(0 if success else 1)

    cleanup_path = None
    grackle_file = args.grackle_file
    if not grackle_file:
        try:
            cleanup_path = download_default_grackle_tables()
            grackle_file = cleanup_path
        except RuntimeError as err:
            parser.error(str(err))

    try:
        resample_cooling_tables(
            grackle_file,
            n_rho=args.n_rho,
            n_eint=args.n_eint,
            zmet=args.zmet,
            output_file=args.output,
            include_pe=not args.exclude_pe
        )
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except FileNotFoundError:
                # It is safe to ignore if the file is already deleted or was never created.
                pass


if __name__ == '__main__':
    main()
