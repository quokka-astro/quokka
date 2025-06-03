#!/usr/bin/env python3
# ABOUTME: Resample cooling tables from grackle_tables.py as a function of specific
# ABOUTME: internal energy and mass density on a logarithmic 2D grid.

import numpy as np
import asdf

from grackle_tables import (
    read_tables, cooling_rate, interpolate_mu, 
    m_H, boltzmann_constant_cgs_, cloudy_H_mass_fraction
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


def fast_log10(x):
    """Fast approximation of log10(x) using the not-quite-logarithmic method.
    Args:
        x: positive number or array
    Returns:
        Approximation of log10(x)
    """
    LOG2_TO_LOG10 = 0.301029995663981195  # log10(2)
    return LOG2_TO_LOG10 * fast_log2(x)


def inverse_fast_log10(y):
    """Inverse of the fast_log10 function.
    Args:
        y: log10 value or array
    Returns:
        x such that fast_log10(x) ≈ y
    """
    # Convert from log10 to log2
    LOG10_TO_LOG2 = 1.0 / 0.301029995663981195
    y2 = LOG10_TO_LOG2 * y
    
    # Find n (integer part) and fractional part
    n = np.floor(y2).astype(int)
    frac = y2 - n
    
    # Solve: frac = 2(mantissa - 1)
    # mantissa = frac/2 + 1
    mantissa = frac / 2 + 1
    
    # Reconstruct x = mantissa * 2^n
    x = np.ldexp(mantissa, n)
    return x


def temperature_from_specific_energy(e_int, mu):
    """Convert specific internal energy to temperature.
    
    Args:
        e_int: specific internal energy (erg/g)
        mu: mean molecular weight in units of m_H
    
    Returns:
        T: temperature (K)
    """
    # For ideal gas: e_int = (3/2) * k_B * T / (mu * m_H)
    # Solve for T: T = (2/3) * e_int * mu * m_H / k_B
    T = (2.0 / 3.0) * e_int * mu * m_H / boltzmann_constant_cgs_
    return T


def specific_energy_from_temperature(T, mu):
    """Convert temperature to specific internal energy.
    
    Args:
        T: temperature (K)
        mu: mean molecular weight in units of m_H
    
    Returns:
        e_int: specific internal energy (erg/g)
    """
    # For ideal gas: e_int = (3/2) * k_B * T / (mu * m_H)
    e_int = (3.0 / 2.0) * boltzmann_constant_cgs_ * T / (mu * m_H)
    return e_int


def find_eint_range(grackle_file, rho_min, rho_max, T_min, T_max):
    """Find the range of specific internal energies for given density and temperature ranges.
    
    This function computes the mean molecular weight for the given density and temperature
    ranges and calculates the corresponding specific internal energy bounds.
    
    Args:
        grackle_file: path to Grackle HDF5 cooling tables
        rho_min: minimum mass density (g/cm^3)
        rho_max: maximum mass density (g/cm^3)
        T_min: minimum temperature (K)
        T_max: maximum temperature (K)
    
    Returns:
        tuple: (eint_min, eint_max) in erg/g
    """
    # Read the tables
    tables = read_tables(grackle_file)
    
    # Convert densities to hydrogen number densities
    nH_min = rho_min * cloudy_H_mass_fraction / m_H
    nH_max = rho_max * cloudy_H_mass_fraction / m_H
    
    # Sample a grid of densities and temperatures to find mu range
    n_samples = 20  # Should be sufficient to capture mu variations
    rho_samples = np.logspace(np.log10(rho_min), np.log10(rho_max), n_samples)
    T_samples = np.logspace(np.log10(T_min), np.log10(T_max), n_samples)
    
    mu_values = []
    
    for rho in rho_samples:
        nH = rho * cloudy_H_mass_fraction / m_H
        for T in T_samples:
            try:
                mu = interpolate_mu(nH, T, tables=tables)
                mu_values.append(mu)
            except:
                # Skip if outside table bounds
                pass
    
    if not mu_values:
        # If no valid mu values found, use typical range
        print("Warning: Could not interpolate mu from tables, using typical values")
        mu_min = 0.6  # Typical for ionized gas
        mu_max = 1.3  # Typical for neutral gas
    else:
        mu_min = min(mu_values)
        mu_max = max(mu_values)
    
    # Calculate specific internal energy bounds
    # Minimum e_int occurs at minimum T and maximum mu
    # Maximum e_int occurs at maximum T and minimum mu
    eint_min = specific_energy_from_temperature(T_min, mu_max)
    eint_max = specific_energy_from_temperature(T_max, mu_min)
    
    print(f"Mean molecular weight range: {mu_min:.3f} to {mu_max:.3f}")
    print(f"Specific internal energy range: {eint_min:.2e} to {eint_max:.2e} erg/g")
    
    return eint_min, eint_max


def resample_cooling_tables(grackle_file, n_rho=100, n_eint=100, 
                          rho_min=1e-30, rho_max=1e-20,
                          eint_min=1e10, eint_max=1e20,
                          output_file='resampled_cooling_tables.asdf'):
    """Resample cooling tables on a not-quite-logarithmic grid of density and specific internal energy.    
    Uses the fast logarithm approximation from https://arxiv.org/pdf/2206.08957 for grid spacing.
    
    Args:
        grackle_file: path to Grackle HDF5 cooling tables
        n_rho: number of density points
        n_eint: number of specific internal energy points
        rho_min: minimum mass density (g/cm^3)
        rho_max: maximum mass density (g/cm^3)
        eint_min: minimum specific internal energy (erg/g)
        eint_max: maximum specific internal energy (erg/g)
        output_file: output ASDF file name
    """
    # Read the original tables
    tables = read_tables(grackle_file)
    
    # Create not-quite-logarithmic grids using fast_log10
    # Create linear spacing in the not-quite-log space
    fast_log_rho = np.linspace(fast_log10(rho_min), fast_log10(rho_max), n_rho)
    fast_log_eint = np.linspace(fast_log10(eint_min), fast_log10(eint_max), n_eint)
    
    # Convert back to linear space using inverse transform
    rho_grid = inverse_fast_log10(fast_log_rho)
    eint_grid = inverse_fast_log10(fast_log_eint)
    
    # Initialize output arrays
    cooling_rates = np.zeros((n_rho, n_eint))
    temperatures = np.zeros((n_rho, n_eint))
    mean_molecular_weights = np.zeros((n_rho, n_eint))
    
    print(f"Resampling cooling tables on {n_rho} x {n_eint} grid using not-quite-logarithmic spacing...")
    print(f"Density range: {rho_min:.2e} to {rho_max:.2e} g/cm^3")
    print(f"Specific energy range: {eint_min:.2e} to {eint_max:.2e} erg/g")
    
    # Loop over the grid and compute cooling rates
    for i, rho in enumerate(rho_grid):
        if i % 10 == 0:
            print(f"Processing density point {i+1}/{n_rho}")
        
        # Convert density to hydrogen number density
        nH = rho * cloudy_H_mass_fraction / m_H
        
        for j, e_int in enumerate(eint_grid):
            # Initial guess for temperature using ideal gas approximation
            # with mu = 1.0 as initial guess
            T_guess = temperature_from_specific_energy(e_int, 1.0)
            
            # Iterate to find consistent temperature and mean molecular weight
            # since mu depends on T through ionization state
            for _ in range(10):  # typically converges in a few iterations
                mu = interpolate_mu(nH, T_guess, tables=tables)
                T_new = temperature_from_specific_energy(e_int, mu)
                if abs(T_new - T_guess) / T_guess < 1e-6:
                    break
                T_guess = T_new
            
            T = T_new
            
            # Store the results
            temperatures[i, j] = T
            mean_molecular_weights[i, j] = mu
            
            # Compute cooling rate
            try:
                Edot = cooling_rate(nH, T, redshift=0., tables=tables)
                cooling_rates[i, j] = Edot
            except:
                # Handle extrapolation errors by setting to NaN
                cooling_rates[i, j] = np.nan
    
    # Save resampled tables to ASDF file
    print(f"\nSaving resampled tables to {output_file}")
    
    # Create the data tree for ASDF
    tree = {
        'grids': {
            'fast_log_rho': fast_log_rho,
            'fast_log_eint': fast_log_eint,
            'rho': rho_grid,
            'eint': eint_grid
        },
        'data': {
            'cooling_rates': cooling_rates,
            'temperatures': temperatures,
            'mean_molecular_weights': mean_molecular_weights
        },
        'metadata': {
            'n_rho': n_rho,
            'n_eint': n_eint,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'eint_min': eint_min,
            'eint_max': eint_max,
            'cloudy_H_mass_fraction': cloudy_H_mass_fraction,
            'description': 'Cooling rates resampled on (rho, e_int) grid using not-quite-logarithmic spacing',
            'spacing_method': 'not-quite-logarithmic (fast log10 approximation)',
            'units': {
                'rho': 'g/cm^3',
                'eint': 'erg/g',
                'cooling_rate': 'erg/cm^3/s',
                'temperature': 'K',
                'mmw': 'dimensionless (in units of m_H)'
            }
        }
    }
    
    # Write the ASDF file
    with asdf.AsdfFile(tree) as af:
        af.write_to(output_file)
    
    print("Done!")
    
    # Print some statistics
    valid_mask = ~np.isnan(cooling_rates)
    if np.any(valid_mask):
        print(f"\nStatistics:")
        print(f"Valid cooling rates: {np.sum(valid_mask)} / {cooling_rates.size}")
        print(f"Temperature range: {np.min(temperatures[valid_mask]):.2e} to {np.max(temperatures[valid_mask]):.2e} K")
        print(f"Cooling rate range: {np.min(cooling_rates[valid_mask]):.2e} to {np.max(cooling_rates[valid_mask]):.2e} erg/cm^3/s")


def main():
    """Main function to run the resampling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Resample Grackle cooling tables on (rho, e_int) grid',
        epilog='Use --find-eint-range to determine appropriate eint_min/eint_max from temperature bounds.')
    
    parser.add_argument('grackle_file', type=str,
                        help='Path to Grackle HDF5 cooling table file')
    
    # Mode selection
    parser.add_argument('--find-eint-range', action='store_true',
                        help='Find eint range from temperature bounds instead of resampling')
    
    # Parameters for resampling
    parser.add_argument('--n_rho', type=int, default=100,
                        help='Number of density points (default: 100)')
    parser.add_argument('--n_eint', type=int, default=100,
                        help='Number of specific energy points (default: 100)')
    parser.add_argument('--rho_min', type=float, default=1e-30,
                        help='Minimum density in g/cm^3 (default: 1e-30)')
    parser.add_argument('--rho_max', type=float, default=1e-20,
                        help='Maximum density in g/cm^3 (default: 1e-20)')
    parser.add_argument('--eint_min', type=float, default=1e10,
                        help='Minimum specific energy in erg/g (default: 1e10)')
    parser.add_argument('--eint_max', type=float, default=1e20,
                        help='Maximum specific energy in erg/g (default: 1e20)')
    parser.add_argument('--output', type=str, default='resampled_cooling_tables.asdf',
                        help='Output ASDF file name (default: resampled_cooling_tables.asdf)')
    
    # Parameters for finding eint range
    parser.add_argument('--T_min', type=float, default=10.0,
                        help='Minimum temperature in K for eint range finding (default: 10)')
    parser.add_argument('--T_max', type=float, default=1e9,
                        help='Maximum temperature in K for eint range finding (default: 1e9)')
    
    args = parser.parse_args()
    
    if args.find_eint_range:
        # Find eint range mode
        print(f"Finding specific internal energy range for:")
        print(f"  Density range: {args.rho_min:.2e} to {args.rho_max:.2e} g/cm^3")
        print(f"  Temperature range: {args.T_min:.2e} to {args.T_max:.2e} K\n")
        
        eint_min, eint_max = find_eint_range(
            args.grackle_file,
            args.rho_min, args.rho_max,
            args.T_min, args.T_max
        )
        
        print(f"\nSuggested parameters for resampling:")
        print(f"  --eint_min {eint_min:.6e}")
        print(f"  --eint_max {eint_max:.6e}")
    else:
        # Resampling mode
        resample_cooling_tables(
            args.grackle_file,
            n_rho=args.n_rho,
            n_eint=args.n_eint,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            eint_min=args.eint_min,
            eint_max=args.eint_max,
            output_file=args.output
        )


if __name__ == '__main__':
    main()
