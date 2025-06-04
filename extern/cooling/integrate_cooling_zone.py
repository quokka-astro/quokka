#!/usr/bin/env python3
# ABOUTME: Numerically integrate density and specific internal energy of a single zone
# ABOUTME: using Cloudy cooling tables resampled on (rho, e_int) grid

import numpy as np
import argparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import h5py

# Physical constants (cgs units)
boltzmann_constant_cgs = 1.380658e-16  # erg/K
m_H = 1.672623e-24  # g


def load_resampled_cooling_tables(filename):
    """Load the resampled cooling tables from HDF5 file.
    
    Args:
        filename: path to resampled cooling table HDF5 file
        
    Returns:
        dict containing grid data and cooling rates
    """
    with h5py.File(filename, 'r') as f:
        data = {
            'rho': f['grids/rho'][:],
            'eint': f['grids/eint'][:],
            'fast_log_rho': f['grids/fast_log_rho'][:],
            'fast_log_eint': f['grids/fast_log_eint'][:],
            'cooling_rates': f['data/cooling_rates'][:],
            'temperatures': f['data/temperatures'][:],
            'sound_speeds': f['data/sound_speeds'][:],
            'pressures': f['data/pressures'][:],
            'entropies': f['data/entropies'][:]
        }
        
        # Load metadata
        metadata = {}
        for key in f['metadata'].attrs.keys():
            metadata[key] = f['metadata'].attrs[key]
        data['metadata'] = metadata
        
    return data


def fast_log2(x):
    """Fast approximation of log2(x) using the not-quite-logarithmic method."""
    if np.any(x <= 0):
        raise ValueError("log divergent for x <= 0")
    x = np.asarray(x)    
    mantissa, exponent = np.frexp(x)
    return 2 * (mantissa - 1) + exponent


def interpolate_table(rho, eint, tables, table='cooling_rates'):
    """Interpolate cooling rate from resampled tables.
    
    Args:
        rho: density (g/cm^3)
        eint: specific internal energy (erg/g)
        tables: dict containing cooling table data
        
    Returns:
        cooling_rate: cooling rate Edot/rho^2 (erg/cm^3/s/(g/cm^3)^2)
    """
    # Convert to fast log space for interpolation
    fast_log_rho = fast_log2(rho)
    fast_log_eint = fast_log2(eint)
    
    # Find indices for interpolation
    # Use searchsorted to find the correct bin
    i_rho = np.searchsorted(tables['fast_log_rho'], fast_log_rho) - 1
    j_eint = np.searchsorted(tables['fast_log_eint'], fast_log_eint) - 1
    
    # Handle boundary cases
    n_rho = len(tables['fast_log_rho'])
    n_eint = len(tables['fast_log_eint'])
    
    if i_rho < 0:
        i_rho = 0
    if i_rho >= n_rho - 1:
        i_rho = n_rho - 2
        
    if j_eint < 0:
        j_eint = 0
    if j_eint >= n_eint - 1:
        j_eint = n_eint - 2
    
    # Get neighboring points
    rho1 = tables['fast_log_rho'][i_rho]
    rho2 = tables['fast_log_rho'][i_rho + 1]
    eint1 = tables['fast_log_eint'][j_eint]
    eint2 = tables['fast_log_eint'][j_eint + 1]
    
    # Get cooling rates at corners
    Q11 = tables[table][i_rho, j_eint]
    Q12 = tables[table][i_rho, j_eint + 1]
    Q21 = tables[table][i_rho + 1, j_eint]
    Q22 = tables[table][i_rho + 1, j_eint + 1]
    
    # Check for NaN values
    if np.isnan(Q11) or np.isnan(Q12) or np.isnan(Q21) or np.isnan(Q22):
        return 0.0  # No cooling if we're outside the valid range
    
    # Bilinear interpolation in fast log space
    t = (fast_log_rho - rho1) / (rho2 - rho1)
    u = (fast_log_eint - eint1) / (eint2 - eint1)
    
    cooling_rate = (1 - t) * (1 - u) * Q11 + \
                   t * (1 - u) * Q21 + \
                   (1 - t) * u * Q12 + \
                   t * u * Q22
    
    return cooling_rate


def cooling_ode_system(t, y, tables=None, rho=None):
    """ODE system for cooling evolution.
    
    The system evolves:
    - density (constant in time for single zone)
    - specific internal energy (decreases due to cooling)
    
    Args:
        t: time (s)
        y: state vector [rho, eint]
        tables: cooling table data
        
    Returns:
        dydt: time derivatives [deint/dt]
    """
    eint = y[0]
    if (eint > 0.):
        # Interpolate cooling rate
        cooling_rate = interpolate_table(rho, eint, tables)
    
        # deint/dt = cooling_rate * rho^2 / rho = cooling_rate * rho
        # Note: cooling_rate is already Edot/rho^2
        deint_dt = cooling_rate * rho
    else:
        return np.nan
    
    return [deint_dt]


def integrate_cooling_zone(rho0, T0, t_end, tables, n_output=100):
    """Integrate the cooling evolution of a single zone.
    
    Args:
        rho0: initial density (g/cm^3)
        T0: initial temperature (K)
        t_end: end time (s)
        tables: cooling table data
        n_output: number of output points
        
    Returns:
        dict with:
            - times: array of times (s)
            - rho: density array (constant)
            - eint: specific internal energy array
            - T: temperature array
    """
    # Convert initial temperature to specific internal energy
    # For ideal gas: e_int = (3/2) * k_B * T / (mu * m_H)
    # We need to get mu from the tables, but for initial estimate use mu ~ 0.6
    mu_init = 0.6  # rough estimate
    eint0 = (3.0 / 2.0) * boltzmann_constant_cgs * T0 / (mu_init * m_H)
    
    print(f"Initial conditions:")
    print(f"  rho0 = {rho0:.3e} g/cm^3")
    print(f"  T0 = {T0:.3e} K")
    print(f"  eint0 = {eint0:.3e} erg/g")
    
    # Initial state vector
    y0 = [eint0]
    
    # Time points for output
    t_span = (0, t_end)
    t_eval = np.logspace(np.log10(t_end/1e6), np.log10(t_end), n_output)
    t_eval = np.concatenate([[0], t_eval])
    
    # Solve ODE
    print(f"\nIntegrating from t=0 to t={t_end:.3e} s...")
    sol = solve_ivp(
        lambda t, y: cooling_ode_system(t, y, tables=tables, rho=rho0),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-10
    )
    
    if not sol.success:
        print(f"Warning: Integration failed with message: {sol.message}")
    
    # Extract results
    times = sol.t
    eint = sol.y[0, :]
    
    # Compute temperatures from specific internal energy
    T = np.zeros_like(eint)
    for i in range(len(times)):
        # Interpolate temperature from tables
        T[i] = interpolate_table(rho0, eint[i], tables, table='temperatures')
    
    return {
        'times': times,
        'rho': rho0,
        'eint': eint,
        'T': T
    }


def plot_cooling_evolution(results, output_file='cooling_evolution.png'):
    """Plot the cooling evolution results.
    
    Args:
        results: dict from integrate_cooling_zone
        output_file: output plot filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    
    times = results['times']
    rho = results['rho']
    eint = results['eint']
    T = results['T']
    
    # Convert time to useful units
    t_yr = times / (365.25 * 24 * 3600)  # years
    t_kyr = t_yr / 1000  # kiloyears
    t_Myr = t_yr / 1e6  # megayears
    
    # Choose appropriate time unit
    if np.max(t_yr) < 1:
        t_plot = times
        t_label = 'Time (s)'
    elif np.max(t_yr) < 1000:
        t_plot = t_yr
        t_label = 'Time (yr)'
    elif np.max(t_yr) < 1e6:
        t_plot = t_kyr
        t_label = 'Time (kyr)'
    else:
        t_plot = t_Myr
        t_label = 'Time (Myr)'
    
    # Plot specific internal energy
    ax = axes[0]
    ax.loglog(t_plot[1:], eint[1:])
    ax.set_xlabel(t_label)
    ax.set_ylabel(r'$e_{\rm int}$ (erg/g)')
    ax.set_title('Specific Internal Energy Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot temperature
    ax = axes[1]
    ax.loglog(t_plot[1:], T[1:])
    ax.set_xlabel(t_label)
    ax.set_ylabel('T (K)')
    ax.set_title('Temperature Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")


def main():
    """Main function to run the cooling integration."""
    parser = argparse.ArgumentParser(
        description='Integrate cooling evolution of a single zone using resampled Cloudy tables'
    )
    
    parser.add_argument('cooling_table', type=str,
                        help='Path to resampled cooling table HDF5 file')
    
    # Initial conditions
    parser.add_argument('--rho0', type=float, default=1e-24,
                        help='Initial density in g/cm^3 (default: 1e-24)')
    parser.add_argument('--T0', type=float, default=1e6,
                        help='Initial temperature in K (default: 1e6)')
    
    # Integration parameters
    parser.add_argument('--t_end', type=float, default=1e15,
                        help='End time in seconds (default: 1e15 ~ 30 Myr)')
    parser.add_argument('--n_output', type=int, default=200,
                        help='Number of output points (default: 200)')
    
    # Output options
    parser.add_argument('--plot', type=str, default='cooling_evolution.png',
                        help='Output plot filename (default: cooling_evolution.png)')
    
    args = parser.parse_args()
    
    # Load cooling tables
    print(f"Loading cooling tables from {args.cooling_table}...")
    tables = load_resampled_cooling_tables(args.cooling_table)
    
    # Check if initial conditions are within table bounds
    rho_min = tables['metadata']['rho_min']
    rho_max = tables['metadata']['rho_max']
    eint_min = tables['metadata']['eint_min']
    eint_max = tables['metadata']['eint_max']
    
    if args.rho0 < rho_min or args.rho0 > rho_max:
        print(f"Warning: Initial density {args.rho0:.3e} is outside table bounds [{rho_min:.3e}, {rho_max:.3e}]")
    
    # Run integration
    results = integrate_cooling_zone(
        args.rho0,
        args.T0,
        args.t_end,
        tables,
        n_output=args.n_output
    )
    
    # Print final state
    print(f"\nFinal state at t = {results['times'][-1]:.3e} s:")
    print(f"  rho = {args.rho0:.3e} g/cm^3")
    print(f"  eint = {results['eint'][-1]:.3e} erg/g")
    print(f"  T = {results['T'][-1]:.3e} K")
    
    # Plot results
    plot_cooling_evolution(results, args.plot)
    
if __name__ == '__main__':
    main()
