#!/usr/bin/env python3
# ABOUTME: Numerically integrate density and specific internal energy of a single zone
# ABOUTME: using Cloudy cooling tables resampled on (rho, e_int) grid

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import h5py

from grackle_tables import (
    read_tables, cooling_rate, interpolate_mu, compute_temperature_from_nH_e,
    m_H, boltzmann_constant_cgs_, cloudy_H_mass_fraction
)


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


def cooling_ode_system_resampled(t, y, tables=None, rho=None):
    """ODE system for cooling evolution.    
    The system evolves:
    - specific internal energy (decreases due to cooling)
    
    Args:
        t: time (s)
        y: state vector [eint]
        tables: cooling table data
        
    Returns:
        dydt: time derivative
    """
    eint = y[0]
    if (eint > 0.):
        interp_value = interpolate_table(rho, eint, tables)
        Edot = interp_value * rho**2
        # deint/dt = interp_value * rho^2 / rho = interp_value * rho
        deint_dt = Edot / rho
    else:
        return np.nan
    
    return [deint_dt]

def cooling_ode_system_original(t, y, tables=None, rho=None):
    """ODE system for cooling evolution.    
    The system evolves:
    - specific internal energy (decreases due to cooling)
    
    Args:
        t: time (s)
        y: state vector [eint]
        tables: cooling table data
        
    Returns:
        dydt: time derivative
    """
    eint = y[0]
    if (eint > 0.):
        # Interpolate cooling rate
        nH = cloudy_H_mass_fraction * (rho / m_H)
        T = compute_temperature_from_nH_e(nH, eint, tables=tables)
        Edot = cooling_rate(nH, T, zmet=1.0, redshift=0., tables=tables)
        deint_dt = Edot / rho # convert to specific rate
    else:
        return np.nan
    
    return [deint_dt]


def integrate_cooling_zone(rho0, T0, t_end, resampled_tables, grackle_tables, n_output=100):
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
    eint0 = (3.0 / 2.0) * boltzmann_constant_cgs_ * T0 / (mu_init * m_H)
    
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
    sol_new = solve_ivp(
        lambda t, y: cooling_ode_system_resampled(t, y, tables=resampled_tables, rho=rho0),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-10
    )
    
    if not sol_new.success:
        print(f"Warning: Integration failed with message: {sol_new.message}")

    # Solve ODE (using original method)
    print(f"\nIntegrating from t=0 to t={t_end:.3e} s...")
    sol_orig = solve_ivp(
        lambda t, y: cooling_ode_system_original(t, y, tables=grackle_tables, rho=rho0),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-10
    )
    
    if not sol_orig.success:
        print(f"Warning: Integration failed with message: {sol_orig.message}")

    results = []
    for sol in [sol_new, sol_orig]:
        # Extract results
        times = sol.t
        eint = sol.y[0, :]
        
        # Compute temperatures from specific internal energy
        T = np.zeros_like(eint)
        for i in range(len(times)):
            # Interpolate temperature from tables
            T[i] = interpolate_table(rho0, eint[i], resampled_tables, table='temperatures')
    
        results.append( {
            'times': times,
            'rho': rho0,
            'eint': eint,
            'T': T
        })
        
    return results



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


def plot_cooling_comparison(results1, results2, labels=('Method 1', 'Method 2'), 
                          output_file='cooling_comparison.png'):
    """Plot two different cooling evolution results on the same plot.
    
    Args:
        results1: dict from integrate_cooling_zone (first method)
        results2: dict from integrate_cooling_zone (second method)
        labels: tuple of labels for the two methods
        output_file: output plot filename
    """
    # Create figure with GridSpec for custom layout
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.3)
    
    # Create axes - properly using 2x2 grid
    ax_eint = fig.add_subplot(gs[0, 0])
    ax_eint_diff = fig.add_subplot(gs[1, 0], sharex=ax_eint)
    ax_T = fig.add_subplot(gs[0, 1])
    ax_T_diff = fig.add_subplot(gs[1, 1], sharex=ax_T)
    
    # Process both results and store for relative difference calculation
    times1 = results1['times']
    t_yr1 = times1 / (365.25 * 24 * 3600)  # years
    t_kyr1 = t_yr1 / 1000  # kiloyears
    t_Myr1 = t_yr1 / 1e6  # megayears
    
    # Choose appropriate time unit based on results1
    if np.max(t_yr1) < 1:
        t_plot1 = times1
        t_label = 'Time (s)'
        t_factor = 1.0
    elif np.max(t_yr1) < 1000:
        t_plot1 = t_yr1
        t_label = 'Time (yr)'
        t_factor = 1.0 / (365.25 * 24 * 3600)
    elif np.max(t_yr1) < 1e6:
        t_plot1 = t_kyr1
        t_label = 'Time (kyr)'
        t_factor = 1.0 / (365.25 * 24 * 3600 * 1000)
    else:
        t_plot1 = t_Myr1
        t_label = 'Time (Myr)'
        t_factor = 1.0 / (365.25 * 24 * 3600 * 1e6)
    
    # Get data for both methods
    eint1 = results1['eint']
    T1 = results1['T']
    
    times2 = results2['times']
    t_plot2 = times2 * t_factor
    eint2 = results2['eint']
    T2 = results2['T']
    
    # Plot specific internal energy
    ax_eint.loglog(t_plot1[1:], eint1[1:], label=labels[0], color='blue', linewidth=2)
    ax_eint.loglog(t_plot2[1:], eint2[1:], label=labels[1], color='red', linewidth=2, linestyle='--')
    ax_eint.set_ylabel(r'$e_{\rm int}$ (erg/g)')
    ax_eint.set_title(f'Specific Internal Energy Evolution\n($\\rho$ = {results1["rho"]:.2e} g/cm³)')
    ax_eint.grid(True, alpha=0.3)
    ax_eint.legend()
    ax_eint.set_xticklabels([])  # Hide x labels for upper panel
    
    # Interpolate results2 onto results1 time grid for difference calculation
    eint2_interp = np.interp(times1, times2, eint2)
    T2_interp = np.interp(times1, times2, T2)
    
    # Calculate relative differences
    eint_rel_diff = (eint2_interp - eint1) / eint1
    T_rel_diff = (T2_interp - T1) / T1
    
    # Plot relative difference for specific internal energy
    ax_eint_diff.semilogx(t_plot1[1:], eint_rel_diff[1:], color='black', linewidth=1)
    ax_eint_diff.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_eint_diff.set_xlabel(t_label)
    ax_eint_diff.set_ylabel('Rel. Diff.')
    ax_eint_diff.grid(True, alpha=0.3)
    ax_eint_diff.set_ylim(-0.025, 0.025)  # ±2.5% range
    
    # Plot temperature
    ax_T.loglog(t_plot1[1:], T1[1:], label=labels[0], color='blue', linewidth=2)
    ax_T.loglog(t_plot2[1:], T2[1:], label=labels[1], color='red', linewidth=2, linestyle='--')
    ax_T.set_ylabel('T (K)')
    ax_T.set_title(f'Temperature Evolution\n($\\rho$ = {results1["rho"]:.2e} g/cm³)')
    ax_T.grid(True, alpha=0.3)
    ax_T.legend()
    ax_T.set_xticklabels([])  # Hide x labels for upper panel
    
    # Plot relative difference for temperature
    ax_T_diff.semilogx(t_plot1[1:], T_rel_diff[1:], color='black', linewidth=1)
    ax_T_diff.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_T_diff.set_xlabel(t_label)
    ax_T_diff.set_ylabel('Rel. Diff.')
    ax_T_diff.grid(True, alpha=0.3)
    ax_T_diff.set_ylim(-0.025, 0.025)  # ±2.5% range
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {output_file}")
