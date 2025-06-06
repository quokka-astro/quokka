#!/usr/bin/env python3
# ABOUTME: Test script for Cloudy table cooling comparing resampled vs original tables
# ABOUTME: Integrates cooling evolution for different initial conditions using Cloudy cooling data

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import h5py

from integrate_cooling_zone import (
    load_resampled_cooling_tables,
    interpolate_table,
    fast_log2,
    plot_cooling_comparison
)
from cloudy_cooling_tools_tables import (
    read_tables as read_cloudy_tables,
    cooling_rate as cloudy_cooling_rate,
    interpolate_mu,
    m_H,
    boltzmann_constant_cgs_,
    cloudy_H_mass_fraction
)
from grackle_tables import specific_energy_from_temperature

def compute_temperature_from_nH_e_cloudy(nH, eint, tables=None, gamma=5./3.):
    """Compute temperature from number density and specific internal energy using Cloudy tables.
    
    This is based on the same approach as compute_temperature_from_nH_e in grackle_tables.py,
    using the TOMS748 root finder to solve for temperature given internal energy.
    
    Args:
        nH: hydrogen number density (cm^-3)
        eint: specific internal energy (erg/g)
        tables: Cloudy table data
        gamma: adiabatic index (default 5/3)
        
    Returns:
        T: temperature (K)
    """
    # Set temperature bounds from table range
    T_min = 10.0 ** tables.log_T[0]
    T_max = 10.0 ** tables.log_T[-1]
    mu_min = np.min(tables.mmw)
    mu_max = np.max(tables.mmw)
    
    # Check whether temperature is (obviously) out-of-bounds
    eint_min = specific_energy_from_temperature(T_min, mu_max)
    eint_max = specific_energy_from_temperature(T_max, mu_min)
    if eint <= eint_min:
        return T_min
    if eint >= eint_max:
        return T_max
    
    # Solve for temperature given Eint (with fixed adiabatic index gamma)
    C = (gamma - 1.) * eint * m_H / boltzmann_constant_cgs_
    
    # Define the function to find the root of: f(T) = C * mu(T) - T = 0
    def f(T):
        # Compute new mu from mu(T) table
        T_clamped = np.clip(T, T_min, T_max)
        mu = interpolate_mu(nH, T_clamped, tables=tables)
        return C * mu - T
    
    # Compute temperature bounds using physics
    T_lower = max(C * mu_min, T_min)
    T_upper = min(C * mu_max, T_max)
    
    # Use scipy's TOMS748 method for root finding (same as C++ code)
    from scipy.optimize import toms748    
    try:
        # TOMS748 method with relative tolerance
        T_sol = toms748(f, T_lower, T_upper, rtol=1.0e-5, maxiter=100)
    except ValueError as e:
        # Root finding failed
        print(f"Tgas iteration failed! eint = {eint:.17e}, nH = {nH:.3e}, T_lower = {T_lower:.3e}, T_upper = {T_upper:.3e}")
        raise e
    
    return T_sol


def cooling_ode_system_cloudy(t, y, tables=None, rho=None):
    """ODE system for cooling evolution using original Cloudy tables.
    
    Args:
        t: time (s)
        y: state vector [eint]
        tables: Cloudy cooling table data
        rho: gas density (g/cm^3)
        
    Returns:
        dydt: time derivative [deint_dt]
    """
    eint = y[0]
    if eint > 0.:
        # Convert density to hydrogen number density
        nH = cloudy_H_mass_fraction * (rho / m_H)
        
        # Compute temperature from specific internal energy
        T = compute_temperature_from_nH_e_cloudy(nH, eint, tables=tables)
        
        # Get cooling rate
        Edot = cloudy_cooling_rate(nH, T, redshift=0., tables=tables)
        
        # Convert to specific rate
        deint_dt = Edot / rho
    else:
        return np.nan
    
    return [deint_dt]


def cooling_ode_system_resampled_cloudy(t, y, tables=None, rho=None):
    """ODE system for cooling evolution using resampled Cloudy tables.
    
    Args:
        t: time (s)
        y: state vector [eint]
        tables: resampled cooling table data
        rho: gas density (g/cm^3)
        
    Returns:
        dydt: time derivative [deint_dt]
    """
    eint = y[0]
    if eint > 0.:
        # Interpolate cooling rate from resampled tables
        interp_value = interpolate_table(rho, eint, tables)
        Edot = interp_value * rho**2
        
        # Convert to specific rate
        deint_dt = Edot / rho
    else:
        return np.nan
    
    return [deint_dt]


def integrate_cloudy_cooling_zone(rho0, T0, t_end, resampled_tables, cloudy_tables, n_output=100):
    """Integrate the cooling evolution of a single zone using both Cloudy table methods.
    
    Args:
        rho0: initial density (g/cm^3)
        T0: initial temperature (K)
        t_end: end time (s)
        resampled_tables: resampled cooling table data
        cloudy_tables: original Cloudy table data
        n_output: number of output points
        
    Returns:
        list of dicts with:
            - times: array of times (s)
            - rho: density (constant)
            - eint: specific internal energy array
            - T: temperature array
    """
    # Convert initial temperature to specific internal energy
    nH_init = cloudy_H_mass_fraction * (rho0 / m_H)
    mu_init = interpolate_mu(nH_init, T0, tables=cloudy_tables)
    eint0 = (3.0 / 2.0) * boltzmann_constant_cgs_ * T0 / (mu_init * m_H)
    
    print(f"Initial conditions:")
    print(f"  rho0 = {rho0:.3e} g/cm^3")
    print(f"  nH0 = {nH_init:.3e} cm^-3")
    print(f"  T0 = {T0:.3e} K")
    print(f"  mu0 = {mu_init:.3f}")
    print(f"  eint0 = {eint0:.3e} erg/g")
    
    # Initial state vector
    y0 = [eint0]
    
    # Time points for output
    t_span = (0, t_end)
    t_eval = np.logspace(np.log10(t_end/1e6), np.log10(t_end), n_output)
    t_eval = np.concatenate([[0], t_eval])
    
    results = []
    
    # Solve using resampled tables
    print(f"\nIntegrating with resampled tables from t=0 to t={t_end:.3e} s...")
    sol_resampled = solve_ivp(
        lambda t, y: cooling_ode_system_resampled_cloudy(t, y, tables=resampled_tables, rho=rho0),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-10
    )
    
    if not sol_resampled.success:
        print(f"Warning: Resampled integration failed with message: {sol_resampled.message}")
    
    # Solve using original Cloudy tables
    print(f"Integrating with original Cloudy tables from t=0 to t={t_end:.3e} s...")
    sol_cloudy = solve_ivp(
        lambda t, y: cooling_ode_system_cloudy(t, y, tables=cloudy_tables, rho=rho0),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-10
    )
    
    if not sol_cloudy.success:
        print(f"Warning: Cloudy integration failed with message: {sol_cloudy.message}")
    
    # Process results for both methods
    for sol, method_name in [(sol_resampled, 'resampled'), (sol_cloudy, 'cloudy')]:
        times = sol.t
        eint = sol.y[0, :]
        
        # Compute temperatures
        T = np.zeros_like(eint)
        for i in range(len(times)):
            if method_name == 'resampled':
                # Use temperature from resampled tables
                T[i] = interpolate_table(rho0, eint[i], resampled_tables, table='temperatures')
            else:
                # Compute temperature using original method
                nH = cloudy_H_mass_fraction * (rho0 / m_H)
                T[i] = compute_temperature_from_nH_e_cloudy(nH, eint[i], tables=cloudy_tables)
        
        results.append({
            'times': times,
            'rho': rho0,
            'eint': eint,
            'T': T,
            'method': method_name
        })
    
    return results


def main():
    """Main test function for Cloudy cooling tables."""
    
    # Path to the cooling tables
    resampled_table = "./isrf_1000Go_grains_resampled.h5"
    cloudy_table = "./isrf_1000Go_grains.h5"
    
    # Define test cases with different initial conditions
    test_cases = [
        # (name, rho0, T0, t_end)
        ("hot_diffuse", 1e-26, 1e7, 1e17),      # Hot diffuse gas
        ("warm_medium", 1e-24, 1e6, 1e15),      # Warm medium 
        ("cool_dense", 1e-22, 1e4, 1e14),       # Cool dense gas
        ("shocked_gas", 1e-23, 5e6, 5e15),      # Post-shock gas
    ]
    
    print("Running Cloudy cooling integration tests...")
    print(f"Using resampled table: {resampled_table}")
    print(f"Using original Cloudy table: {cloudy_table}")
    print("")
    
    # Load cooling tables
    print("Loading resampled cooling tables...")
    resampled_tables = load_resampled_cooling_tables(resampled_table)
    print(f"  Density range: {resampled_tables['metadata']['rho_min']:.2e} to {resampled_tables['metadata']['rho_max']:.2e} g/cm^3")
    print(f"  Energy range: {resampled_tables['metadata']['eint_min']:.2e} to {resampled_tables['metadata']['eint_max']:.2e} erg/g")
    print("")
    
    print("Loading original Cloudy cooling tables...")
    cloudy_tables = read_cloudy_tables(cloudy_table, apply_unit_conversion=True)
    print(f"  nH range: 10^{cloudy_tables.log_nH[0]:.1f} to 10^{cloudy_tables.log_nH[-1]:.1f} cm^-3")
    print(f"  T range: 10^{cloudy_tables.log_T[0]:.1f} to 10^{cloudy_tables.log_T[-1]:.1f} K")
    print("")
    
    # Run test cases
    for name, rho0, T0, t_end in test_cases:
        print(f"Test case: {name}")
        print(f"  Initial density: {rho0:.2e} g/cm^3")
        print(f"  Initial temperature: {T0:.2e} K") 
        print(f"  Integration time: {t_end:.2e} s ({t_end/(365.25*24*3600*1e6):.1f} Myr)")
        
        # Run the integration
        results = integrate_cloudy_cooling_zone(
            rho0, T0, t_end, resampled_tables, cloudy_tables, n_output=100
        )
        
        # Print final states
        for result in results:
            method = result['method']
            print(f"\n[{method}] Final state at t = {result['times'][-1]:.3e} s:")
            print(f"  rho = {rho0:.3e} g/cm^3")
            print(f"  eint = {result['eint'][-1]:.3e} erg/g")
            print(f"  T = {result['T'][-1]:.3e} K")
        
        # Save comparison plot
        comparison_plot = f"cloudy_cooling_comparison_{name}.png"
        plot_cooling_comparison(results[0], results[1], 
                              labels=('Resampled Cloudy', 'Original Cloudy'), 
                              output_file=comparison_plot)
        
        print(f"\n{'='*60}\n")
    
    print("All Cloudy cooling test cases completed!")


if __name__ == "__main__":
    main()
