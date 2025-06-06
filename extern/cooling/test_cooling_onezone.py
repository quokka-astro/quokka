#!/usr/bin/env python3
# ABOUTME: Example script showing how to integrate cooling evolution
# ABOUTME: for different initial conditions using the resampled cooling tables

import numpy as np
from integrate_cooling_zone import (
    load_resampled_cooling_tables,
    integrate_cooling_zone,
    plot_cooling_comparison
)
from grackle_tables import read_tables

# Path to the resampled cooling table
cooling_table = "./CloudyData_UVB=HM2012_resampled.h5"
grackle_cooling_table = "../grackle_data_files/input/CloudyData_UVB=HM2012.h5"

# Define test cases with different initial conditions
test_cases = [
    # (name, rho0, T0, t_end)
    ("hot_diffuse", 1e-26, 1e7, 1e17),      # Hot diffuse gas
    ("warm_medium", 1e-24, 1e6, 1e15),      # Warm medium 
    ("cool_dense", 1e-22, 1e4, 1e14),       # Cool dense gas
    ("shocked_gas", 1e-23, 5e6, 5e15),      # Post-shock gas
]

print("Running cooling integration examples...")
print(f"Using cooling table: {cooling_table}")
print("")

# Load cooling tables once
print("Loading cooling tables...")
tables = load_resampled_cooling_tables(cooling_table)
print(f"  Density range: {tables['metadata']['rho_min']:.2e} to {tables['metadata']['rho_max']:.2e} g/cm^3")
print(f"  Energy range: {tables['metadata']['eint_min']:.2e} to {tables['metadata']['eint_max']:.2e} erg/g")
print("")

# Load cooling tables once
print("Loading Grackle cooling tables...")
grackle_tables = read_tables(grackle_cooling_table)
print("")

for name, rho0, T0, t_end in test_cases:
    print(f"Test case: {name}")
    print(f"  Initial density: {rho0:.2e} g/cm^3")
    print(f"  Initial temperature: {T0:.2e} K") 
    print(f"  Integration time: {t_end:.2e} s ({t_end/(365.25*24*3600*1e6):.1f} Myr)")
    
    # Run the integration
    my_results = integrate_cooling_zone(
        rho0, T0, t_end, tables, grackle_tables, n_output=100
    )

    for results, runname in zip(my_results, ['resampled', 'Grackle']):
        # Print final state
        print(f"\n[{runname}] Final state at t = {results['times'][-1]:.3e} s:")
        print(f"  rho = {rho0:.3e} g/cm^3")
        print(f"  eint = {results['eint'][-1]:.3e} erg/g")
        print(f"  T = {results['T'][-1]:.3e} K")
    
    # Save comparison plot
    comparison_plot = f"cooling_comparison_{name}.png"
    plot_cooling_comparison(my_results[0], my_results[1], 
                          labels=('Resampled', 'Grackle'), 
                          output_file=comparison_plot)
    
print("\nAll test cases completed!")
