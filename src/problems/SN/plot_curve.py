import os
import numpy as np
import matplotlib.pyplot as plt

def main(f1, f2, outfile):
    # Read the CSV files using numpy
    table1 = np.genfromtxt(f1, delimiter=',', skip_header=1)
    table2 = np.genfromtxt(f2, delimiter=',', skip_header=1)
    title1 = os.path.basename(f1)
    title2 = os.path.basename(f2)

    # Extract columns (Time_yr is column 1, Max_Internal_Energy_erg is column 2)
    time1 = table1[:, 1]
    energy1 = table1[:, 2]
    time2 = table2[:, 1]
    energy2 = table2[:, 2]

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # First panel: Original comparison
    ax1.loglog(time1, energy1, 
            label=title1, linewidth=2, alpha=0.8)
    ax1.loglog(time2, energy2, 
            label=title2, linewidth=2, alpha=0.8, linestyle='--')

    ax1.set_ylabel('Max Internal Energy (erg)', fontsize=12)
    ax1.set_title('Supernova Energy History Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Second panel: Relative error
    # Interpolate resampled data to grackle time points for comparison
    resampled_interp = np.interp(time1, time2, energy2)

    # Check for out-of-bounds and set to np.nan
    resampled_min_time = np.min(time2)
    resampled_max_time = np.max(time2)
    out_of_bounds = (time1 < resampled_min_time) | (time1 > resampled_max_time)
    resampled_interp[out_of_bounds] = np.nan

    # Calculate relative error: (resampled - grackle) / grackle
    relative_error = (resampled_interp - energy1) / energy1

    ax2.semilogx(time1, relative_error * 100, 
                linewidth=2, color='red', alpha=0.8)
    ax2.set_xlabel('Time (yr)', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Relative Difference', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    f1 = 'tests/sn_energy_history_grackle.csv'
    f2 = 'tests/sn_energy_history_resampled.csv'
    f3 = 'tests/sn_energy_history_datatable.csv'

    main(f1, f2, 'sn_energy_grackle_vs_resampled.png')
    main(f1, f3, 'sn_energy_grackle_vs_datatable.png')
