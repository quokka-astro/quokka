import numpy as np
import matplotlib.pyplot as plt

def main(f1, f2):
    # Read the CSV files using numpy
    grackle_data = np.genfromtxt(f1, delimiter=',', skip_header=1)
    resampled_data = np.genfromtxt(f2, delimiter=',', skip_header=1)

    # Extract columns (Time_yr is column 1, Max_Internal_Energy_erg is column 2)
    grackle_time = grackle_data[:, 1]
    grackle_energy = grackle_data[:, 2]
    resampled_time = resampled_data[:, 1]
    resampled_energy = resampled_data[:, 2]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # First panel: Original comparison
    ax1.loglog(grackle_time, grackle_energy, 
            label='Grackle', linewidth=2, alpha=0.8)
    ax1.loglog(resampled_time, resampled_energy, 
            label='Resampled', linewidth=2, alpha=0.8, linestyle='--')

    ax1.set_ylabel('Max Internal Energy (erg)', fontsize=12)
    ax1.set_title('Supernova Energy History Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Second panel: Relative error
    # Interpolate resampled data to grackle time points for comparison
    resampled_interp = np.interp(grackle_time, resampled_time, resampled_energy)

    # Check for out-of-bounds and set to np.nan
    resampled_min_time = np.min(resampled_time)
    resampled_max_time = np.max(resampled_time)
    out_of_bounds = (grackle_time < resampled_min_time) | (grackle_time > resampled_max_time)
    resampled_interp[out_of_bounds] = np.nan

    # Calculate relative error: (resampled - grackle) / grackle
    relative_error = (resampled_interp - grackle_energy) / grackle_energy

    ax2.semilogx(grackle_time, relative_error * 100, 
                linewidth=2, color='red', alpha=0.8)
    ax2.set_xlabel('Time (yr)', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Relative Error (Resampled - Grackle) / Grackle', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig('sn_energy_comparison.png', dpi=300, bbox_inches='tight')

    # Optionally save the plot
    # plt.savefig('sn_energy_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    f1 = 'tests/sn_energy_history_grackle.csv'
    f2 = 'tests/sn_energy_history_resampled.csv'
    # f2 = 'tests/sn_energy_history_datatable.csv'
    main(f1, f2)
