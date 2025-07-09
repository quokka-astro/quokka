#!/usr/bin/env python3
"""
Script to plot slices of the Gaussian random vector field from FewModesFT test.
Uses yt to read AMReX plotfiles and matplotlib for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt
import argparse
import os

def plot_random_field(plotfile_path, output_dir="./"):
    """
    Plot slices of the random vector field from the plotfile.
    
    Parameters:
    -----------
    plotfile_path : str
        Path to the plotfile directory
    output_dir : str
        Directory to save the output plots
    """
    
    # Load the dataset
    print(f"Loading dataset from {plotfile_path}")
    ds = yt.load(plotfile_path)
    
    # Print basic dataset info
    print(f"Dataset info:")
    print(f"  Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")
    print(f"  Resolution: {ds.domain_dimensions}")
    print(f"  Fields: {ds.field_list}")
    
    # Create a figure with subplots for each velocity component
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Gaussian Random Vector Field (FewModesFT)', fontsize=16)
    
    # Field names (yt uses tuples for AMReX fields)
    field_names = [('boxlib', 'vx'), ('boxlib', 'vy'), ('boxlib', 'vz')]
    field_labels = [r'$v_x$', r'$v_y$', r'$v_z$']
    
    # Plot slices through the center of the domain
    slice_coord = 0.5  # middle of the domain
    
    for i, (field, label) in enumerate(zip(field_names, field_labels)):
        # Create z-slice (xy plane)
        slc = yt.SlicePlot(ds, 'z', field, center=[0.5, 0.5, slice_coord])
        slc.set_cmap(field, 'RdBu_r')
        slc.set_zlim(field, -3, 3)  # Set reasonable limits
        
        # Get the slice data
        slice_data = slc.frb.data[field]
        extent = [ds.domain_left_edge[0], ds.domain_right_edge[0],
                  ds.domain_left_edge[1], ds.domain_right_edge[1]]
        
        # Plot the slice
        im = axes[0, i].imshow(slice_data, extent=extent, cmap='RdBu_r', 
                              vmin=-3, vmax=3, origin='lower')
        axes[0, i].set_title(f'{label} (z-slice at z=0.5)')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, i], label=label)
        
        # Plot histogram of field values
        field_values = ds.all_data()[field].v.flatten()
        axes[1, i].hist(field_values, bins=50, alpha=0.7, density=True)
        axes[1, i].set_xlabel(f'{label} values')
        axes[1, i].set_ylabel('Probability density')
        axes[1, i].set_title(f'{label} histogram')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add statistics to the histogram
        mean_val = np.mean(field_values)
        std_val = np.std(field_values)
        axes[1, i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
        axes[1, i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                          label=f'±σ: {std_val:.3f}')
        axes[1, i].axvline(mean_val - std_val, color='orange', linestyle='--')
        axes[1, i].legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'random_field_slices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Create a vector field plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create a slice and get the data
    slice_z = ds.slice('z', slice_coord)
    
    # Get coordinate arrays
    x_coords = slice_z['x'].v
    y_coords = slice_z['y'].v
    vx_vals = slice_z[('boxlib', 'vx')].v
    vy_vals = slice_z[('boxlib', 'vy')].v
    
    # Reshape for plotting (assuming uniform grid)
    nx, ny = ds.domain_dimensions[0], ds.domain_dimensions[1]
    x_2d = x_coords.reshape(nx, ny)
    y_2d = y_coords.reshape(nx, ny)
    vx_2d = vx_vals.reshape(nx, ny)
    vy_2d = vy_vals.reshape(nx, ny)
    
    # Subsample for vector plot (every nth point)
    skip = max(1, nx // 20)  # Show ~20 vectors per dimension
    
    # Create vector field plot
    Q = ax.quiver(x_2d[::skip, ::skip], y_2d[::skip, ::skip], 
                  vx_2d[::skip, ::skip], vy_2d[::skip, ::skip], 
                  np.sqrt(vx_2d[::skip, ::skip]**2 + vy_2d[::skip, ::skip]**2),
                  cmap='viridis', alpha=0.8)
    
    ax.set_title('Random Vector Field (z-slice at z=0.5)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    # Add colorbar for vector magnitude
    cbar = plt.colorbar(Q, ax=ax, label='|v|')
    
    # Save vector plot
    vector_output = os.path.join(output_dir, 'random_field_vectors.png')
    plt.savefig(vector_output, dpi=300, bbox_inches='tight')
    print(f"Vector plot saved as {vector_output}")
    
    # Print some statistics
    print("\nField Statistics:")
    for field, label in zip(field_names, field_labels):
        field_data = ds.all_data()[field].v
        print(f"  {label}: mean = {np.mean(field_data):.4f}, "
              f"std = {np.std(field_data):.4f}, "
              f"min = {np.min(field_data):.4f}, "
              f"max = {np.max(field_data):.4f}")
    
    # Calculate and print divergence statistics using yt's API
    print("\nDivergence Analysis:")
    try:
        # Force periodicity so yt can handle boundary conditions properly
        ds.force_periodicity()
        
        # Use yt's built-in divergence calculation
        # First, add a velocity field that yt can use for divergence calculation
        def _velocity_x(field, data):
            return data[('boxlib', 'vx')]
        
        def _velocity_y(field, data):
            return data[('boxlib', 'vy')]
        
        def _velocity_z(field, data):
            return data[('boxlib', 'vz')]
        
        # Add velocity fields in the format yt expects
        ds.add_field(('gas', 'velocity_x'), function=_velocity_x, units='dimensionless', sampling_type='cell')
        ds.add_field(('gas', 'velocity_y'), function=_velocity_y, units='dimensionless', sampling_type='cell')
        ds.add_field(('gas', 'velocity_z'), function=_velocity_z, units='dimensionless', sampling_type='cell')
        
        # Use yt's add_gradient_fields to automatically calculate divergence
        # This properly handles AMR grids and boundary conditions
        ds.add_gradient_fields(('gas', 'velocity_x'))
        ds.add_gradient_fields(('gas', 'velocity_y'))
        ds.add_gradient_fields(('gas', 'velocity_z'))
        
        # Calculate divergence using yt's gradient fields
        def _velocity_divergence(field, data):
            return (data[('gas', 'velocity_x_gradient_x')] + 
                    data[('gas', 'velocity_y_gradient_y')] + 
                    data[('gas', 'velocity_z_gradient_z')])
        
        ds.add_field(('gas', 'velocity_divergence'), function=_velocity_divergence, 
                    units='1/code_length', sampling_type='cell')
        
        # Get the divergence data
        ad = ds.all_data()
        divergence_data = ad[('gas', 'velocity_divergence')]
        
        print(f"  Divergence (yt): mean = {np.mean(divergence_data):.6f}, "
              f"std = {np.std(divergence_data):.6f}, "
              f"max = {np.max(np.abs(divergence_data)):.6f}")
        
        # Create a slice plot of the divergence
        slc = yt.SlicePlot(ds, 'z', ('gas', 'velocity_divergence'), center=[0.5, 0.5, slice_coord])
        slc.set_cmap(('gas', 'velocity_divergence'), 'RdBu_r')
        
        # Get the slice data for plotting
        slice_data = slc.frb.data[('gas', 'velocity_divergence')]
        
        print(f"  Divergence (slice): mean = {np.mean(slice_data):.6f}, "
              f"std = {np.std(slice_data):.6f}, "
              f"max = {np.max(np.abs(slice_data)):.6f}")
        
        # Plot divergence using yt's calculated values
        fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        extent = [ds.domain_left_edge[0], ds.domain_right_edge[0],
                  ds.domain_left_edge[1], ds.domain_right_edge[1]]
        
        # Plot with symmetric colorbar
        vmax = np.max(np.abs(slice_data))
        im = ax.imshow(slice_data, extent=extent, cmap='RdBu_r', 
                      vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title('Divergence of Random Vector Field (yt calculation)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label=r'$\nabla \cdot \mathbf{v}$ [1/code_length]')
        
        div_output = os.path.join(output_dir, 'random_field_divergence.png')
        plt.savefig(div_output, dpi=300, bbox_inches='tight')
        print(f"Divergence plot saved as {div_output}")
        
        # For comparison, also show manual calculation vs yt calculation
        print("\nComparison with manual finite difference calculation:")
        
        # Manual calculation for comparison
        ad_manual = ds.all_data()
        vx_data = ad_manual[('boxlib', 'vx')].v
        vy_data = ad_manual[('boxlib', 'vy')].v
        vz_data = ad_manual[('boxlib', 'vz')].v
        
        # Simple numpy gradient (non-periodic)
        nx, ny, nz = ds.domain_dimensions
        vx_3d = vx_data.reshape(nx, ny, nz)
        vy_3d = vy_data.reshape(nx, ny, nz)
        vz_3d = vz_data.reshape(nx, ny, nz)
        
        dx = float(ds.domain_width[0] / ds.domain_dimensions[0])
        dy = float(ds.domain_width[1] / ds.domain_dimensions[1])
        dz = float(ds.domain_width[2] / ds.domain_dimensions[2])
        
        dvx_dx_manual = np.gradient(vx_3d, dx, axis=0)
        dvy_dy_manual = np.gradient(vy_3d, dy, axis=1)
        dvz_dz_manual = np.gradient(vz_3d, dz, axis=2)
        
        divergence_manual = dvx_dx_manual + dvy_dy_manual + dvz_dz_manual
        
        print(f"  Divergence (manual): mean = {np.mean(divergence_manual):.6f}, "
              f"std = {np.std(divergence_manual):.6f}, "
              f"max = {np.max(np.abs(divergence_manual)):.6f}")
        
        # Check correlation between yt and manual calculation
        slice_idx = int(slice_coord * nz)
        divergence_manual_slice = divergence_manual[:, :, slice_idx]
        
        print(f"\nMethod comparison:")
        # Convert yt slice data to numpy array for correlation
        slice_data_np = np.array(slice_data)
        if slice_data_np.shape == divergence_manual_slice.shape:
            print(f"  yt vs manual correlation: {np.corrcoef(slice_data_np.flatten(), divergence_manual_slice.flatten())[0,1]:.4f}")
        else:
            print(f"  yt slice shape: {slice_data_np.shape}, manual slice shape: {divergence_manual_slice.shape}")
            print(f"  Cannot calculate correlation due to shape mismatch")
        
    except Exception as e:
        print(f"  Could not calculate divergence: {e}")
        import traceback
        traceback.print_exc()
    
    # plt.show()  # Comment out to avoid unit errors

def main():
    parser = argparse.ArgumentParser(description='Plot Gaussian random vector field from FewModesFT test')
    parser.add_argument('plotfile', nargs='?', default='plt_few_modes_ft', 
                       help='Path to plotfile directory (default: plt_few_modes_ft)')
    parser.add_argument('-o', '--output', default='./', 
                       help='Output directory for plots (default: current directory)')
    
    args = parser.parse_args()
    
    # Check if plotfile exists
    if not os.path.exists(args.plotfile):
        print(f"Error: Plotfile {args.plotfile} does not exist.")
        print("Make sure to run the test_few_modes_ft executable first.")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Plot the random field
    plot_random_field(args.plotfile, args.output)
    
    return 0

if __name__ == '__main__':
    exit(main())