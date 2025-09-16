#!/usr/bin/env python3
"""
Velocity Field Computation for Turbulent Box Simulation

This script computes a velocity field defined by perturbations made up of several
Fourier modes for hydrodynamic simulation of a turbulent box.

The velocity field is defined by:
v = sum_{i,j,k} A_{ijk} * c_s * exp(i * k_{ijk} · x)

Where:
- k_{ijk} = (k_x, k_y, k_z) with k_x = i * (2π/L), k_y = j * (2π/L), k_z = k * (2π/L)
- i, j, k = 1, 2, 3 (Fourier mode indices)
- A_{ijk} are amplitudes drawn from Gaussian distribution
- c_s is the sound speed
- x is the spatial coordinate vector

Example usage to generate a turbulent box with unit size, sound speed, and turbulent magnitude:

    python velocity_field.py --N 60 --L 1.0 --cs 1.0 --sigma 1.0 --seed 123 --output velocity_field.csv
"""

import os
import numpy as np
import argparse


def generate_velocity_field(N=60, L=1.0, c_s=1.0, sigma_A=0.05, seed=None):
    """
    Generate a 3D velocity vector field using Fourier modes.
    
    Parameters:
    -----------
    N : int
        Grid resolution (N×N×N)
    L : float
        Box size
    c_s : float
        Sound speed
    sigma_A : float
        Standard deviation for Gaussian amplitude distribution
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    velocity_field : ndarray
        4D array of shape (N, N, N, 3) containing velocity vector field
        Last dimension: [vx, vy, vz]
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create coordinate arrays
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    z = np.linspace(0, L, N, endpoint=False)
    
    # Create 3D coordinate grids
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize velocity field (complex) - 3 vector components
    velocity_field = np.zeros((N, N, N, 3), dtype=complex)
    
    # Define Fourier mode indices (i, j, k = 1, 2, 3)
    mode_indices = [(i, j, k) for i in range(1, 4) for j in range(1, 4) for k in range(1, 4)]
    
    print(f"Generating velocity field with {len(mode_indices)} Fourier modes...")
    print(f"Grid resolution: {N}×{N}×{N}")
    print(f"Box size: {L}")
    print(f"Sound speed: {c_s}")
    print(f"Amplitude std dev: {sigma_A}")
    
    # Generate velocity field by summing over Fourier modes
    for i, j, k in mode_indices:
        # Calculate wave vector components
        k_x = i * 2 * np.pi / L
        k_y = j * 2 * np.pi / L
        k_z = k * 2 * np.pi / L
        
        # Generate random vector amplitude from Gaussian distribution
        # Each component gets independent random amplitudes
        A_x_real = np.random.normal(0, sigma_A)
        A_x_imag = np.random.normal(0, sigma_A)
        A_y_real = np.random.normal(0, sigma_A)
        A_y_imag = np.random.normal(0, sigma_A)
        A_z_real = np.random.normal(0, sigma_A)
        A_z_imag = np.random.normal(0, sigma_A)
        
        A_ijk = np.array([
            A_x_real + 1j * A_x_imag,
            A_y_real + 1j * A_y_imag,
            A_z_real + 1j * A_z_imag
        ])
        
        # Calculate phase: k · x
        phase = k_x * X + k_y * Y + k_z * Z
        
        # Add contribution from this mode to each velocity component
        for comp in range(3):
            velocity_field[:, :, :, comp] += A_ijk[comp] * c_s * np.exp(1j * phase)
        
        print(f"Mode ({i},{j},{k}): k=({k_x:.3f}, {k_y:.3f}, {k_z:.3f})")
        print(f"  A_x=({A_x_real:.4f}+{A_x_imag:.4f}i), A_y=({A_y_real:.4f}+{A_y_imag:.4f}i), A_z=({A_z_real:.4f}+{A_z_imag:.4f}i)")
    
    # Take real part of velocity field (physical velocity is real)
    velocity_field_real = np.real(velocity_field)
    
    # Calculate statistics for vector field
    v_magnitude = np.sqrt(np.sum(velocity_field_real**2, axis=3))  # |v| = sqrt(vx^2 + vy^2 + vz^2)
    v_rms = np.sqrt(np.mean(v_magnitude**2))
    v_max = np.max(v_magnitude)
    v_min = np.min(v_magnitude)
    
    # Component-wise statistics
    components = ['vx', 'vy', 'vz']
    print(f"\nVelocity field statistics:")
    print(f"RMS velocity magnitude: {v_rms:.6f}")
    print(f"Max velocity magnitude: {v_max:.6f}")
    print(f"Min velocity magnitude: {v_min:.6f}")
    print(f"RMS/Mach number: {v_rms/c_s:.6f}")
    print(f"\nComponent-wise statistics:")
    
    for i, comp in enumerate(components):
        v_comp = velocity_field_real[:, :, :, i]
        v_mean = np.mean(v_comp)
        v_rms_comp = np.sqrt(np.mean(v_comp**2))
        v_max_abs = np.max(np.abs(v_comp))
        v_min_comp = np.min(v_comp)
        v_max_comp = np.max(v_comp)
        
        print(f"  {comp}: mean={v_mean:.6f}, RMS={v_rms_comp:.6f}, max(abs)={v_max_abs:.6f}, range=[{v_min_comp:.6f}, {v_max_comp:.6f}]")
    
    return velocity_field_real


def save_velocity_field(velocity_field, filename="velocity_field.csv", format="csv", L=1.0):
    """
    Save velocity vector field to file.
    
    Parameters:
    -----------
    velocity_field : ndarray
        4D velocity field array of shape (nx, ny, nz, 3)
    filename : str
        Output filename
    format : str
        Output format ("csv" or "npy")
    L : float
        Box size for coordinate bounds
    """
    
    if format.lower() == "csv":
        # Get grid dimensions
        nx, ny, nz, ncomp = velocity_field.shape
        
        # Calculate coordinate bounds
        xmin, xmax = 0.0, L
        ymin, ymax = 0.0, L
        zmin, zmax = 0.0, L
        
        # Reshape the 4D array to nx*ny rows with nz*ncomp columns
        # Each row contains: vx(z=0), vy(z=0), vz(z=0), vx(z=1), vy(z=1), vz(z=1), ...
        reshaped = velocity_field.reshape(ncomp * nx * ny, nz)
        
        # Write CSV file with header information
        header_lines = '\n'.join([
            f"{nx}",
            f"{ny}",
            f"{nz}",
            f"{xmin},{xmax}",
            f"{ymin},{ymax}",
            f"{zmin},{zmax}"
        ])
        # Write CSV file using np.savetxt with header comments
        np.savetxt(filename, reshaped, delimiter=',', 
                  header=header_lines, 
                  comments='', fmt='%.12e')
        
        print(f"Velocity field saved to {filename}")
        print(f"Shape: {velocity_field.shape}")
        print(f"Data points: {nx * ny * nz} × {ncomp} components")
        print(f"Output format: {nx * ny} rows × {nz * ncomp} columns")
        print(f"Grid dimensions: {nx}×{ny}×{nz}")
        print(f"Coordinate bounds: x∈[{xmin},{xmax}], y∈[{ymin},{ymax}], z∈[{zmin},{zmax}]")
        print(f"Column order: vx(z=0), vy(z=0), vz(z=0), vx(z=1), vy(z=1), vz(z=1), ...")
        
    elif format.lower() == "npy":
        np.save(filename, velocity_field)
        print(f"Velocity field saved to {filename}")
        print(f"Shape: {velocity_field.shape}")


def main():
    """Main function to run the velocity field computation."""
    
    parser = argparse.ArgumentParser(description="Generate velocity field for turbulent box simulation")
    parser.add_argument("--N", type=int, default=60, help="Grid resolution (default: 60)")
    parser.add_argument("--L", type=float, default=1.0, help="Box size (default: 1.0)")
    parser.add_argument("--cs", type=float, default=1.0, help="Sound speed (default: 1.0)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Amplitude std dev (default: 1.0)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="velocity_field.csv", help="Output filename")
    parser.add_argument("--format", type=str, choices=["csv", "npy"], default="csv", help="Output format")
    
    args = parser.parse_args()
    
    print("="*60)
    print("VELOCITY FIELD COMPUTATION FOR TURBULENT BOX SIMULATION")
    print("="*60)
    
    # Generate velocity field
    velocity_field = generate_velocity_field(
        N=args.N,
        L=args.L,
        c_s=args.cs,
        sigma_A=args.sigma,
        seed=args.seed
    )
    
    # Save to file
    save_velocity_field(velocity_field, args.output, args.format, args.L)
    
    print("\n" + "="*60)
    print("COMPUTATION COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
