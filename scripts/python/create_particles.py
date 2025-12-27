#!/usr/bin/env python3
"""
Particle creation script for Quokka astrophysical simulations.

This script generates initial particle distributions for various particle types
used in Quokka simulations, including CIC, Sink, Rad, CICRad, StochasticStellarPop,
and Test particles.

Usage:
    python create_particles.py [part_type] [n_star] [box_size] [sample_type] [output] [--m_min M_MIN] [--m_max M_MAX] [--velocity_disp VEL] [--lifetime TIME] [--center_origin] [--random_seed SEED]

Arguments:
    part_type: Particle type (CIC, Sink, Rad, CICRad, StochasticStellarPop, Test)
    n_star: Number of particles to generate
    box_size: Size of the simulation box in cm (domain: [0, box_size])
    sample_type: Spatial sampling type (uniform, Gaussian, Plummer)
    output: Output filename
    --m_min: Minimum stellar mass in M_sun (optional, default: 1.0)
    --m_max: Maximum stellar mass in M_sun (optional, default: 120.0)
    --velocity_disp: Velocity dispersion in cm/s (optional, default: 10.0 km/s)
    --lifetime: Particle lifetime in s (optional, default: 10.0 Myr)
    --center_origin: Center coordinate origin at domain center [-box_size/2, box_size/2] (optional, default: origin at (0,0,0))
    --random_seed: Random seed for reproducible results (optional)

Examples:
    # Generate 100 CIC particles in uniform distribution (domain [0, 1e18] cm)
    python create_particles.py CIC 100 1e18 uniform particles.txt

    # Generate 50 sink particles in Plummer sphere with centered origin (domain [-2.5e17, 2.5e17] cm)
    python create_particles.py Sink 50 5e17 Plummer sink_particles.txt --center_origin

    # Generate radiation particles with custom mass range
    python create_particles.py Rad 20 1e16 Gaussian rad_particles.txt --m_min 0.5 --m_max 60 --velocity_disp 5e6

    # Generate particles with reproducible results using random seed
    python create_particles.py CIC 50 1e16 Plummer cluster.txt --random_seed 42

Physical parameters (CGS units):
    - mass range: 1-120 solar masses (Salpeter IMF, power law with exponent -2.35)
    - velocity_dispersion: 10 km/s = 1e7 cm/s
    - lifetime: 10 Myr = 3.156e14 s
"""

import argparse
import numpy as np
import sys
from typing import Tuple, List

# Physical constants (CGS units)
M_SUN = 1.989e33  # g
KM_S = 1e5        # cm/s
MYR = 3.156e14    # s (seconds in 1 million years)
PC = 3.086e18     # cm (parsec)

# Default physical parameters for star cluster
DEFAULT_MEAN_MASS = 1.0 * M_SUN      # 1 solar mass
DEFAULT_VELOCITY_DISP = 10.0 * KM_S  # 10 km/s velocity dispersion
DEFAULT_LIFETIME = 10.0 * MYR        # 10 million years
DEFAULT_N_GROUPS = 1                 # Default number of radiation groups

# Particle type definitions matching C++ enum
PARTICLE_TYPES = {
    'Rad': 'Rad',
    'CIC': 'CIC',
    'CICRad': 'CICRad',
    'StochasticStellarPop': 'StochasticStellarPop',
    'Sink': 'Sink',
    'Test': 'Test'
}

def sample_positions_uniform(n_particles: int, box_size: float) -> np.ndarray:
    """Sample positions uniformly within a cube."""
    return np.random.uniform(-box_size/2, box_size/2, (n_particles, 3))

def sample_positions_gaussian(n_particles: int, box_size: float) -> np.ndarray:
    """Sample positions from a 3D Gaussian distribution."""
    # Use box_size/6 as sigma to keep most particles within the box
    sigma = box_size / 6.0
    positions = np.random.normal(0, sigma, (n_particles, 3))
    return positions

def sample_positions_plummer(n_particles: int, box_size: float) -> np.ndarray:
    """Sample positions from a Plummer sphere profile."""
    # Plummer radius - use box_size/4 as a typical cluster radius
    a = box_size / 4.0

    positions = []
    for _ in range(n_particles):
        # Sample radius from Plummer distribution
        while True:
            x = np.random.uniform(0, 1)
            r = a / np.sqrt(x**(-2/3) - 1)
            if r < box_size/2:  # Keep within box
                break

        # Sample direction uniformly on sphere
        theta = np.arccos(2 * np.random.uniform(0, 1) - 1)
        phi = 2 * np.pi * np.random.uniform(0, 1)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        positions.append([x, y, z])

    return np.array(positions)

def sample_velocities(n_particles: int, velocity_disp: float) -> np.ndarray:
    """Sample velocities from a 3D Gaussian distribution."""
    return np.random.normal(0, velocity_disp, (n_particles, 3))

def sample_masses(n_particles: int, m_min: float, m_max: float) -> np.ndarray:
    """Sample masses from a Salpeter IMF (power law with exponent -2.35)."""
    alpha = 2.35  # Salpeter IMF exponent

    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n_particles)

    # Inverse CDF for power law: m = [(m_min^(1-α) + (m_max^(1-α) - m_min^(1-α)) * u)]^(1/(1-α))
    # For α=2.35, 1-α = -1.35
    m_min_pow = m_min ** (1 - alpha)
    m_max_pow = m_max ** (1 - alpha)

    masses = (m_min_pow + (m_max_pow - m_min_pow) * u) ** (1 / (1 - alpha))

    return masses

def get_particle_data_components(part_type: str) -> Tuple[List[str], int]:
    """Get the data components for each particle type (excluding positions)."""
    if part_type in ['CIC', 'Sink']:
        # mass, vx, vy, vz
        components = ['mass', 'vx', 'vy', 'vz']
        n_extra = 4
    elif part_type == 'Rad':
        # birth_time, death_time, luminosity (for each group)
        components = ['birth_time', 'death_time'] + [f'lum_{i}' for i in range(DEFAULT_N_GROUPS)]
        n_extra = 2 + DEFAULT_N_GROUPS
    elif part_type == 'CICRad':
        # mass, vx, vy, vz, birth_time, death_time, luminosity (for each group)
        components = ['mass', 'vx', 'vy', 'vz', 'birth_time', 'death_time'] + [f'lum_{i}' for i in range(DEFAULT_N_GROUPS)]
        n_extra = 6 + DEFAULT_N_GROUPS
    elif part_type == 'StochasticStellarPop':
        # mass, vx, vy, vz, birth_time, death_time, mass_at_birth, luminosity (for each group)
        components = ['mass', 'vx', 'vy', 'vz', 'birth_time', 'death_time', 'mass_at_birth'] + [f'lum_{i}' for i in range(DEFAULT_N_GROUPS)]
        n_extra = 7 + DEFAULT_N_GROUPS
    elif part_type == 'Test':
        # mass, vx, vy, vz, birth_time, death_time, luminosity (for each group)
        components = ['mass', 'vx', 'vy', 'vz', 'birth_time', 'death_time'] + [f'lum_{i}' for i in range(DEFAULT_N_GROUPS)]
        n_extra = 6 + DEFAULT_N_GROUPS
    else:
        raise ValueError(f"Unknown particle type: {part_type}")

    return components, n_extra

def generate_particle_data(part_type: str, n_particles: int, box_size: float,
                          sample_type: str, m_min: float, m_max: float, velocity_disp: float,
                          lifetime: float, center_origin: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complete particle data including positions and all required components."""

    # Sample positions (always around center)
    if sample_type == 'uniform':
        positions = sample_positions_uniform(n_particles, box_size)
    elif sample_type == 'Gaussian':
        positions = sample_positions_gaussian(n_particles, box_size)
    elif sample_type == 'Plummer':
        positions = sample_positions_plummer(n_particles, box_size)
    else:
        raise ValueError(f"Unknown sampling type: {sample_type}")

    # Shift positions if origin should be at (0,0,0) instead of domain center
    if not center_origin:
        positions += box_size / 2.0

    # Get components for this particle type
    components, n_extra = get_particle_data_components(part_type)

    # Initialize data array (positions + extra components)
    data = np.zeros((n_particles, 3 + n_extra))
    data[:, :3] = positions  # x, y, z positions

    # Sample masses and velocities
    masses = sample_masses(n_particles, m_min, m_max)
    velocities = sample_velocities(n_particles, velocity_disp)

    col_idx = 3  # Start after positions

    for comp in components:
        if comp == 'mass':
            data[:, col_idx] = masses
            col_idx += 1
        elif comp in ['vx', 'vy', 'vz']:
            if comp == 'vx':
                data[:, col_idx] = velocities[:, 0]
            elif comp == 'vy':
                data[:, col_idx] = velocities[:, 1]
            elif comp == 'vz':
                data[:, col_idx] = velocities[:, 2]
            col_idx += 1
        elif comp == 'birth_time':
            # Particles are born at t=0
            data[:, col_idx] = 0.0
            col_idx += 1
        elif comp == 'death_time':
            # Particles die after lifetime
            data[:, col_idx] = lifetime
            col_idx += 1
        elif comp == 'mass_at_birth':
            # Same as current mass for simplicity
            data[:, col_idx] = masses
            col_idx += 1
        elif comp.startswith('lum_'):
            # Set luminosity proportional to mass (simple scaling)
            # Typical stellar luminosity ~ mass^3.5, but use mass for simplicity
            luminosity_per_group = masses * 1e-3  # Arbitrary scaling, adjust as needed
            data[:, col_idx] = luminosity_per_group
            col_idx += 1

    return positions, data

def save_particles_to_file(positions: np.ndarray, data: np.ndarray, output_file: str):
    """Save particle data to ASCII file in Quokka format."""
    n_particles = len(data)

    with open(output_file, 'w') as f:
        # Write number of particles
        f.write(f"{n_particles}\n")

        # Write each particle's data
        for i in range(n_particles):
            line = " ".join(f"{val:.6e}" for val in data[i])
            f.write(f"{line}\n")

        # Empty line at end (matching example format)
        f.write("\n")

def print_particle_info(part_type: str, n_particles: int, positions: np.ndarray,
                        data: np.ndarray, components: List[str]):
    """Print summary information about generated particles."""
    print(f"\nGenerated {n_particles} {part_type} particles")
    print(f"Components: {components}")

    # Position statistics
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    pos_mean = np.mean(positions, axis=0)
    print(f"Position range: x=[{pos_min[0]:.2e}, {pos_max[0]:.2e}] cm")
    print(f"                y=[{pos_min[1]:.2e}, {pos_max[1]:.2e}] cm")
    print(f"                z=[{pos_min[2]:.2e}, {pos_max[2]:.2e}] cm")
    print(f"Position center: ({pos_mean[0]:.2e}, {pos_mean[1]:.2e}, {pos_mean[2]:.2e}) cm")

    # Mass statistics (if applicable)
    if 'mass' in components:
        mass_idx = 3 + components.index('mass')
        masses = data[:, mass_idx] / M_SUN  # Convert to M_sun
        mass_min = np.min(masses)
        mass_max = np.max(masses)
        mass_median = np.median(masses)
        print(".1f")

    # Velocity statistics (if applicable)
    if 'vx' in components:
        vx_idx = 3 + components.index('vx')
        vy_idx = 3 + components.index('vy')
        vz_idx = 3 + components.index('vz')
        velocities = data[:, [vx_idx, vy_idx, vz_idx]]
        vel_disp = np.std(velocities, axis=0) / KM_S
        print(".1f")

def main():
    parser = argparse.ArgumentParser(
        description='Generate initial particle distributions for Quokka simulations',
        usage='%(prog)s [part_type] [n_star] [box_size] [sample_type] [output] [--m_min M_MIN] [--m_max M_MAX] [--velocity_disp VEL] [--lifetime TIME] [--center_origin] [--random_seed SEED]'
    )
    parser.add_argument('part_type', choices=PARTICLE_TYPES.keys(),
                       help='Particle type to generate')
    parser.add_argument('n_star', type=int,
                       help='Number of particles to generate')
    parser.add_argument('box_size', type=float,
                       help='Size of simulation box in cm')
    parser.add_argument('sample_type', choices=['uniform', 'Gaussian', 'Plummer'],
                       help='Spatial sampling distribution')
    parser.add_argument('output',
                       help='Output filename')
    parser.add_argument('--m_min', type=float, default=1.0,
                       help='Minimum stellar mass in M_sun (default: 1.0)')
    parser.add_argument('--m_max', type=float, default=120.0,
                       help='Maximum stellar mass in M_sun (default: 120.0)')
    parser.add_argument('--velocity_disp', type=float, default=DEFAULT_VELOCITY_DISP,
                       help=f'Velocity dispersion in cm/s (default: {DEFAULT_VELOCITY_DISP/KM_S:.1f} km/s)')
    parser.add_argument('--lifetime', type=float, default=DEFAULT_LIFETIME,
                       help=f'Particle lifetime in s (default: {DEFAULT_LIFETIME/MYR:.1f} Myr)')
    parser.add_argument('--center_origin', action='store_true',
                       help='Center the coordinate origin at the domain center [-box_size/2, box_size/2] (default: origin at (0,0,0))')
    parser.add_argument('--random_seed', type=int,
                       help='Random seed for reproducible results (optional)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")

    # Validate inputs
    if args.n_star <= 0:
        print("Error: n_star must be positive")
        sys.exit(1)
    if args.box_size <= 0:
        print("Error: box_size must be positive")
        sys.exit(1)

    domain_str = f"[-{args.box_size/2:.2e}, {args.box_size/2:.2e}]" if args.center_origin else f"[0, {args.box_size:.2e}]"
    print(f"Generating {args.n_star} {args.part_type} particles...")
    print(f"Box size: {args.box_size:.2e} cm (domain: {domain_str} cm)")
    print(f"Sampling type: {args.sample_type}")
    print(f"Mass range: {args.m_min:.1f} - {args.m_max:.1f} M_sun (Salpeter IMF)")
    print(f"Velocity dispersion: {args.velocity_disp/KM_S:.1f} km/s")
    print(f"Lifetime: {args.lifetime/MYR:.1f} Myr")

    try:
        # Generate particle data
        positions, data = generate_particle_data(
            args.part_type, args.n_star, args.box_size, args.sample_type,
            args.m_min * M_SUN, args.m_max * M_SUN, args.velocity_disp, args.lifetime, args.center_origin
        )

        # Get component information for printing
        components, _ = get_particle_data_components(args.part_type)

        # Print statistics
        print_particle_info(args.part_type, args.n_star, positions, data, components)

        # Save to file
        save_particles_to_file(positions, data, args.output)
        print(f"\nParticle data saved to: {args.output}")

    except Exception as e:
        print(f"Error generating particles: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
