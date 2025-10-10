#!/usr/bin/env python3
"""
Generate a star cluster data file with stars following a Kroupa IMF (M > 5 Msun),
Plummer spatial distribution, and virial equilibrium.

Usage:
    python init_star_cluster.py <n_stars> <half_mass_radius_pc> <n_radiation_groups>

Output format:
    First line: number of particles N
    Subsequent lines: x y z mass vx vy vz birth_time death_time lum0 lum1 ... lum(N-1)
    
All units are in CGS (cm, g, cm/s, s).

Command to generate a star cluster with 500 stars, 20 pc half-mass radius, and 4 radiation groups:
    python3 init_star_cluster.py 500 20 4

Output file: cluster_N500_r20.0.txt
"""

import numpy as np
import argparse
import sys

# Global random seed
RANDOM_SEED = 42

# Physical constants in CGS
PARSEC_IN_CM = 3.085677581491367e18      # 1 pc in cm
MSUN_IN_G = 1.98841586e33                 # 1 solar mass in g
KM_TO_CM = 1.0e5                          # 1 km in cm
YEAR_IN_S = 3.1536e7                      # 1 year in seconds
MYR_IN_S = 3.1536e13                      # 1 Myr in seconds
G_CGS = 6.67430e-8                        # Gravitational constant in cm³/(g·s²)


def sample_kroupa_imf(n_stars, m_min=5.0, m_max=120.0, alpha=2.3):
    """
    Sample stellar masses from Kroupa IMF for M > 5 Msun.
    
    The IMF is a power law: dN/dM ∝ M^(-α) where α = 2.3 for M > 0.5 Msun.
    
    Parameters:
    -----------
    n_stars : int
        Number of stars to generate
    m_min : float
        Minimum stellar mass in solar masses (default: 5.0)
    m_max : float
        Maximum stellar mass in solar masses (default: 120.0)
    alpha : float
        IMF power-law index (default: 2.3 for Kroupa)
    
    Returns:
    --------
    masses : ndarray
        Array of stellar masses in solar masses
    """
    # For power law dN/dM ∝ M^(-α), the CDF is:
    # P(M) = (M^(1-α) - M_min^(1-α)) / (M_max^(1-α) - M_min^(1-α))
    
    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n_stars)
    
    # Invert the CDF to get masses
    beta = 1.0 - alpha
    if np.abs(beta) < 1e-10:  # α ≈ 1 case (logarithmic)
        masses = m_min * np.exp(u * np.log(m_max / m_min))
    else:
        masses = (u * (m_max**beta - m_min**beta) + m_min**beta)**(1.0 / beta)
    
    return masses


def plummer_positions(n_stars, half_mass_radius):
    """
    Generate stellar positions following a Plummer model.
    
    The Plummer model has density profile:
    ρ(r) = (3M / 4πa³) * (1 + r²/a²)^(-5/2)
    
    The half-mass radius r_h ≈ 1.305 * a, where a is the Plummer scale radius.
    
    Parameters:
    -----------
    n_stars : int
        Number of stars
    half_mass_radius : float
        Half-mass radius in parsecs
    
    Returns:
    --------
    positions : ndarray
        Array of shape (n_stars, 3) with positions in parsecs
    """
    # Convert half-mass radius to Plummer scale radius
    a = half_mass_radius / 1.305
    
    # Sample radii from Plummer distribution using inverse transform sampling
    # For Plummer: M(r) / M_total = (r² / (a² + r²))^(3/2)
    # Inverting: r = a * (u^(-2/3) - 1)^(-1/2), where u is uniform on (0,1)
    u = np.random.uniform(0, 1, n_stars)
    radii = a / np.sqrt(u**(-2.0/3.0) - 1.0)
    
    # Generate random directions (isotropic)
    cos_theta = np.random.uniform(-1, 1, n_stars)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    
    # Convert to Cartesian coordinates
    x = radii * sin_theta * np.cos(phi)
    y = radii * sin_theta * np.sin(phi)
    z = radii * cos_theta
    
    positions = np.column_stack([x, y, z])
    return positions


def compute_virial_velocities(positions, masses, half_mass_radius):
    """
    Compute velocities for stars in virial equilibrium.
    
    For virial equilibrium: 2K + U = 0, where K is kinetic energy and U is potential energy.
    For a Plummer sphere: U = -GM² / (2a) ≈ -GM² / (2 * r_h / 1.305)
    
    We assign velocities randomly with a velocity dispersion that satisfies virial equilibrium.
    
    Parameters:
    -----------
    positions : ndarray
        Star positions in cm, shape (n_stars, 3)
    masses : ndarray
        Star masses in g, shape (n_stars,)
    half_mass_radius : float
        Half-mass radius in cm
    
    Returns:
    --------
    velocities : ndarray
        Array of shape (n_stars, 3) with velocities in cm/s
    """
    # Plummer scale radius
    a = half_mass_radius / 1.305
    
    # Total mass
    M_total = np.sum(masses)
    
    # Virial velocity dispersion for Plummer model
    # <v²> = GM / (6a) for a Plummer sphere
    v_disp_squared = G_CGS * M_total / (6.0 * a)
    v_disp = np.sqrt(v_disp_squared)
    
    # Generate velocities from a 3D Gaussian distribution
    # Each component has dispersion v_disp / sqrt(3) to get total <v²> = 3 * (v_disp/sqrt(3))² = v_disp²
    velocities = np.random.normal(0, v_disp / np.sqrt(3.0), size=(len(masses), 3))
    
    # Remove center-of-mass velocity (important for virial equilibrium)
    v_com = np.sum(masses[:, np.newaxis] * velocities, axis=0) / M_total
    velocities -= v_com
    
    return velocities


def stellar_lifetime(mass):
    """
    Estimate main-sequence lifetime of a star based on its mass.
    
    Uses the approximate relation: τ_MS ≈ 10 Gyr * (M/Msun)^(-2.5)
    
    Parameters:
    -----------
    mass : float or ndarray
        Stellar mass in solar masses
    
    Returns:
    --------
    lifetime : float or ndarray
        Main-sequence lifetime in seconds (CGS)
    """
    tau_sun = 1.0e10 * YEAR_IN_S  # Solar main-sequence lifetime in seconds
    return tau_sun * mass**(-2.5)


def generate_star_cluster(n_stars, half_mass_radius_pc, n_radiation_groups, output_file=None):
    """
    Generate a star cluster data file in CGS units.
    
    Parameters:
    -----------
    n_stars : int
        Total number of stars
    half_mass_radius_pc : float
        Half-mass radius in parsecs (for convenience, will be converted to CGS internally)
    n_radiation_groups : int
        Number of radiation groups (determines number of luminosity columns)
    output_file : str, optional
        Output filename. If None, auto-generated as cluster_N{nstars}_r{radius}.txt
    """
    print(f"Generating star cluster with {n_stars} stars...")
    print(f"Half-mass radius: {half_mass_radius_pc} pc")
    print(f"Number of radiation groups: {n_radiation_groups}")
    
    # Generate default output filename if not provided
    if output_file is None:
        output_file = f"cluster_N{n_stars}_r{half_mass_radius_pc:.1f}.txt"
    
    # Generate stellar masses from Kroupa IMF (M > 5 Msun) - work in solar masses first
    masses_msun = sample_kroupa_imf(n_stars)
    print(f"Mass range: {masses_msun.min():.2f} - {masses_msun.max():.2f} Msun")
    print(f"Total mass: {masses_msun.sum():.2f} Msun")
    
    # Generate positions from Plummer model (in pc first)
    positions_pc = plummer_positions(n_stars, half_mass_radius_pc)
    print(f"Position range: {np.abs(positions_pc).max():.2f} pc")
    
    # Convert to CGS for velocity calculation
    positions_cgs = positions_pc * PARSEC_IN_CM
    masses_cgs = masses_msun * MSUN_IN_G
    half_mass_radius_cgs = half_mass_radius_pc * PARSEC_IN_CM
    
    # Generate velocities in virial equilibrium (in CGS)
    velocities_cgs = compute_virial_velocities(positions_cgs, masses_cgs, half_mass_radius_cgs)
    v_disp_km_s = np.std(velocities_cgs) / KM_TO_CM  # Convert to km/s for display
    print(f"Velocity dispersion: {v_disp_km_s:.2f} km/s")
    
    # Generate random birth times from -3 Myr to 0 (in CGS seconds)
    birth_times_cgs = np.random.uniform(-3.0 * MYR_IN_S, 0.0, n_stars)
    
    # Calculate death times based on stellar lifetimes (in CGS seconds)
    death_times_cgs = stellar_lifetime(masses_msun)
    
    # Initialize luminosities (all zeros)
    luminosities = np.zeros((n_stars, n_radiation_groups))
    
    # Write output file (all in CGS units)
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        # First line: number of particles
        f.write(f"{n_stars}\n")
        
        # Subsequent lines: particle properties (all CGS)
        for i in range(n_stars):
            # positions (3), mass (1), velocities (3), birth_time (1), death_time (1), luminosities (n_groups)
            line_parts = [
                f"{positions_cgs[i, 0]:.6e}",
                f"{positions_cgs[i, 1]:.6e}",
                f"{positions_cgs[i, 2]:.6e}",
                f"{masses_cgs[i]:.6e}",
                f"{velocities_cgs[i, 0]:.6e}",
                f"{velocities_cgs[i, 1]:.6e}",
                f"{velocities_cgs[i, 2]:.6e}",
                f"{birth_times_cgs[i]:.6e}",
                f"{death_times_cgs[i]:.6e}",
            ]
            # Add luminosity columns
            for j in range(n_radiation_groups):
                line_parts.append(f"{luminosities[i, j]:.6e}")
            
            f.write(" ".join(line_parts) + "\n")
    
    print(f"Successfully wrote {n_stars} stars to {output_file}")
    
    # Print some statistics
    print("\n=== Cluster Statistics ===")
    print(f"Total mass: {masses_msun.sum():.2f} Msun")
    print(f"Mean mass: {masses_msun.mean():.2f} Msun")
    print(f"Median mass: {np.median(masses_msun):.2f} Msun")
    print(f"Velocity dispersion: {v_disp_km_s:.2f} km/s")
    print(f"Mean lifetime: {death_times_cgs.mean() / YEAR_IN_S:.2e} years")
    print(f"Lifetime range: {death_times_cgs.min() / YEAR_IN_S:.2e} - {death_times_cgs.max() / YEAR_IN_S:.2e} years")
    print(f"Birth time range: {birth_times_cgs.min() / MYR_IN_S:.2f} - {birth_times_cgs.max() / MYR_IN_S:.2f} Myr")


def main():
    """Main function to parse arguments and generate cluster."""
    parser = argparse.ArgumentParser(
        description="Generate a star cluster data file with Kroupa IMF, Plummer model, and virial equilibrium.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format:
    First line: number of particles N
    Subsequent lines: x y z mass vx vy vz birth_time death_time lum0 lum1 ... lum(N-1)
    
    All units are in CGS:
        positions: cm
        mass: g
        velocities: cm/s
        birth_time: s (randomly drawn from -3 Myr to 0)
        death_time: s (stellar lifetime)
        luminosities: dimensionless (all zeros)
    
    Default output filename: cluster_N{nstars}_r{radius}.txt
        
Example:
    python3 init_star_cluster.py 1000 1.0 4
    
    This generates 1000 stars with 1 pc half-mass radius and 4 radiation groups.
    Output file: cluster_N1000_r1.0.txt
    """
    )
    
    parser.add_argument("n_stars", type=int, help="Total number of stars")
    parser.add_argument("half_mass_radius", type=float, help="Half-mass radius in parsecs")
    parser.add_argument("n_radiation_groups", type=int, help="Number of radiation groups")
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output filename (default: auto-generated as cluster_N{nstars}_r{radius}.txt)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, 
                        help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_stars <= 0:
        print("Error: n_stars must be positive", file=sys.stderr)
        sys.exit(1)
    if args.half_mass_radius <= 0:
        print("Error: half_mass_radius must be positive", file=sys.stderr)
        sys.exit(1)
    if args.n_radiation_groups <= 0:
        print("Error: n_radiation_groups must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed
    np.random.seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
    # Generate cluster
    generate_star_cluster(args.n_stars, args.half_mass_radius, 
                         args.n_radiation_groups, args.output)


if __name__ == "__main__":
    main()

