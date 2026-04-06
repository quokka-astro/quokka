import os
import numpy as np
from scipy.integrate import solve_ivp
from datatable import DataTable
from scipy.optimize import brentq
from astropy import units as u
from astropy import constants as const
import pandas as pd
import sys
import matplotlib.pyplot as plt

import argparse

# Constants and Parameters
# ------------------------
G = const.G.cgs.value

# Conversions to CGS
pc_to_cm = u.pc.to(u.cm)
Msun_to_g = u.M_sun.to(u.g)
kms_to_cms = (u.km/u.s).to(u.cm/u.s)
Msun_pc2_to_g_cm2 = (u.M_sun/u.pc**2).to(u.g/u.cm**2)
Msun_pc3_to_g_cm3 = (u.M_sun/u.pc**3).to(u.g/u.cm**3)
mpcc = const.m_p.cgs.value

# Global variables will be set in main
Sigma_star = None
z_star = None
rho_dm = None
R0 = None
sigma_1_sq = None
target_Sigma_gas = None

Density_floor = 1e-33 # g/cm^3

def Phi_ext(z):
    """
    Calculate the external gravitational potential at height z (CGS).
    
    Phi_ext = 2*pi*G*Sigma_* * z_* * [sqrt(1+(z/z_*)^2) - 1] 
              + 2*pi*G*rho_dm * R0^2 * ln(1+(z/R0)^2)
              
    Returns potential in cm^2/s^2 (erg/g).
    """
    # Term 1: Star disk
    term1 = 2 * np.pi * G * Sigma_star * z_star * (np.sqrt(1 + (z / z_star)**2) - 1)
    
    # Term 2: Dark Matter halo
    term2 = 2 * np.pi * G * rho_dm * R0**2 * np.log(1 + (z / R0)**2)
    
    return term1 + term2

def g_ext(z):
    """
    Calculate the external gravitational acceleration at height z (CGS).
    g_ext = - dPhi_ext / dz
    
    Phi_ext = 2*pi*G*Sigma_* * z_* * [sqrt(1+(z/z_*)^2) - 1] 
              + 2*pi*G*rho_dm * R0^2 * ln(1+(z/R0)^2)
              
    Returns acceleration in cm/s^2.
    """
    # Term 1 derivative (Star disk)
    term1 = (2 * np.pi * G * Sigma_star * z_star * 
             (z / z_star**2) / np.sqrt(1 + (z / z_star)**2))
    
    # Term 2 derivative (Dark Matter halo)
    term2 = (2 * np.pi * G * rho_dm * R0**2 * 
             (2 * z / R0**2) / (1 + (z / R0)**2))
             
    # Force is attractive (negative z direction for z>0)
    return - (term1 + term2)

def odes(z, y):
    """
    System of ODEs (CGS):
    y[0] = rho_1 (density)
    y[1] = g_1 (acceleration due to gas)
    y[2] = Sigma_accumulated (integral of 2 * rho_1)
    """
    rho_1 = y[0]
    g_1 = y[1]
    
    # Apply density floor to prevent numerical issues
    rho_floor = Density_floor  # g/cm^3
    if rho_1 < rho_floor:
        rho_1 = rho_floor
    
    drho_dz = (rho_1 / sigma_1_sq) * (g_1 + g_ext(z))
    dg_dz = -4 * np.pi * G * rho_1
    dSigma_dz = 2 * rho_1
    
    return [drho_dz, dg_dz, dSigma_dz]

def integrate_profile(rho_0_guess, z_max, t_eval=None):
    """
    Integrate the ODEs from z=0 to z_max given rho_1(0) = rho_0_guess.
    g_1(0) = 0 by symmetry.
    Sigma(0) = 0.
    """
    y0 = [rho_0_guess, 0.0, 0.0]
    
    # Event to stop if density reaches floor
    def density_floor_event(z, y):
        return y[0] - Density_floor
    density_floor_event.terminal = True
    density_floor_event.direction = -1 # only trigger when going from above to below
    
    # Solve ODE
    sol = solve_ivp(odes, (0, z_max), y0, t_eval=t_eval, events=density_floor_event, rtol=1e-10, atol=1e-12)
    
    # Calculate surface density Sigma_gas = y[2] at the end
    Sigma_calc = sol.y[2][-1]
    
    return Sigma_calc, sol

def objective(rho_0_guess):
    """Function to find root of: Calculated_Sigma - Target_Sigma"""
    # We need a sufficient z_max to capture most of the mass.
    # z_star is 245 pc. 20 * z_star is ~5000 pc.
    z_max_integration = 20.0 * z_star # Go deep enough (~5k pc)
    # z_max_integration = 2.0 * z_star # (~500 pc)
    Sigma_calc, _ = integrate_profile(rho_0_guess, z_max_integration)
    val = Sigma_calc - target_Sigma_gas
    # print(f"DEBUG: rho_0 = {rho_0_guess:.6e}, Sigma_calc = {Sigma_calc:.6e}, Val = {val:.6e}")
    return val

def main():
    parser = argparse.ArgumentParser(description='Solve for vertical density profile.')
    parser.add_argument('--Sigma_gas', type=float, default=13.0, help='Gas surface density in M_sun/pc^2')
    parser.add_argument('--Sigma_star', type=float, default=42.0, help='Star surface density in M_sun/pc^2')
    parser.add_argument('--sigma_1', type=float, default=7.0, help='Gas velocity dispersion in km/s')
    parser.add_argument('--rho_dm', type=float, default=6.4e-3, help='Dark matter density in M_sun/pc^3')
    parser.add_argument('--R0', type=float, default=8000.0, help='Galactic radius in pc')
    parser.add_argument('--z_star', type=float, default=245.0, help='Star scale height in pc')
    parser.add_argument('--output_suffix', type=str, default='', help='Output filename suffix')
    
    args = parser.parse_args()
    
    global Sigma_star, z_star, rho_dm, R0, sigma_1_sq, target_Sigma_gas

    # Set parameters from args
    target_Sigma_gas_Msun_pc2 = args.Sigma_gas
    Sigma_star_val_Msun_pc2 = args.Sigma_star
    sigma_1_val_kms = args.sigma_1
    rho_dm_val_Msun_pc3 = args.rho_dm
    R0_val_pc = args.R0
    z_star_val_pc = args.z_star

    # Convert to CGS
    sigma_1 = sigma_1_val_kms * kms_to_cms
    Sigma_star = Sigma_star_val_Msun_pc2 * Msun_pc2_to_g_cm2
    z_star = z_star_val_pc * pc_to_cm
    rho_dm = rho_dm_val_Msun_pc3 * Msun_pc3_to_g_cm3
    R0 = R0_val_pc * pc_to_cm
    target_Sigma_gas = target_Sigma_gas_Msun_pc2 * Msun_pc2_to_g_cm2
    sigma_1_sq = sigma_1**2

    print(f"Solving for Vertical Density Profile...")
    print(f"Target Sigma_gas = {target_Sigma_gas_Msun_pc2} M_sun/pc^2")
    print(f"Sigma_star = {Sigma_star_val_Msun_pc2} M_sun/pc^2")
    print(f"sigma_1 = {sigma_1_val_kms} km/s")
    print(f"rho_dm = {rho_dm_val_Msun_pc3} M_sun/pc^3")
    print(f"R0 = {R0_val_pc} pc")

    # 1. Find the correct central density rho_0 (in cgs)
    # Search range: 1e-5 to 10.0 M_sun/pc^3 converted to cgs
    # 1 m_H/cm^3 is approx 1.67e-24 g/cm^3
    rho_min = 0.01 * mpcc
    rho_max = 100.0 * mpcc
    rho_ref = 1.0 * mpcc
    
    try:
        # Need very small xtol because rho in cgs is ~1e-24
        rho_0_solution = brentq(objective, rho_min, rho_max, xtol=1e-10 * rho_ref)
    except Exception as e:
        print(f"Error finding root: {e}")
        # Try to print values at bounds to debug
        print(f"Val at min: {objective(rho_min)}")
        print(f"Val at max: {objective(rho_max)}")
        sys.exit(1)
        
    print(f"Converged! rho_1(0) = {rho_0_solution:.6e} g/cm^3 = {rho_0_solution/mpcc:.6e} mpcc")
    
    # 2. Generate solution on the requested grid
    # theta = z / z_star, theta in [0, 20]
    theta_max = 20.0
    z_max_final = theta_max * z_star
    
    # Create evaluation points (e.g., 1000 points)
    theta_eval = np.linspace(0, theta_max, 5001)
    z_eval = theta_eval * z_star
    
    # Re-integrate with denser points for plotting, but use the rho_0 found
    Sigma_final, sol = integrate_profile(rho_0_solution, z_max_final * 1.01, t_eval=z_eval)
    
    # Recalculate Sigma more accurately for validation using quad if needed, or just trust the dense grid
    # Let's trust the dense grid integration for now, but note that the optimization used a different grid implicitly via solve_ivp

    sigma_final_Msun_pc2 = Sigma_final/Msun_pc2_to_g_cm2
    print(f"Calculated Sigma_gas over [0, {theta_max} z*]: {sigma_final_Msun_pc2:.4f} M_sun/pc^2")
    
    # Check if integration stopped early
    if sol.status == 1:
        z_stop_pc = sol.t_events[0][0] / pc_to_cm if len(sol.t_events[0]) > 0 else z_max_final / pc_to_cm
        print(f"Integration stopped at z = {z_stop_pc:.2f} pc (density reached floor of {Density_floor:.2e} g/cm^3)")
    
    # Validation check
    error = abs(sigma_final_Msun_pc2 - target_Sigma_gas_Msun_pc2) / target_Sigma_gas_Msun_pc2
    if error < 0.01:
        print(f"SUCCESS: Calculated surface density agrees with target ({target_Sigma_gas_Msun_pc2} M_sun/pc^2) within 1%.")
    else:
        print(f"WARNING: Calculated surface density deviates from target by {error*100:.2f}%.")

    # 3. Prepare output
    # theta = z/z_star
    # xi = rho / rho_0
    
    # Handle the case where solution stopped early (density -> floor)
    rho_sol = sol.y[0]
    g_1_sol = sol.y[1]
    
    if len(rho_sol) < len(theta_eval):
        # Pad with floor density
        padding_rho = np.full(len(theta_eval) - len(rho_sol), Density_floor)
        rho_sol = np.concatenate([rho_sol, padding_rho])
        # Pad g_1 with last value (constant beyond cutoff)
        padding_g1 = np.full(len(theta_eval) - len(g_1_sol), g_1_sol[-1] if len(g_1_sol) > 0 else 0.0)
        g_1_sol = np.concatenate([g_1_sol, padding_g1])
    
    xi_sol = rho_sol / rho_0_solution
    
    # Calculate total gravitational field and potential
    g_ext_sol = np.array([g_ext(z) for z in z_eval])
    g_tot_sol = g_1_sol + g_ext_sol
    
    Phi_ext_sol = np.array([Phi_ext(z) for z in z_eval])
    # Phi_1 is obtained by integrating -g_1 from 0 to z
    # Phi_1(z) = -integral_0^z g_1(z') dz'
    Phi_1_sol = -np.cumsum(g_1_sol) * (z_eval[1] - z_eval[0]) if len(z_eval) > 1 else np.zeros_like(z_eval)
    Phi_1_sol -= Phi_1_sol[0] # Set Phi_1(0) = 0
    Phi_tot_sol = Phi_1_sol + Phi_ext_sol
    
    # Conversion back for output if needed
    rho_msun_pc3 = rho_sol / Msun_pc3_to_g_cm3
    z_pc = z_eval / pc_to_cm
    
    df = pd.DataFrame({
        'theta': theta_eval,
        'xi': xi_sol,
        'z_pc': z_pc,
        'z_cm': z_eval,
        'rho_msun_pc3': rho_msun_pc3,
        'rho_g_cm3': rho_sol,
        'g_1_cgs': g_1_sol,
        'g_ext_cgs': g_ext_sol,
        'g_tot_cgs': g_tot_sol,
        'Phi_1_cgs': Phi_1_sol,
        'Phi_ext_cgs': Phi_ext_sol,
        'Phi_tot_cgs': Phi_tot_sol
    })
    
    os.makedirs('output', exist_ok=True)
    output_filename = f'output/disk_solution_all_vars_{args.output_suffix}.csv'
    df.to_csv(output_filename, index=False, float_format='%.12e')
    print(f"Solution saved to {output_filename}")

    # Save using DataTable format
    # Input dimension: z
    ndim = 1
    nx = [len(z_eval)]
    nout = 3
    
    input_names = ["z"]
    output_names = ["g_1", "g_ext", "Phi_tot"]
    
    input_units = ["cm"]
    output_units = ["cm/s^2", "cm/s^2", "erg/g"]
    
    xlo = [z_eval[0]]
    xhi = [z_eval[-1]]
    spacing = ["linear"]
    
    # ydata shape: (Nout, Nx[0])
    ydata_dt = np.zeros((nout, nx[0]))
    ydata_dt[0, :] = g_1_sol
    ydata_dt[1, :] = g_ext_sol
    ydata_dt[2, :] = Phi_tot_sol
    
    dt = DataTable(
        ndim=ndim, 
        nx=nx, 
        nout=nout, 
        input_names=input_names, 
        output_names=output_names, 
        input_units=input_units, 
        output_units=output_units, 
        xlo=xlo, 
        xhi=xhi, 
        spacing=spacing, 
        ydata=ydata_dt
    )
    dt.write(f'output/disk_solution_datatable_{args.output_suffix}.csv')

    # 4. Visualize results
    plot_results(df, args.output_suffix)

def plot_results(df, output_suffix):
    """
    Plot the density profiles.
    """
    # Plot 1: Dimensionless Profile (xi vs theta)
    plt.figure(figsize=(8, 6))
    plt.plot(df['theta'], df['xi'], label=r'$\xi = \rho / \rho_0$', color='blue', linewidth=2)
    plt.xlabel(r'$\theta = z / z_*$')
    plt.ylabel(r'$\xi = \rho / \rho_0$')
    plt.xlim(0, 3.1)
    plt.title('Dimensionless Vertical Density Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'output/profile_dimensionless_{output_suffix}.png', dpi=150)
    print("Saved plot to profile_dimensionless.png")
    
    # Plot 2: Physical Profile (rho vs z)
    plt.figure(figsize=(8, 6))
    ngas = df['rho_g_cm3'] / mpcc
    plt.plot(df['z_pc'], ngas, label=r'$\rho(z)$', color='red', linewidth=2)
    plt.xlabel(r'$z$ [pc]')
    plt.ylabel(r'$n_{\rm H}$ [cm$^{-3}$]')
    plt.xlim((0, 300))
    plt.title('Physical Vertical Density Profile')
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.savefig(f'output/profile_physical_{output_suffix}.png', dpi=150)
    print("Saved plot to profile_physical.png")

if __name__ == "__main__":
    main()

