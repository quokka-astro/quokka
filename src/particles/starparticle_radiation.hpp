#ifndef STAR_PARTICLE_DATA_HPP_
#define STAR_PARTICLE_DATA_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include <cmath>

namespace quokka
{

// Constants for stellar astrophysics

namespace StellarConstants
{
    // Solar properties
    static constexpr amrex::Real M_solar = 1.99e33;
    static constexpr amrex::Real L_solar = 3.90e33;
    static constexpr amrex::Real R_solar = 6.96e10;
  //    static constexpr amrex::Real seconds_per_year = 3.15576e7;

    // Physical constants
    static constexpr amrex::Real G = 6.67e-8;
    static constexpr amrex::Real pi = 3.14159265358979323846;
    static constexpr amrex::Real a_rad = 7.56e-15;        // radiation constant
    static constexpr amrex::Real k_B = 1.38e-16;          // Boltzmann constant
    static constexpr amrex::Real m_H = 1.67e-24;          // hydrogen mass
    static constexpr amrex::Real sigma_SB = 5.67e-5;      // Stefan-Boltzmann constant
    static constexpr amrex::Real mu = 0.613;              // mean molecular weight

    // Model parameters
    static constexpr amrex::Real F_acc = 0.5;            // fraction of accreted energy radiated
    static constexpr amrex::Real F_k = 0.5;              // fraction of energy from inner disk
    static constexpr amrex::Real F_rad = 0.33;           // radiative barrier parameter
    static constexpr amrex::Real shell_factor = 2.1;     // radius increase for shell burning
    static constexpr amrex::Real T_Hayashi = 3000.0;     // Hayashi temperature
    static constexpr amrex::Real T_deuterium = 1.5e6;    // deuterium ignition temperature
    static constexpr amrex::Real M_rad_min = 0.01 * M_solar; // minimum mass for model
    static constexpr amrex::Real PSIION = 16.0 * ERGEV * NAVOG) // Energy per gram needed to dissociate and ionize a molecular gas with solar abundances. ERGEV=1.6e-12 number pf ergs per eV and NAVOG=6.022e23
    static constexpr amrex::Real PSID = 100 * ERGEV * NAVOG //Energy per gram released by burning the deuterium in a gas with solar abundances
}

// Central density/pressure tables (GPU-compatible static data)

namespace StellarTables
{
    // rho_mean/rho_c table for n=1.5 to 3.0 (step 0.1)
    static constexpr int n_rho_table = 17;
    AMREX_GPU_CONSTANT constexpr amrex::Real rho_factor_table[n_rho_table] = {
        0.166931, 0.14742, 0.129933, 0.114265, 0.100242,
        0.0877, 0.0764968, 0.0665109, 0.0576198, 0.0497216,
        0.0427224, 0.0365357, 0.0310837, 0.0262952, 0.0221057,
        0.0184553, 0.01529
    };

    // P_c/(G M^2/R^4) table for n=1.5 to 3.0 (step 0.1)
    AMREX_GPU_CONSTANT constexpr amrex::Real pressure_factor_table[n_rho_table] = {
        0.770087, 0.889001, 1.02979, 1.19731, 1.39753,
        1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921,
        4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885
    };

    // Beta table for interpolation (M: 5-50 Msun, n: 1.5-3.0)
    static constexpr int n_beta_mass = 19;
    static constexpr int n_beta_poly = 4;
    AMREX_GPU_CONSTANT constexpr amrex::Real beta_table[n_beta_mass][n_beta_poly] = {
        {0.98785, 0.988928, 0.98947, 0.989634},
        {0.97438, 0.976428, 0.977462, 0.977774},
        {0.957927, 0.960895, 0.962397, 0.962846},
        {0.939787, 0.943497, 0.945369, 0.945922},
        {0.92091, 0.925151, 0.927276, 0.927896},
        {0.901932, 0.906512, 0.908785, 0.909436},
        {0.883254, 0.888017, 0.890353, 0.891013},
        {0.865111, 0.86994, 0.872277, 0.872927},
        {0.847635, 0.852445, 0.854739, 0.855367},
        {0.830886, 0.835619, 0.837842, 0.838441},
        {0.814885, 0.8195, 0.821635, 0.822201},
        {0.799625, 0.804095, 0.806133, 0.806664},
        {0.785082, 0.789394, 0.791328, 0.791825},
        {0.771226, 0.775371, 0.777202, 0.777665},
        {0.758022, 0.761997, 0.763726, 0.764156},
        {0.745433, 0.749238, 0.750869, 0.751268},
        {0.733423, 0.73706, 0.738596, 0.738966},
        {0.721954, 0.725429, 0.726874, 0.727216},
        {0.710993, 0.714311, 0.715671, 0.715987}
    };

    // Tout et al. (1996) main sequence fitting parameters
    namespace Tout96
    {
        static constexpr amrex::Real alpha = 0.39704170;
        static constexpr amrex::Real beta = 8.52762600;
        static constexpr amrex::Real gamma = 0.00025546;
        static constexpr amrex::Real delta = 5.43288900;
        static constexpr amrex::Real epsilon = 5.56357900;
        static constexpr amrex::Real zeta = 0.78866060;
        static constexpr amrex::Real eta = 0.00586685;        
        static constexpr amrex::Real theta = 1.71535900;
        static constexpr amrex::Real iota = 6.59778800;
        static constexpr amrex::Real kappa = 10.08855000;
        static constexpr amrex::Real lambda = 1.01249500;
        static constexpr amrex::Real gmu = 0.07490166;
        static constexpr amrex::Real nu = 0.01077422;
        static constexpr amrex::Real xi = 3.08223400;
        static constexpr amrex::Real upsilon = 17.84778000;
        static constexpr amrex::Real gpi = 0.00022582;
    }
}

// Utility functions for stellar physics

namespace StellarPhysics
{
    // Burning state enum
    enum class BurningState : int {
        Uninitialized = 0,
        None,
        VariableCoreDeuterium,
        SteadyCoreDeuterium,
        ShellDeuterium,
        ZAMS
    };

    // Bisection solver for GPU
    template<typename Func>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto bisection_solve(Func f, amrex::Real x1, amrex::Real x2, 
                                                              int max_iter = 40, amrex::Real tol = 1.0e-7) -> amrex::Real
    {
        amrex::Real f1 = f(x1);
        amrex::Real f2 = f(x2);
        
        if (f1 * f2 > 0) {
            return x1; // No root in interval
        }
        
        amrex::Real rtb = (f1 < 0) ? x1 : x2;
        amrex::Real dx = (f1 < 0) ? (x2 - x1) : (x1 - x2);
        
        for (int j = 0; j < max_iter; ++j) {
            dx *= 0.5;
            amrex::Real xmid = rtb + dx;
            amrex::Real fmid = f(xmid);
            
            if (fmid <= 0) {
                rtb = xmid;
            }
            if (std::abs(dx) < tol * std::abs(xmid) || fmid == 0) {
                return rtb;
            }
        }
        return rtb; // Return best estimate
    }

    // Initialize polytropic index from accretion rate
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto n_init(amrex::Real mdot) -> amrex::Real
    {
        using namespace StellarConstants;
        
        if (mdot == 0.0) {
            return 1.5;
        }
        
        amrex::Real log_term = std::log10(mdot * seconds_per_year / M_solar);
        amrex::Real aG_init = 1.475 + 0.07 * log_term;
        amrex::Real n_val = 5.0 - 3.0 / aG_init;
        
        // Clamp to valid range
        n_val = std::max(1.5, std::min(3.0, n_val));
        return n_val;
    }

    // Initialize stellar radius from accretion rate
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto rad_init(amrex::Real mdot) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real mdot_norm = mdot * seconds_per_year / M_solar * 1.0e5;
        amrex::Real rad_factor = 2.5 * std::pow(mdot_norm, 0.2);
        rad_factor = std::max(rad_factor, 2.0); // Minimum factor of 2
        
        return R_solar * rad_factor;
    }

    // Polytropic gravitational energy parameter
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto aG(amrex::Real n) -> amrex::Real
    {
        return 3.0 / (5.0 - n);
    }

    // Interpolate from central density table
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto rho_factor_interp(amrex::Real n) -> amrex::Real
    {
        using namespace StellarTables;
        
        // n ranges from 1.5 to 3.0, table index 0-16
        amrex::Real idx_real = (n - 1.5) / 0.1;
        int idx = static_cast<int>(std::floor(idx_real));
        amrex::Real weight = idx_real - idx;
        
        // Clamp indices to table bounds
        idx = std::max(0, std::min(n_rho_table - 2, idx));
        
        return rho_factor_table[idx] * (1.0 - weight) + rho_factor_table[idx + 1] * weight;
    }

    // Interpolate from central pressure table
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto pressure_factor_interp(amrex::Real n) -> amrex::Real
    {
        using namespace StellarTables;
        
        amrex::Real idx_real = (n - 1.5) / 0.1;
        int idx = static_cast<int>(std::floor(idx_real));
        amrex::Real weight = idx_real - idx;
        
        idx = std::max(0, std::min(n_rho_table - 2, idx));
        
        return pressure_factor_table[idx] * (1.0 - weight) + pressure_factor_table[idx + 1] * weight;
    }

    // Central density of polytropic star
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto rho_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real volume = (4.0 / 3.0) * pi * radius * radius * radius;
        amrex::Real rho_mean = mass / volume;
        amrex::Real rho_factor = rho_factor_interp(n);
        
        return rho_mean / rho_factor;
    }

    // Central pressure of polytropic star
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto pressure_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real p_factor = pressure_factor_interp(n);
        return p_factor * G * mass * mass / (radius * radius * radius * radius);
    }

    // Central temperature from equation of state
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto temperature_central(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real rho_c = rho_central(mass, radius, n);
        amrex::Real P_c = pressure_central(mass, radius, n);
        
        // Gas temperature if radiation pressure negligible
        amrex::Real T_gas = P_c * mu * m_H / (k_B * rho_c);
        
        // Radiation temperature if gas pressure negligible
        amrex::Real T_rad = std::pow(3.0 * P_c / a_rad, 0.25);
        
        // Use bisection to solve full EOS: P = ρkT/μm_H + aT⁴/3
        auto pressure_func = [=](amrex::Real T) {
            return P_c - rho_c * k_B * T / (mu * m_H) - (a_rad * std::pow(T, 4)) / 3.0;
        };
        
        // Initial bounds
        amrex::Real T_low = 0.0;
        amrex::Real T_high = (T_rad > T_gas) ? 2.0 * T_rad : 2.0 * T_gas;
        
        return bisection_solve(pressure_func, T_low, T_high);
    }

    // Beta parameter (gas pressure / total pressure) for n=3 (Eddington quartic)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_n3(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real rho_c = rho_central(mass, radius, n);
        amrex::Real P_c = pressure_central(mass, radius, n);
        
        amrex::Real coefficient = 3.0 / a_rad * std::pow(k_B * rho_c / (mu * m_H), 4);
        
        auto beta_func = [=](amrex::Real beta) {
            return std::pow(P_c, 3) - coefficient * (1.0 - beta) / std::pow(beta, 4);
        };
        
        return bisection_solve(beta_func, 1.0e-4, 1.0);
    }

    // Beta parameter from interpolation table
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_table_interp(amrex::Real mass, amrex::Real n) -> amrex::Real
    {
        using namespace StellarConstants;
        using namespace StellarTables;
        
        // Table parameters
        constexpr amrex::Real M_min = 5.0 * M_solar;
        constexpr amrex::Real M_max = 50.0 * M_solar;
        constexpr amrex::Real M_step = 2.5 * M_solar;
        constexpr amrex::Real n_min = 1.5;
        constexpr amrex::Real n_max = 3.0;
        constexpr amrex::Real n_step = 0.5;
        
        // Clamp to table bounds
        if (mass < M_min) {
            return 1.0; // Fully gas pressure supported for low mass
        }
        if (mass >= M_max || n >= n_max) {
            return -1.0; // Out of table bounds
        }
        
        // Mass index
        amrex::Real m_idx_real = (mass - M_min) / M_step;
        int m_idx = static_cast<int>(std::floor(m_idx_real));
        amrex::Real m_weight = m_idx_real - m_idx;
        
        // Polytropic index
        amrex::Real n_idx_real = (n - n_min) / n_step;
        int n_idx = static_cast<int>(std::floor(n_idx_real));
        amrex::Real n_weight = n_idx_real - n_idx;
        
        // Clamp indices
        m_idx = std::max(0, std::min(n_beta_mass - 2, m_idx));
        n_idx = std::max(0, std::min(n_beta_poly - 2, n_idx));
        
        // Bilinear interpolation
        amrex::Real beta00 = beta_table[m_idx][n_idx];
        amrex::Real beta10 = beta_table[m_idx + 1][n_idx];
        amrex::Real beta01 = beta_table[m_idx][n_idx + 1];
        amrex::Real beta11 = beta_table[m_idx + 1][n_idx + 1];
        
        return beta00 * (1.0 - m_weight) * (1.0 - n_weight) +
               beta10 * m_weight * (1.0 - n_weight) +
               beta01 * (1.0 - m_weight) * n_weight +
               beta11 * m_weight * n_weight;
    }

    // Unified beta function
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto beta_total(amrex::Real mass, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
        if (n == 3.0) {
            return beta_n3(mass, radius, n);
        } else {
            return beta_table_interp(mass, n);
        }
    }

    // ZAMS luminosity from Tout et al. (1996)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto luminosity_ZAMS(amrex::Real mass) -> amrex::Real
    {
        using namespace StellarConstants;
        using namespace StellarTables::Tout96;
        
        amrex::Real m_sol = mass / M_solar;
        
        amrex::Real numerator = alpha * std::pow(m_sol, 5.5) + beta * std::pow(m_sol, 11.0);
        amrex::Real denominator = gamma + std::pow(m_sol, 3.0) + delta * std::pow(m_sol, 5.0) +
                                 epsilon * std::pow(m_sol, 7.0) + zeta * std::pow(m_sol, 8.0) +
                                 eta * std::pow(m_sol, 9.5);
        
        return (numerator / denominator) * L_solar;
    }

    // ZAMS radius from Tout et al. (1996)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto radius_ZAMS(amrex::Real mass) -> amrex::Real
    {
        using namespace StellarConstants;
        using namespace StellarTables::Tout96;
        
        amrex::Real m_sol = mass / M_solar;
        
        amrex::Real numerator = theta * std::pow(m_sol, 2.5) + iota * std::pow(m_sol, 6.5) +
                               kappa * std::pow(m_sol, 11.0) + lambda * std::pow(m_sol, 19.0) +
                               gmu * std::pow(m_sol, 19.5);
        amrex::Real denominator = nu + xi * std::pow(m_sol, 2.0) + upsilon * std::pow(m_sol, 8.5) +
                                 std::pow(m_sol, 18.5) + gpi * std::pow(m_sol, 19.5);
        
        return (numerator / denominator) * R_solar;
    }

    // Stellar luminosity (ZAMS + accretion)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto luminosity_star(amrex::Real mass, amrex::Real radius, amrex::Real mdot) -> amrex::Real
    {
        using namespace StellarConstants;
        
        amrex::Real L_zams = luminosity_ZAMS(mass);
        amrex::Real L_acc = F_acc * F_k * G * mass * mdot / radius;
        amrex::Real L_total = L_zams + L_acc;
        
        // Hayashi limit check
        amrex::Real T_eff = std::pow(L_total / (4.0 * pi * radius * radius * sigma_SB), 0.25);
        if (T_eff > T_Hayashi) {
            return L_total;
        } else {
            return 4.0 * pi * radius * radius * sigma_SB * std::pow(T_Hayashi, 4);
        }
    }

    // numerical derivative dlogBeta_dlogM
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dlogBeta_dlogM(amrex::mass, amrex::Real beta_1, amrex::Real radius, amrex::Real n) -> amrex::Real
    {
      Real beta1;
      if (beta_1==-1.0)
	{
	  beta1 = beta_total(mass, radius, n);
	}else
	{
	  beta1 = beta_1;
	}
      amrex::Real beta2 = beta_total(1.01*mass, radius, n);
      return( mass/beta1 * (beta2-beta1) / (1.01*mass));
}

  // eDotIon
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto eDotIon()(amrex:: mdot, amrex::PSIION)
  {
    return(mdot * PSIION);
  }

  // dlogBetaOverBetac_dlogM
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dlogBetaOverBetac_dlogM(amrex::mass, amrex::beta_1, amrex::Real radius, amrex::Real n)
  {  
    if (n==3) return(0.0);

    // Otherwise take a numerical derivative
    Real beta1;
    if (beta_1==-1.0) beta1 = beta_total(mass, radius, n);
    else beta1 = beta_1;
    Real beta2 = beta_total(1.01*mass, radius, n);
    Real betac1 = betac(mass);
    Real betac2 = betac(1.01*mass);
    return( mass/(beta1/betac1) * ((beta2/betac2) - (beta1/betac1)) / (0.01*mass) );
}

  // Deuterium luminosity
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto L_Deut(amrex::Real mass, amrex::Real radius,amrex::Real n, BurningState burn_state) -> amrex::Real
      {
	if (burn_state == BurningState::Uninitialized || burn_state == BurningState::None) {
	  return(0.0);
	} else if (burn_state == BurningState::VariableCoreDeuterium) {
	  if (beta1 == -1.0) beta1=beta_total(m, radius, n);
	  return( luminosity_star(mass,radius,mdot) + eDotIon() + G*mass*mdot/radius * (1.0 - F_k - aG(n)*beta1/2.0 * (1.0 + dlogBetaOverBetac_dlogM(mass,beta1,radius,n))) );
	} else if (burn_state ==	BurningState::SteadyCoreDeuterium) {
	  return( mdot * PSID );
	} else if (burn_state == BurningState::ShellDeuterium) {
	  return( mdot * PSID );
      }
	
    // Disk luminosity
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto luminosity_disk(amrex::Real mass, amrex::Real radius, amrex::Real mdot) -> amrex::Real
    {
        using namespace StellarConstants;
        return (1.0 - F_k) * G * mass * mdot / radius;
    }

    // Total luminosity
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto luminosity_total(amrex::Real mass, amrex::Real radius, amrex::Real mdot,
                                                               BurningState burn_state) -> amrex::Real
    {
        if (burn_state == BurningState::Uninitialized) {
            return 0.0;
        }
        
        amrex::Real L_star = luminosity_star(mass, radius, mdot);
        amrex::Real L_disk = luminosity_disk(mass, radius, mdot);
        
        return L_star + L_disk;
    }
}

// Main update function for stellar particles

class StellarUpdate
{
  public:
     template <typename problem_t, typename ParticleType, int Nout,
     quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateStellarProperties(
 	  ParticleType &p, amrex::Real current_time, Real dt,
          LuminosityGpuConstTables<Nout, oob_policy> const &gpu_tables) noexcept
    {	  
        // Get dt from particle data or compute it
        constexpr int dt_idx = StarParticleDtIdx; // Define this constant
        
        // Call the internal update function
        updateStellarPropertiesImpl(p, dt);
    }

private:
    template <typename ParticleType>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void 
    updateStellarPropertiesImpl(ParticleType &p, amrex::Real dt) noexcept
    {
        using namespace StellarConstants;
        using namespace StellarPhysics;
        
        // Particle data indices (these should match your particle type definition)
        constexpr int mass_idx = 0;          // particle mass
        constexpr int mlast_idx = 1;         // previous mass
        constexpr int mdeut_idx = 2;         // deuterium mass
        constexpr int mdot_idx = 3;          // accretion rate
        constexpr int radius_idx = 4;        // stellar radius
        constexpr int n_idx = 5;             // polytropic index
        constexpr int burn_state_idx = 6;    // burning state (as int)
        
        // Get current values
        amrex::Real mass = p.rdata(mass_idx);
        amrex::Real mlast = p.rdata(mlast_idx);
        amrex::Real mdeut = p.rdata(mdeut_idx);
        amrex::Real mdot = p.rdata(mdot_idx);
        amrex::Real radius = p.rdata(radius_idx);
        amrex::Real n = p.rdata(n_idx);
        auto burn_state = static_cast<BurningState>(static_cast<int>(p.rdata(burn_state_idx)));
        
        // Update accretion rate (simple difference)
        mdot = (mass - mlast) / dt;
        
        // Update deuterium mass
        mdeut += mass - mlast;
        
        // Initialize if needed
        if (burn_state == BurningState::Uninitialized) {
            if (mass < M_rad_min || mdot == 0.0) {
                p.rdata(mlast_idx) = mass;
                return;
            }
            
            n = n_init(mdot);
            radius = rad_init(mdot);
            burn_state = BurningState::None;
            
            p.rdata(n_idx) = n;
            p.rdata(radius_idx) = radius;
            p.rdata(burn_state_idx) = static_cast<amrex::Real>(static_cast<int>(burn_state));
        }

	// Update radius
	if (burnState != ZAMS) {
	  Real beta1 = beta_total(m, radius, n);
	  Real dr = (2.0*mdot/mass*radius*(F_k/(aG(n)*beta1)+1.0-1.0/(aG(n)*beta1))
		     + beta1/mass * dlogBeta_dlogM(mass,beta1,radius,n) * mdot * radius / beta1
		     - 2.0/(beta1*aG(n))*radius*radius/(G*mass*mass)*(luminosity_star(mass,radius,mdot)
		    +eDotIon()-L_Deut(mass,beta1,radius,n)));
	  Real rdottime = fabs(radius/dr)/100.0;
	  Real mdottime = fabs(mass/mdot)/100.0;
    
	  if( rdottime < dt)
	    {
	      int rdotfac = ceil(dt/rdottime);
	      Real rdotfacr = rdotfac;
	      Real dtprime = dt/rdotfac;

	      for(int rdotloop = 0; rdotloop < rdotfac; rdotloop++)
		{
		  beta1 = beta_total(mass,radius,n);
		  dr = (2.0*mdot/mass*radius*(F_k/(aG(n)*beta1)+1.0-1.0/(aG(n)*beta1))
		      + beta1/mass * dlogBeta_dlogM(mass,beta1,radius,n) * mdot * radius / beta1
		       - 2.0/(beta1*aG(n))*radius*radius/(G*mass*mass)*(luminosity_star(mass,radius,mdot)+eDotIon()
		       -L_Deut(mass,beta1,radius,n)));
		  radius += dtprime * dr;
		}
	      
	    }else if( mdottime < dt )
	    {
	      int mdotfac = ceil(dt/mdottime);
	      Real mdotfacr = mdotfac;
	      Real dtprime = dt/mdotfacr;
	      for(int mdotloop = 0; mdotloop < mdotfac; mdotloop++)
		{
		  beta1=beta_total(m,radius,n);
		  dr = (2.0*mdot/mass*radius*(F_k/(aG(n)*beta1)+1.0-1.0/(aG(n)*beta1))
		      + beta1/mass * dlogBeta_dlogM(mass,beta1,radius,n) * mdot * radius / beta1
		       - 2.0/(beta1*aG(n))*radius*radius/(G*mass*mass)*(luminosity_star(mass,radius,mdot)
		      + eDotIon() - lDeut(mass,beta1,radius,n)));
		  radius += dtprime * dr;
		}
	    } else
	    {
	      beta1=beta_total(mass,radius,n);
	      dr = (2.0*mdot/mass*radius*(F_k/(aG(n)*beta1)+1.0-1.0/(aG(n)*beta1))
		    + beta1/mass * dlogBeta_dlogM(mass,beta1,radius,n) * mdot * radius / beta1
		    - 2.0/(beta1*aG(n))*radius*radius/(G*mass*mass)*(luminosity_star(mass,radius,mdot)
		    +eDotIon() - lDeut(mass,beta1,radius,n)));
	      radius += dt * dr;
	    }
	  // Resetting to 0.2 R_sun, if r is -ve.
	  if(r < 0.0e0)
	    {
	      radius = 0.2*6.96e10; //Worst case and we do get a neg radius. reset it
	    }
	}

        // Update burning state
        if (burn_state == BurningState::None) {
                n = n_init(mdot);
                if (temperature_central(mass, radius, n) > T_deuterium) {
                    burn_state = BurningState::VariableCoreDeuterium;
                    n = 1.5;
                }
	} else if (burn_state == BurningState::VariableCoreDeuterium) {
                // Deuterium burning reduces deuterium mass
                // (simplified for example)
                mdeut -= 0.1 * mdot * dt;
                if (mdeut <= mdot * dt) {
                    burn_state = BurningState::SteadyCoreDeuterium;
                    mdeut = 0.0;
                }
	} else if (burn_state == BurningState::SteadyCoreDeuterium) {
                mdeut = 0.0;
                // Check for radiative barrier
                // (simplified condition for example)
                if (luminosity_star(mass, radius, mdot) < F_rad * luminosity_ZAMS(mass)) {
                    burn_state = BurningState::ShellDeuterium;
                    n = 3.0;
                    radius *= shell_factor;
                }
	} else if (burn_state == BurningState::ShellDeuterium) {
                mdeut = 0.0;
                if (radius <= radius_ZAMS(mass)) {
                    burn_state = BurningState::ZAMS;
                    radius = radius_ZAMS(mass);
                }
	} else if (burn_state == BurningState::ZAMS) {
                mdeut = 0.0;
        }

        // Update particle data
        p.rdata(mlast_idx) = mass;
        p.rdata(mdeut_idx) = mdeut;
        p.rdata(mdot_idx) = mdot;
        p.rdata(radius_idx) = radius;
        p.rdata(n_idx) = n;
        p.rdata(burn_state_idx) = static_cast<amrex::Real>(static_cast<int>(burn_state));
    }
};

} // namespace quokka

#endif // STAR_PARTICLE_DATA_HPP_

