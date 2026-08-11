#ifndef PHYSICS_INFO_HPP_ // NOLINT
#define PHYSICS_INFO_HPP_

#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "physics_numVars.hpp"
#include <AMReX.H>
#include <array>

using Real = amrex::Real;

// enum for unit system, one of CGS, CONSTANTS, CUSTOM
enum class UnitSystem { CGS, CONSTANTS, CUSTOM };

// enum for MHD resistivity model
enum class ResistivityModel {
	none,		 // no physical resistivity model
	constant,	 // uniform Ohmic resistivity; eta read from mhd.resistivity in the TOML input file
	problem_defined, // per-cell Ohmic resistivity; eta returned by a problem-specific computeResistivity device function
};

// enum for hydro (shear + bulk) viscosity model
enum class ViscosityModel {
	none,		 // no physical viscosity model
	constant,	 // uniform shear/bulk viscosity; read from hydro.shear_viscosity / hydro.bulk_viscosity in the TOML input file
	problem_defined, // per-cell viscosity; returned by a problem-specific computeViscosity device function
};

// default values for all Physics_Traits fields; specialize Physics_Traits by inheriting from this
// struct and overriding only the fields that differ from the defaults
struct DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;
	// NOTE: numPassiveScalars is evaluated at the point of definition of DefaultPhysicsTraits, not
	// at the point of specialization. If you override numMassScalars, you MUST also explicitly
	// override numPassiveScalars, or it will silently inherit the pre-evaluated default of 0.
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr ResistivityModel resistivity_model = ResistivityModel::none;
	static constexpr ViscosityModel viscosity_model = ViscosityModel::none;
	static constexpr int nGroups = 1;     // number of radiation groups
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;	    // Hydro, EOS
	static constexpr double gravitational_constant = C::Gconst; // gravity
	static constexpr double c_light = C::c_light;		    // radiation
	static constexpr double radiation_constant = C::a_rad;	    // radiation
	static constexpr double unit_length = 1.0;
	static constexpr double unit_mass = 1.0;
	static constexpr double unit_time = 1.0;
	static constexpr double unit_temperature = 1.0;
};

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits : DefaultPhysicsTraits {
};

// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
	// number of cc quantities required for advection problems
	static constexpr int nvarTotal_cc_adv = 1;
	// number of cc quantities required for rad /+ hydro problem
	static constexpr int nvarTotal_cc = []() constexpr {
		if constexpr (!(Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled)) {
			return nvarTotal_cc_adv;
		}
		return Physics_Traits<problem_t>::numPassiveScalars + Physics_NumVars::numHydroVars +
		       Physics_NumVars::numDustVarsPerGroup * Physics_Traits<problem_t>::nDustGroups *
			   static_cast<int>(Physics_Traits<problem_t>::is_dust_enabled) +
		       Physics_NumVars::numRadVarsPerGroup * Physics_Traits<problem_t>::nGroups *
			   static_cast<int>(Physics_Traits<problem_t>::is_radiation_enabled);
	}();
	// cell-centered
	static constexpr int hydroFirstIndex = 0;
	static constexpr int pscalarFirstIndex = Physics_NumVars::numHydroVars;
	static constexpr int dustFirstIndex = pscalarFirstIndex + Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int radFirstIndex = dustFirstIndex + Physics_NumVars::numDustVarsPerGroup * Physics_Traits<problem_t>::nDustGroups *
								  static_cast<int>(Physics_Traits<problem_t>::is_dust_enabled);
	// face-centered
	static constexpr int nvarPerDim_fc = Physics_NumVars::numMHDVars_per_dim * static_cast<int>(Physics_Traits<problem_t>::is_mhd_enabled);
	static constexpr int nvarTotal_fc = AMREX_SPACEDIM * nvarPerDim_fc;
	static constexpr int mhdFirstIndex = 0;
};

// Compute cell-centered magnetic energy density (0.5 * B^2) from face-centered field data.
// Defined here (rather than inline in a ParallelFor lambda) to work around an NVCC limitation
// that disallows first-capturing variables in constexpr-if contexts inside extended device lambdas.
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto ComputeCellCenteredMagneticEnergy(int i, int j, int k,
									   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &fc) -> double
{
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		constexpr int mhdIdx = Physics_Indices<problem_t>::mhdFirstIndex;
		const amrex::Real bx = 0.5 * (fc[0](i, j, k, mhdIdx) + fc[0](i + 1, j, k, mhdIdx));
#if (AMREX_SPACEDIM >= 2)
		const amrex::Real by = 0.5 * (fc[1](i, j, k, mhdIdx) + fc[1](i, j + 1, k, mhdIdx));
#endif
#if (AMREX_SPACEDIM == 3)
		const amrex::Real bz = 0.5 * (fc[2](i, j, k, mhdIdx) + fc[2](i, j, k + 1, mhdIdx));
#endif
		return 0.5 * (AMREX_D_TERM(bx * bx, +by * by, +bz * bz));
	}
	return 0.0;
}

#endif // PHYSICS_INFO_HPP_
