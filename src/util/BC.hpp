#ifndef BC_HPP_
#define BC_HPP_

#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"
#include "physics_numVars.hpp"
#include "radiation/radiation_system.hpp"
#include <array>

namespace quokka
{

namespace BCType
{
// Standard AMReX boundary conditions (using actual amrex::BCType values, safe from AMReX changes)
AMREX_ENUM(mathematicalBndryTypes, bogus, reflect_odd,
	   int_dir, // interior/periodic
	   reflect_even,
	   foextrap,   // first order extrapolation
	   ext_dir,    // external Dirichlet
	   hoextrap,   // higher order extrapolation
	   hoextrapcc, // higher order extrapolation to cell center
	   ext_dir_cc, // external Dirichlet at cell center
	   direction_dependent, user_1, user_2, user_3,

	   // Quokka-specific boundary conditions (custom values, not conflicting with AMReX values)
	   reflecting, // Special: uses reflect_odd/reflect_even based on component
	   outflow_nscbc, inflow_nscbc, custom_wall);
} // namespace BCType

namespace detail
{

// Check if a component is a normal component (momentum or radiation flux) in a given dimension
template <typename problem_t> constexpr auto isNormalComponent(int n, int dim) -> bool
{
	// Check radiation flux components if radiation is enabled
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		// Check gas momentum components in RadSystem
		if ((n == RadSystem<problem_t>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<problem_t>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<problem_t>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}

		// Check radiation flux components for all groups
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			if ((n == RadSystem<problem_t>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 0)) {
				return true;
			}
			if ((n == RadSystem<problem_t>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 1)) {
				return true;
			}
			if ((n == RadSystem<problem_t>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) && (dim == 2)) {
				return true;
			}
		}
	} else {
		// Check hydro momentum components
		if ((n == HydroSystem<problem_t>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<problem_t>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<problem_t>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
	}

	return false;
}
} // namespace detail

// Three parameter version - sets each dimension separately
template <typename problem_t> auto BC(int bc_x, int bc_y, int bc_z) -> amrex::Vector<amrex::BCRec>
{
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	std::array<int, 3> bcs = {bc_x, bc_y, bc_z};

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (bcs[i] == static_cast<int>(BCType::mathematicalBndryTypes::reflecting)) {
				// For reflecting boundaries, use reflect_odd for normal momentum components
				// and reflect_even for all other components (including tangential momentum)
				if (detail::isNormalComponent<problem_t>(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			} else {
				// For standard AMReX boundaries, use the BC type directly
				// (quokka::BCType values map directly to amrex::BCType values)
				BCs_cc[n].setLo(i, bcs[i]);
				BCs_cc[n].setHi(i, bcs[i]);
			}
		}
	}

	return BCs_cc;
}

// Overloads for BCType enum - provides clean, type-safe syntax
template <typename problem_t> auto BC(int bc) -> amrex::Vector<amrex::BCRec> { return BC<problem_t>(bc, bc, bc); }

// Note: BCType values implicitly convert to int, so we can use them directly with the int version
// Usage: BC<Problem>(BCType::reflecting) or BC<Problem>(BCType::ext_dir, BCType::int_dir, BCType::reflecting)

} // namespace quokka

#endif // BC_HPP_
