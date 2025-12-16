#ifndef BC_HPP_
#define BC_HPP_

#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_IntVect.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"
#include "physics_numVars.hpp"
#include "radiation/radiation_system.hpp"
#include <array>

namespace quokka
{

namespace BCType
{
enum mathematicalBndryTypesInteger : int {
	// Standard AMReX boundary conditions (using actual amrex::BCType values, safe from AMReX changes)
	bogus = amrex::BCType::bogus,
	reflect_odd = amrex::BCType::reflect_odd,
	int_dir = amrex::BCType::int_dir, // interior/periodic
	reflect_even = amrex::BCType::reflect_even,
	foextrap = amrex::BCType::foextrap,	// first order extrapolation
	ext_dir = amrex::BCType::ext_dir,	// external Dirichlet
	hoextrap = amrex::BCType::hoextrap,	// higher order extrapolation
	hoextrapcc = amrex::BCType::hoextrapcc, // higher order extrapolation to cell center
	ext_dir_cc = amrex::BCType::ext_dir_cc, // external Dirichlet at cell center
	direction_dependent = amrex::BCType::direction_dependent,
	user_1 = amrex::BCType::user_1,
	user_2 = amrex::BCType::user_2,
	user_3 = amrex::BCType::user_3,

	// Quokka-specific boundary conditions (custom values, not conflicting with AMReX values)
	reflecting = 8881, // Special: uses reflect_odd/reflect_even based on component
			   // outflow_nscbc = 8882, // Future: NSCBC outflow
			   // inflow_nscbc = 8883, // Future: NSCBC inflow
			   // custom_wall = 8884  // Future: custom wall treatment
};

// Quokka boundary conditions
// NOTE: Standard BC values must match amrex::BCType exactly (verified by static_assert below)
// AMREX_ENUM requires literal integers for runtime string parsing to work
AMREX_ENUM(mathematicalBndryTypes,
	   // Standard AMReX boundary conditions (literal values from AMReX_BC_TYPES.H)
	   bogus = -666, reflect_odd = -1,
	   periodic = 0, // corresponds to amrex::BCType::int_dir (interior/periodic); this conversion of int_dir to periodic requires enforcing
			 // geometry.is_periodic in main.cpp
	   reflect_even = 1,
	   foextrap = 2,   // first order extrapolation
	   ext_dir = 3,	   // external Dirichlet
	   hoextrap = 4,   // higher order extrapolation
	   hoextrapcc = 5, // higher order extrapolation to cell center
	   ext_dir_cc = 6, // external Dirichlet at cell center
	   direction_dependent = 7, user_1 = 1001, user_2 = 1002, user_3 = 1003,

	   // Quokka-specific boundary conditions (custom values, not conflicting with AMReX values)
	   reflecting = 8881 // Special: uses reflect_odd/reflect_even based on component
			     // outflow_nscbc = 8882, // Future: NSCBC outflow
			     // inflow_nscbc = 8883, // Future: NSCBC inflow
			     // custom_wall = 8884  // Future: custom wall treatment
);

// Compile-time verification that our values match AMReX's BCType values
static_assert(static_cast<int>(mathematicalBndryTypes::bogus) == amrex::BCType::bogus);
static_assert(static_cast<int>(mathematicalBndryTypes::reflect_odd) == amrex::BCType::reflect_odd);
static_assert(static_cast<int>(mathematicalBndryTypes::periodic) == amrex::BCType::int_dir);
static_assert(static_cast<int>(mathematicalBndryTypes::reflect_even) == amrex::BCType::reflect_even);
static_assert(static_cast<int>(mathematicalBndryTypes::foextrap) == amrex::BCType::foextrap);
static_assert(static_cast<int>(mathematicalBndryTypes::ext_dir) == amrex::BCType::ext_dir);
static_assert(static_cast<int>(mathematicalBndryTypes::hoextrap) == amrex::BCType::hoextrap);
static_assert(static_cast<int>(mathematicalBndryTypes::hoextrapcc) == amrex::BCType::hoextrapcc);
static_assert(static_cast<int>(mathematicalBndryTypes::ext_dir_cc) == amrex::BCType::ext_dir_cc);
static_assert(static_cast<int>(mathematicalBndryTypes::direction_dependent) == amrex::BCType::direction_dependent);
static_assert(static_cast<int>(mathematicalBndryTypes::user_1) == amrex::BCType::user_1);
static_assert(static_cast<int>(mathematicalBndryTypes::user_2) == amrex::BCType::user_2);
static_assert(static_cast<int>(mathematicalBndryTypes::user_3) == amrex::BCType::user_3);
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

// For manual BC setting in problem creator
// Three parameter version - sets each dimension separately
template <typename problem_t> auto BC(int bc_x, int bc_y, int bc_z) -> amrex::Vector<amrex::BCRec>
{
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	std::array<int, 3> bcs = {bc_x, bc_y, bc_z};

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (bcs[i] == BCType::reflecting) {
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

// Three parameter version - sets each dimension separately
template <typename problem_t>
auto BC_cc(BCType::mathematicalBndryTypes bc_x, BCType::mathematicalBndryTypes bc_y, BCType::mathematicalBndryTypes bc_z) -> amrex::Vector<amrex::BCRec>
{
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	std::array<BCType::mathematicalBndryTypes, 3> bcs = {bc_x, bc_y, bc_z};

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (bcs[i] == BCType::mathematicalBndryTypes::reflecting) {
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
				BCs_cc[n].setLo(i, static_cast<int>(bcs[i]));
				BCs_cc[n].setHi(i, static_cast<int>(bcs[i]));
			}
		}
	}

	return BCs_cc;
}

template <typename problem_t>
auto BC_fc(BCType::mathematicalBndryTypes bc_x, BCType::mathematicalBndryTypes bc_y, BCType::mathematicalBndryTypes bc_z) -> amrex::Vector<amrex::BCRec>
{
	const int nvars_fc = Physics_Indices<problem_t>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		std::array<BCType::mathematicalBndryTypes, 3> bcs = {bc_x, bc_y, bc_z};
		for (int icomp = 0; icomp < nvars_fc; ++icomp) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (bcs[idim] == BCType::mathematicalBndryTypes::reflecting) {
					// TODO(cch): Support reflecting B.C. for MHD
					BCs_fc[icomp].setLo(idim, static_cast<int>(BCType::mathematicalBndryTypes::reflect_even));
					BCs_fc[icomp].setHi(idim, static_cast<int>(BCType::mathematicalBndryTypes::reflect_even));
				} else {
					BCs_fc[icomp].setLo(idim, static_cast<int>(bcs[idim]));
					BCs_fc[icomp].setHi(idim, static_cast<int>(bcs[idim]));
				}
			}
		}
	}
	return BCs_fc;
}

} // namespace quokka

#endif // BC_HPP_
