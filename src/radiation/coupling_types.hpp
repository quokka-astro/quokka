#ifndef COUPLING_TYPES_HPP_
#define COUPLING_TYPES_HPP_

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Array.H"
#include "util/valarray.hpp"
#include <array>

enum class ChemicalBandRole { PE, HI_ion, HeI_ion, HeII_ion };

/// Default Chemistry_Traits: no chemical bands.
/// Problems with chemical bands specialize this struct.
template <typename problem_t> struct Chemistry_Traits {
	static constexpr int nChemicalGroups = 0;
	static constexpr std::array<ChemicalBandRole, 0> chemical_band_roles = {};
};

namespace detail
{
/// Find the global group index of a chemical band with the given role.
/// Returns -1 if not found.
template <typename problem_t> constexpr auto FindChemicalBand(ChemicalBandRole role, int nThermalGroups) -> int
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		if (roles[i] == role) {
			return nThermalGroups + i;
		}
	}
	return -1;
}

/// Count occurrences of a chemical band role.
template <typename problem_t> constexpr auto CountChemicalBand(ChemicalBandRole role) -> int
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	int count = 0;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		if (roles[i] == role) {
			++count;
		}
	}
	return count;
}

/// Check that all chemical band roles are unique.
template <typename problem_t> constexpr auto AllUniqueRoles() -> bool
{
	constexpr auto &roles = Chemistry_Traits<problem_t>::chemical_band_roles;
	for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
		for (int j = i + 1; j < Chemistry_Traits<problem_t>::nChemicalGroups; ++j) {
			if (roles[i] == roles[j]) {
				return false;
			}
		}
	}
	return true;
}
} // namespace detail

/// Per-cell coupling state loaded from consVar at the top of AddSourceTerms.
/// GPU-copyable (no pointers, no virtual functions).
template <int nGroups, int nmscalars> struct CouplingState {
	double rho;
	double Egas0;
	double Ekin0;
	amrex::GpuArray<double, 3> gasMomentum0;
	quokka::valarray<double, nGroups> Erad0;
	quokka::valarray<double, nGroups> Src;
	amrex::GpuArray<double, 3 * nGroups> Frad0_flat; // flattened [dim][nGroups] as [dim * nGroups]
	amrex::GpuArray<amrex::Real, nmscalars> massScalars;
	double dt;
	double Etot0;
};

enum class DustModel { gas_only, coupled, decoupled };

/// Solver control parameters. Not per-cell.
struct SolverParams {
	double resid_tol;
	double rel_change_tol;
	int max_newton_iter;
	int max_outer_iter;
};

/// Structured per-cell debug output. Compiled away when debug_mode = false.
template <bool enabled, int nGroups> struct DiagnosticTrace {
};

template <int nGroups> struct DiagnosticTrace<true, nGroups> {
	static constexpr int max_recorded_iters = 20;
	int n_recorded = 0;
	struct IterationSnapshot {
		double Egas;
		double T_gas;
		double T_d;
		quokka::valarray<double, nGroups> Rvec;
		quokka::valarray<double, nGroups> Erad;
		double F0;
		double Fg_abs_sum;
		double damping_factor;
	};
	amrex::GpuArray<IterationSnapshot, max_recorded_iters> snapshots;
};

#endif // COUPLING_TYPES_HPP_
