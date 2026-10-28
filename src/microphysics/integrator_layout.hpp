#ifndef MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
#define MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
/// \file integrator_layout.hpp
/// \brief The layout of the integrated vector y: what each entry holds and where. Shared by the integrators and the
/// networks, so it does not include a network.

#include <cstdint>
#include <type_traits>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "microphysics/microphysics_traits.hpp"

// still from extern/Microphysics: NumSpec, net_ienuc, and integrator_neqs (only to check the layout)
#include "integrator_data.H"

namespace quokka::microphysics
{

/// What an entry of the integrated vector y holds. The role decides the entry's tolerances, whether it counts toward
/// the step error, and how clean_state floors it.
enum class VarRole : std::uint8_t { Species, Energy, ChemPhotonNumber, ChemPhotonFlux, ThermalBandEnergy, DustAbsorbedEnergy };

/// Whether entries with this role count toward the step error.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto counts_toward_error(const VarRole role) -> bool
{
	return role == VarRole::Species || role == VarRole::Energy || role == VarRole::ChemPhotonNumber;
}

/// The length of an array member of a burn state: a C array in Microphysics' burn_t, an amrex::GpuArray in BurnState.
/// Transitional: it goes with burn_t, once the primordial problems use BurnState.
template <typename ArrayT> constexpr auto burn_array_size() -> int
{
	if constexpr (std::is_array_v<ArrayT>) {
		return static_cast<int>(std::extent_v<ArrayT>);
	} else {
		return static_cast<int>(ArrayT::size());
	}
}

template <typename problem_t, typename BurnT> struct IntegratorLayout {
	using traits = Microphysics_Traits<problem_t>;

	static constexpr bool photochemistry = traits::photochemistry;

	static constexpr auto NumChemRadEqs() -> int
	{
		if constexpr (photochemistry) {
			return burn_array_size<decltype(BurnT::rn)>();
		} else {
			return 0;
		}
	}

	static constexpr auto NumThermalBands() -> int
	{
		if constexpr (photochemistry) {
			return burn_array_size<decltype(BurnT::re_thermal)>();
		} else {
			return 0;
		}
	}

	static constexpr int num_chem_rad_eqs = NumChemRadEqs();
	static constexpr int num_thermal_bands = NumThermalBands();
	static constexpr int vars_per_group = Microphysics_Traits<problem_t>::integrate_rad_flux ? 2 : 1;
	static constexpr bool q_dust = traits::dust_chemical_band_absorption;
	static constexpr int neqs = NumSpec + 1 + num_chem_rad_eqs + num_thermal_bands + (q_dust ? 1 : 0);

	/// the entry (1-based) of the first thermal band; the others follow it
	static constexpr int thermal_band_row = net_ienuc + num_chem_rad_eqs + 1;
	/// the entry (1-based) of the energy absorbed by dust, with q_dust
	static constexpr int dust_row = net_ienuc + num_chem_rad_eqs + num_thermal_bands + 1;

	/// The role of entry n (1-based) of y: the species, then the energy, then a photon number (and with
	/// integrate_rad_flux a photon flux) per chemical band, then the thermal bands, then the dust slot.
	///
	/// Port-only exception, removed by the post-port cleanup PR: rtol_for and atol_for (rosenbrock.hpp) should switch
	/// over the role, but a switch breaks bitwise identity with Microphysics, which we need during the port but not at
	/// the end. Without photochemistry the switch does not reduce to one select, so numerical_jacobian's ewt loop becomes
	/// too big for Apple clang to unroll fully for PrimordialChem (15 entries). The loop vectorizer then sums fac without
	/// the fused multiply-adds of Microphysics' fully unrolled loop, and the finite-difference Jacobian changes in the
	/// last bits. So they keep Microphysics' shape (energy first, the photon branch only with photochemistry) for now.
	AMREX_GPU_HOST_DEVICE static constexpr auto role(const int n) -> VarRole
	{
		if (n <= NumSpec) {
			return VarRole::Species;
		}
		if (n == net_ienuc) {
			return VarRole::Energy;
		}
		const int r = n - net_ienuc;
		if (r <= num_chem_rad_eqs) {
			return (r - 1) % vars_per_group == 0 ? VarRole::ChemPhotonNumber : VarRole::ChemPhotonFlux;
		}
		if (r <= num_chem_rad_eqs + num_thermal_bands) {
			return VarRole::ThermalBandEnergy;
		}
		return VarRole::DustAbsorbedEnergy;
	}

	static_assert(
	    photochemistry == requires { &BurnT::rn; },
	    "Microphysics_Traits::photochemistry must be true exactly when the problem is built with PHOTOCHEMISTRY");
	static_assert(
	    q_dust == requires { &BurnT::e_dust_absorbed; },
	    "Microphysics_Traits::dust_chemical_band_absorption must be true exactly when the problem is built with DUST_CHEMICAL_BAND_ABSORPTION");
	static_assert(net_ienuc == NumSpec + 1, "the energy must follow the species");
	static_assert(num_chem_rad_eqs % vars_per_group == 0, "every chemical band must have vars_per_group variables");
	static_assert(integrator_neqs<BurnT>() == neqs, "the layout must match Microphysics' integrator_neqs");
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
