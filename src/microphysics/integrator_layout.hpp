#ifndef MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
#define MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
/// \file integrator_layout.hpp
/// \brief The layout of the integrated vector y: what each entry holds and where. Shared by the integrators and the
/// networks, so it does not include a network.

#include <cstdint>

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "microphysics/microphysics_traits.hpp"

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

template <typename problem_t, typename BurnT> struct IntegratorLayout {
	using traits = Microphysics_Traits<problem_t>;

	static constexpr bool photochemistry = traits::photochemistry;

	static constexpr auto NumChemRadEqs() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(decltype(BurnT::rn)::size());
		} else {
			return 0;
		}
	}

	static constexpr auto NumThermalBands() -> int
	{
		if constexpr (photochemistry) {
			return static_cast<int>(decltype(BurnT::re_thermal)::size());
		} else {
			return 0;
		}
	}

	static constexpr int num_spec = static_cast<int>(decltype(BurnT::xn)::size());
	static constexpr int num_chem_rad_eqs = NumChemRadEqs();
	static constexpr int num_thermal_bands = NumThermalBands();
	static constexpr int vars_per_group = Microphysics_Traits<problem_t>::integrate_rad_flux ? 2 : 1;
	static constexpr bool q_dust = traits::dust_chemical_band_absorption;
	static constexpr int neqs = num_spec + 1 + num_chem_rad_eqs + num_thermal_bands + (q_dust ? 1 : 0);

	/// the entry (1-based) of the specific internal energy, right after the species
	static constexpr int energy_row = num_spec + 1;
	/// the entry (1-based) of the first thermal band; the others follow it
	static constexpr int thermal_band_row = energy_row + num_chem_rad_eqs + 1;
	/// the entry (1-based) of the energy absorbed by dust, with q_dust
	static constexpr int dust_row = energy_row + num_chem_rad_eqs + num_thermal_bands + 1;

	/// The role of entry n (1-based) of y: the species, then the energy, then a photon number (and with
	/// integrate_rad_flux a photon flux) per chemical band, then the thermal bands, then the dust slot.
	AMREX_GPU_HOST_DEVICE static constexpr auto role(const int n) -> VarRole
	{
		if (n <= num_spec) {
			return VarRole::Species;
		}
		if (n == energy_row) {
			return VarRole::Energy;
		}
		const int r = n - energy_row;
		if (r <= num_chem_rad_eqs) {
			return (r - 1) % vars_per_group == 0 ? VarRole::ChemPhotonNumber : VarRole::ChemPhotonFlux;
		}
		if (r <= num_chem_rad_eqs + num_thermal_bands) {
			return VarRole::ThermalBandEnergy;
		}
		return VarRole::DustAbsorbedEnergy;
	}

	static_assert(photochemistry == requires { &BurnT::rn; }, "The burn state must have rn exactly when Microphysics_Traits::photochemistry is true");
	static_assert(
	    q_dust == requires { &BurnT::e_dust_absorbed; },
	    "The burn state must have e_dust_absorbed exactly when Microphysics_Traits::dust_chemical_band_absorption is true");
	static_assert(num_chem_rad_eqs % vars_per_group == 0, "every chemical band must have vars_per_group variables");
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_LAYOUT_HPP_
