// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_BURNER_HPP_
#define MICROPHYSICS_BURNER_HPP_
/// \file burner.hpp
/// \brief Burn<problem_t>, which integrates a burn state over one timestep with the generic Rosenbrock solver
/// (math/rosenbrock/rosenbrock.hpp) and checks the result. Ported from Microphysics' interfaces/burner.H and
/// integration/integrator_*_strang.H, Rosenbrock/actual_integrator.H and the burn-specific parts of
/// Rosenbrock/rosenbrock_integrator.H.

#include <concepts>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

#include "AMReX_ANSIEscCode.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "hydro/EOS.hpp"
#include "math/rosenbrock/rosenbrock.hpp"
#include "microphysics/burn_state.hpp"
#include "microphysics/integrator_params.hpp"
#include "microphysics/microphysics_traits.hpp"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// What an entry of the integrated vector y holds. The role decides the entry's tolerances, whether it counts toward
/// the step error, and how clean_state floors it.
enum class VarRole : std::uint8_t { Species, Energy, ChemPhotonNumber, ChemPhotonFlux, ThermalBandEnergy, DustAbsorbedEnergy };

/// What each entry of the integrated vector y (0-based) of a burn state type BurnT (burn_state.hpp) holds, and where
/// each part starts.
template <typename BurnT> struct BurnIndexMap;

/// The entries of a chemistry burn: the species, then the specific internal energy.
template <int NumSpec> struct BurnIndexMap<BurnState<NumSpec>> {
	static constexpr bool photochemistry = false;
	static constexpr bool q_dust = false;

	static constexpr int num_spec = NumSpec;
	/// the entry of the specific internal energy, right after the species
	static constexpr int energy_row = num_spec;
	static constexpr int neqs = num_spec + 1;

	/// The role of entry n of y.
	AMREX_GPU_HOST_DEVICE static constexpr auto role(const int n) -> VarRole
	{
		if (n < num_spec) {
			return VarRole::Species;
		}
		return VarRole::Energy;
	}
};

/// The entries of a photochemistry burn: those of the chemistry, then the chemical-band radiation, then the thermal bands.
template <int NumSpec, int NumChemBands, int VarsPerGroup, int NumThermalBands>
struct BurnIndexMap<PhotochemBurnState<NumSpec, NumChemBands, VarsPerGroup, NumThermalBands>> : BurnIndexMap<BurnState<NumSpec>> {
	using Base = BurnIndexMap<BurnState<NumSpec>>;
	static constexpr bool photochemistry = true;

	static constexpr int num_chem_rad_eqs = NumChemBands * VarsPerGroup;
	static constexpr int vars_per_group = VarsPerGroup;
	static constexpr int num_thermal_bands = NumThermalBands;
	/// the entry of the first thermal band; the others follow it
	static constexpr int thermal_band_row = Base::energy_row + num_chem_rad_eqs + 1;
	static constexpr int neqs = thermal_band_row + num_thermal_bands;

	/// Whether entry n is a chemical-band photon number (the only radiation entries in error control).
	AMREX_GPU_HOST_DEVICE static constexpr auto is_photon_number(const int n) -> bool
	{
		return n > Base::energy_row && n < thermal_band_row && (n - Base::energy_row - 1) % vars_per_group == 0;
	}

	/// The role of entry n of y.
	AMREX_GPU_HOST_DEVICE static constexpr auto role(const int n) -> VarRole
	{
		if (n <= Base::energy_row) {
			return Base::role(n);
		}
		if (n < thermal_band_row) {
			return is_photon_number(n) ? VarRole::ChemPhotonNumber : VarRole::ChemPhotonFlux;
		}
		return VarRole::ThermalBandEnergy;
	}
};

/// The entries of a photochemistry burn with dust absorption: those of the photochemistry, then the energy absorbed by dust.
template <int NumSpec, int NumChemBands, int VarsPerGroup, int NumThermalBands>
struct BurnIndexMap<DustyPhotochemBurnState<NumSpec, NumChemBands, VarsPerGroup, NumThermalBands>>
    : BurnIndexMap<PhotochemBurnState<NumSpec, NumChemBands, VarsPerGroup, NumThermalBands>> {
	using Base = BurnIndexMap<PhotochemBurnState<NumSpec, NumChemBands, VarsPerGroup, NumThermalBands>>;
	static constexpr bool q_dust = true;

	/// the entry of the energy absorbed by dust
	static constexpr int dust_row = Base::neqs;
	static constexpr int neqs = dust_row + 1;

	/// The role of entry n of y.
	AMREX_GPU_HOST_DEVICE static constexpr auto role(const int n) -> VarRole
	{
		if (n < dust_row) {
			return Base::role(n);
		}
		return VarRole::DustAbsorbedEnergy;
	}
};

/// An EOS a burn can use: the specific internal energy and the temperature from each other, and the energy's
/// temperature derivative, given the species number densities Species; and the lowest valid temperature.
template <typename EOS, typename Species> concept BurnEOS = requires(const amrex::Real x, const Species &xn)
{
	{EOS::TemperatureFromEnergy(x, xn)}->std::convertible_to<amrex::Real>;
	{EOS::EnergyFromTemperature(x, xn)}->std::convertible_to<amrex::Real>;
	{EOS::dEnergy_dT(xn)}->std::convertible_to<amrex::Real>;
	{EOS::eosMinTemp_}->std::convertible_to<amrex::Real>;
};

/// The burn as an ODE system for math::rosenbrock::integrate: the network's right-hand side and Jacobian on the burn
/// state, with Microphysics' tolerances, state checks and error norm.
template <typename problem_t, typename BurnT> struct BurnSystem {
	using traits = Microphysics_Traits<problem_t>;
	using IndexMap = BurnIndexMap<BurnT>;
	using Vector = amrex::GpuArray<amrex::Real, IndexMap::neqs>;
	using EOS = quokka::EOS<problem_t>;
	static_assert(BurnEOS<EOS, decltype(BurnT::xn)>,
		      "The problem's EOS (quokka::EOS<problem_t>) must give the temperature from the species number densities for "
		      "a burn: use EOSMultigamma (the default with chemistry or photochemistry), not EOSIdeal or EOSTabulated.");

	static constexpr int neqs = IndexMap::neqs;

	// The system is a view of one burn, built and used inside Burn, so it holds references.
	BurnT &state;						// NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
	const IntegratorParams &params;				// NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
	const typename traits::Network::Params &network_params; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

	/// Every entry counts toward the step error except, with photochemistry, the radiation entries other than the
	/// photon numbers.
	AMREX_GPU_HOST_DEVICE static constexpr auto counts_toward_error(const int n) -> bool
	{
		if constexpr (IndexMap::photochemistry) {
			return n <= IndexMap::energy_row || IndexMap::is_photon_number(n);
		} else {
			return true;
		}
	}

	/// The tolerances of entry n by its role.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto rtol(const int n) const -> amrex::Real
	{
		AMREX_ASSERT(counts_toward_error(n));
		const VarRole role = IndexMap::role(n);
		if (role == VarRole::Energy) {
			return params.rtol_enuc;
		}
		if (role == VarRole::ChemPhotonNumber) {
			return params.rtol_rad_num;
		}
		return params.rtol_spec;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto atol(const int n) const -> amrex::Real
	{
		AMREX_ASSERT(counts_toward_error(n));
		const VarRole role = IndexMap::role(n);
		if (role == VarRole::Energy) {
			return params.atol_enuc;
		}
		if (role == VarRole::ChemPhotonNumber) {
			return params.atol_rad_num;
		}
		return params.atol_spec;
	}

	/// The smallest magnitude of the finite-difference step of an entry that does not count toward the error.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto passive_floor() const -> amrex::Real { return params.SMALL_X_SAFE; }

	/// Copies the integrated vector y into the burn state.
	template <bool include_dust> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void copy_y_to_state(const Vector &y) const
	{
		for (int n = 0; n < IndexMap::num_spec; ++n) {
			state.xn[n] = y[n];
		}
		state.e = y[IndexMap::energy_row];
		if constexpr (IndexMap::photochemistry) {
			for (int n = 0; n < IndexMap::num_chem_rad_eqs; ++n) {
				state.rn[n] = y[IndexMap::energy_row + 1 + n];
			}
			for (int n = 0; n < IndexMap::num_thermal_bands; ++n) {
				state.re_thermal[n] = y[IndexMap::thermal_band_row + n];
			}
			if constexpr (include_dust && IndexMap::q_dust) {
				state.e_dust_absorbed = y[IndexMap::dust_row];
			}
		}
	}

	/// Copies the burn state into the integrated vector y.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void copy_state_to_y(Vector &y) const
	{
		for (int n = 0; n < IndexMap::num_spec; ++n) {
			y[n] = state.xn[n];
		}
		y[IndexMap::energy_row] = state.e;
		if constexpr (IndexMap::photochemistry) {
			for (int n = 0; n < IndexMap::num_chem_rad_eqs; ++n) {
				y[IndexMap::energy_row + 1 + n] = state.rn[n];
			}
			for (int n = 0; n < IndexMap::num_thermal_bands; ++n) {
				y[IndexMap::thermal_band_row + n] = state.re_thermal[n];
			}
			if constexpr (IndexMap::q_dust) {
				y[IndexMap::dust_row] = state.e_dust_absorbed;
			}
		}
	}

	/// Keeps the integrated variables within their valid ranges before the right-hand side is evaluated.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void clean_state(Vector &y) const
	{
		for (int n = 0; n < neqs; ++n) {
			switch (IndexMap::role(n)) {
				case VarRole::Species: // enforce positivity of the number densities
				case VarRole::ChemPhotonNumber:
				case VarRole::ThermalBandEnergy:
					y[n] = amrex::max(y[n], params.SMALL_X_SAFE);
					break;
				case VarRole::DustAbsorbedEnergy:
					y[n] = amrex::max(y[n], 0.0_rt);
					break;
				case VarRole::Energy:
				case VarRole::ChemPhotonFlux:
					break;
			}
		}
	}

	/// Whether the temperature of the last right-hand side is within the bounds where the network is evaluated.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto temperature_in_bounds() const -> bool
	{
		return !(state.T <= EOS::eosMinTemp_ || state.T >= traits::max_temp);
	}

	/// The right-hand side: cleans y, syncs the burn state with it, updates its temperature and calls the network. Zero
	/// outside the temperature bounds.
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rhs(const amrex::Real /*time*/, Vector &y, Vector &ydot) const
	{
		clean_state(y);

		copy_y_to_state<false>(y);
		if constexpr (traits::call_eos_in_rhs) {
			state.T = EOS::TemperatureFromEnergy(state.e, state.xn);
			state.dedT = EOS::dEnergy_dT(state.xn);
		}

		const bool in_bounds = temperature_in_bounds();

		if (in_bounds) {
			traits::Network::rhs(state, network_params, ydot);
			if constexpr (!traits::integrate_energy) {
				ydot[IndexMap::energy_row] = 0.0_rt;
			}
		}

		for (int n = 0; n < neqs; ++n) {
			ydot[n] = in_bounds ? ydot[n] : 0.0_rt;
		}
	}

	/// The analytic Jacobian at y, the state of the last right-hand side. Zero outside the temperature bounds. The
	/// matrix is zeroed before the network fills it, so a network only has to write its nonzero entries.
	// TODO(james471): share the rates between the right-hand side and the Jacobian (they are evaluated at the same state),
	// once jaff generates networks that support it. On hold: jaff's developers are adding the feature.
	template <typename Matrix> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void jac(const amrex::Real /*time*/, Vector &y, Matrix &J) const
	{
		for (int j = 0; j < neqs; ++j) {
			for (int i = 0; i < neqs; ++i) {
				J(i, j) = 0.0_rt;
			}
		}

		if (!temperature_in_bounds()) {
			return;
		}

		copy_y_to_state<true>(y);

		traits::Network::jac(state, network_params, J);

		if constexpr (!traits::integrate_energy) {
			for (int j = 0; j < neqs; ++j) {
				J(IndexMap::energy_row, j) = 0.0_rt;
			}
		}
	}

	/// Whether y is a state the right-hand side may be evaluated at: finite, the species and the photon numbers not
	/// below minus their absolute tolerances, and the energy positive.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto valid_state(const Vector &y) const -> bool
	{
		for (int n = 0; n < neqs; ++n) {
			if (std::isnan(y[n]) || std::isinf(y[n])) {
				return false;
			}
		}

		for (int n = 0; n < neqs; ++n) {
			switch (IndexMap::role(n)) {
				case VarRole::Species:
					if (y[n] < -params.atol_spec) {
						return false;
					}
					break;
				case VarRole::Energy:
					if (y[n] <= 0.0_rt) {
						return false;
					}
					break;
				case VarRole::ChemPhotonNumber:
					if (y[n] < -params.atol_rad_num) {
						return false;
					}
					break;
				case VarRole::ChemPhotonFlux:
				case VarRole::ThermalBandEnergy:
				case VarRole::DustAbsorbedEnergy:
					break;
			}
		}

		return true;
	}

	/// Whether a step from y to ynew changes no species above X_reject_buffer * atol_spec by more than a factor of 4.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto valid_update(const Vector &y, const Vector &ynew) const -> bool
	{
		constexpr amrex::Real increase_change_factor = 4.0_rt;
		constexpr amrex::Real decrease_change_factor = 0.25_rt;

		for (int i = 0; i < IndexMap::num_spec; ++i) {
			if (std::abs(y[i]) > params.X_reject_buffer * params.atol_spec && std::abs(ynew[i]) > params.X_reject_buffer * params.atol_spec &&
			    (std::abs(ynew[i]) > increase_change_factor * std::abs(y[i]) || std::abs(ynew[i]) < decrease_change_factor * std::abs(y[i]))) {
				return false;
			}
		}

		return true;
	}
};

/// The solver options from Microphysics_Traits.
template <typename problem_t> struct BurnOptions {
	using traits = Microphysics_Traits<problem_t>;
	static constexpr bool analytic_jacobian = traits::analytic_jacobian;
	static constexpr bool numerical_jacobian_fallback = traits::numerical_jacobian_fallback;
	static constexpr bool pivoting = traits::pivoting;
	static constexpr amrex::Real max_dt = traits::ode_max_dt;
};

namespace detail
{
/// for outputting the burn state failure, use enough precision to be reproducible (not guaranteed for std::fixed)
constexpr int OUTDIGITS = std::numeric_limits<amrex::Real>::max_digits10;

/// Prints a summary of the burn with params.burner_verbose, and the state of a failed burn; state_save is the state at
/// the start of the burn. Host code: Burn calls it only in CPU builds.
template <typename IndexMap, typename BurnT>
AMREX_INLINE void Report(const BurnT &state, const BurnT &state_save, const amrex::Real dt, const IntegratorParams &params)
{
	if (params.burner_verbose) {
		std::cout << "integration summary: " << '\n';
		std::cout << "dens: " << state.rho << " temp: " << state.T << '\n';
		std::cout << " energy released: " << state.e << '\n';
		std::cout << "number of steps taken: " << state.n_step << '\n';
		std::cout << "number of f evaluations: " << state.n_rhs << '\n' << std::flush;
	}

	if (!state.success) {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay): AMReX's escape codes are char arrays
		std::cout << amrex::Font::Bold << amrex::FGColor::Red << "[ERROR] integration failed in net" << amrex::ResetDisplay << '\n';
		std::cout << "integrator status = " << math::rosenbrock::StatusName(state.integrator_status) << '\n';
		if (state.integrator_status == math::rosenbrock::IntegratorStatus::Success) {
			std::cout << "  Integration exited successfully, but a check on the data values failed" << '\n';
		}
		std::cout << "zone = (" << state.i << ", " << state.j << ", " << state.k << ")" << '\n';
		std::cout << "time = " << state.time << '\n';
		std::cout << "dt = " << std::setprecision(OUTDIGITS) << dt << '\n';
		std::cout << "temp start = " << std::setprecision(OUTDIGITS) << state_save.T << '\n';
		std::cout << "xn start = ";
		for (const double X : state_save.xn) {
			std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
		}
		if constexpr (IndexMap::photochemistry) {
			std::cout << '\n';
			std::cout << "rn start = ";
			for (int n = 0; n < IndexMap::num_chem_rad_eqs; ++n) {
				const double X = state_save.rn[n];
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			std::cout << '\n';
			std::cout << "re_thermal start = ";
			for (int n = 0; n < IndexMap::num_thermal_bands; ++n) {
				const double X = state_save.re_thermal[n];
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			if constexpr (IndexMap::q_dust) {
				std::cout << '\n';
				std::cout << "e_dust_absorbed start = " << std::scientific << std::setprecision(OUTDIGITS) << state_save.e_dust_absorbed;
			}
		}
		std::cout << '\n';
		std::cout << "dens current = " << std::setprecision(OUTDIGITS) << state.rho << '\n';
		std::cout << "temp current = " << std::setprecision(OUTDIGITS) << state.T << '\n';
		std::cout << "xn current = ";
		for (const double X : state.xn) {
			std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
		}
		if constexpr (IndexMap::photochemistry) {
			std::cout << '\n';
			std::cout << "rn current = ";
			for (const double X : state.rn) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			std::cout << '\n';
			std::cout << "re_thermal current = ";
			for (const double X : state.re_thermal) {
				std::cout << std::scientific << std::setprecision(OUTDIGITS) << X << " ";
			}
			if constexpr (IndexMap::q_dust) {
				std::cout << '\n';
				std::cout << "e_dust_absorbed current = " << std::scientific << std::setprecision(OUTDIGITS) << state.e_dust_absorbed;
			}
		}
		std::cout << '\n';
		std::cout << "energy generated = " << state.e << '\n' << std::flush;
	}
}
} // namespace detail

/// Integrates the burn state over dt with Microphysics_Traits<problem_t>::Tableau. The state comes in with rho, T, e,
/// the species and, with photochemistry, the radiation variables, and leaves with the integrated variables at the end
/// of the burn, the work counters, integrator_status and success. success is false if the integration failed or a species or
/// a chemical-band photon number ended below minus its failure tolerance.
template <typename problem_t, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Burn(BurnT &state, const amrex::Real dt, const IntegratorParams &params,
						   const typename Microphysics_Traits<problem_t>::Network::Params &network_params)
{
	namespace rb = quokka::math::rosenbrock;
	using Tableau = typename Microphysics_Traits<problem_t>::Tableau;
	using System = BurnSystem<problem_t, BurnT>;
	using IndexMap = typename System::IndexMap;

	if constexpr (IndexMap::photochemistry) {
		// The network's integrate_rad_flux is the source of truth; the burn state carries it as vars_per_group, which the
		// network reads together with the rest of the variable layout.
		static_assert(BurnT::vars_per_group == (Microphysics_Traits<problem_t>::Network::integrate_rad_flux ? 2 : 1),
			      "the burn state's flux layout (vars_per_group) must match the network's integrate_rad_flux");
	}

	System sys{state, params, network_params};
	rb::State<System::neqs, Tableau::stages> rstate{};

	state.success = true;

	rstate.t = 0.0_rt;
	rstate.tout = dt;

	// The (rho, T) coming in are valid: fill e and de/dT from them.
	state.e = System::EOS::EnergyFromTemperature(state.T, state.xn);
	state.dedT = System::EOS::dEnergy_dT(state.xn);

	if constexpr (IndexMap::q_dust) {
		state.e_dust_absorbed = 0.0_rt;
	}

	sys.copy_state_to_y(rstate.y);

#ifndef AMREX_USE_GPU
	const BurnT state_save = state; // the state at the start of the burn, for the failure report
#endif

	const rb::IntegratorStatus istate = rb::integrate<Tableau, BurnOptions<problem_t>>(sys, rstate, params.ode_max_steps);

	// copy the integration data back to the burn state
	sys.template copy_y_to_state<true>(rstate.y);
	state.time = rstate.t;

	state.n_rhs = rstate.n_rhs;
	state.n_jac = rstate.n_jac;
	state.n_step = rstate.n_step;
	state.integrator_status = istate;

	// The integrator may not always fail even though it can lead to
	// unphysical states.  Add some checks that indicate a burn fail
	// even if the integrator thinks the integration was successful.

	if (istate != rb::IntegratorStatus::Success) {
		state.success = false;
	}

	for (int n = 0; n < System::neqs; ++n) {
		switch (IndexMap::role(n)) {
			case VarRole::Species:
				if (rstate.y[n] < -params.species_failure_tolerance) {
					state.success = false;
				}
				break;
			case VarRole::ChemPhotonNumber:
				if (rstate.y[n] < -params.radiation_failure_tolerance) {
					state.success = false;
				}
				break;
			case VarRole::Energy:
			case VarRole::ChemPhotonFlux:
			case VarRole::ThermalBandEnergy:
			case VarRole::DustAbsorbedEnergy:
				break;
		}
	}

#ifndef AMREX_USE_GPU
	detail::Report<IndexMap>(state, state_save, dt, params);
#endif
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_BURNER_HPP_
