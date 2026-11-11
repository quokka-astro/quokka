// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef EOS_HPP_
#define EOS_HPP_
/// \file EOS.hpp
/// \brief A class for equation of state calculations.

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "util/Optional.hpp"

#include "AMReX.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "physics_info.hpp"
#include <AMReX_Print.H>

#include "cooling/EOSTabulatedRegistry.hpp"
#include "fundamental_constants.H"
#include "hydro/EOSFloors.hpp"
#include "microphysics/microphysics_traits.hpp"

// Microphysics' EOS, which every file including this header used to get through eos.H, put amrex::literals (_rt, _prt)
// in the global namespace, and Quokka code relies on it (for example particles/PhysicsParticles.hpp).
using namespace amrex::literals; // NOLINT(google-build-using-namespace,google-global-names-in-headers)

namespace quokka
{

// The EOSs. Each is a complete struct; quokka::EOS<problem_t> (below) is the one the problem uses.
template <typename problem_t> struct EOSIdeal;
template <typename problem_t> struct EOSMultigamma;
template <typename problem_t> struct EOSTabulated;

// Primary EOS_Traits template: default values for an ideal gamma-law gas. Full specializations need only define the
// values they override. A problem without a network may also name its EOS (using EOS = ...); without it, it gets
// EOSIdeal. A problem with a network gets the EOS its network names (see DefaultEOS below).
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
};

// What every EOS has: the conversions between the gas internal and total energy, which only add or subtract the
// kinetic energy.
struct EOSBase {
	// Compute gas internal energy from gas total energy (Eint + Ekin, NOT including B field).
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double rho, double mx, double my, double mz, double Etot, double magnetic_energy)
	    -> double
	{
		const double Ekin = 0.5 * (mx * mx + my * my + mz * mz) / rho;
		const double Eint = Etot - Ekin - magnetic_energy;
		AMREX_ASSERT_WITH_MESSAGE(Eint > 0., "Gas internal energy is not positive!");
		return Eint;
	}

	// Compute gas total energy (Eint + Ekin, NOT including B field) from gas internal energy.
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double rho, double mx, double my, double mz, double Eint, double magnetic_energy)
	    -> double
	{
		const double Ekin = 0.5 * (mx * mx + my * my + mz * mz) / rho;
		return Eint + Ekin + magnetic_energy;
	}
};

// ==================== EOSIdeal ====================
// gamma-law ideal gas. Always available.

template <typename problem_t> struct EOSIdeal : EOSBase {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;
	static constexpr amrex::Real boltzmann_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::boltzmann_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return C::k_B /
			       (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
				(Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
	}();

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Tgas = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			GasState estate;
			estate.rho = rho;
			estate.e = Eint / rho;
			estate.mu = mean_molecular_weight_ / C::m_u;
			Solve(Input::RhoE, estate);
			Tgas = estate.T * C::k_B / boltzmann_constant_;
		}
		return Tgas;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Eint = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			GasState estate;
			estate.rho = rho;
			estate.T = Tgas;
			estate.mu = mean_molecular_weight_ / C::m_u;
			Solve(Input::RhoT, estate);
			Eint = estate.e * rho * boltzmann_constant_ / C::k_B;
		}
		return Eint;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real Eint = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			GasState estate;
			estate.rho = rho;
			estate.p = Pressure;
			estate.mu = mean_molecular_weight_ / C::m_u;
			Solve(Input::RhoP, estate);
			Eint = estate.e * rho;
		}
		return Eint;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::Real dEint_dT = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			GasState estate;
			estate.rho = rho;
			estate.T = Tgas;
			estate.mu = mean_molecular_weight_ / C::m_u;
			Solve(Input::RhoT, estate);
			dEint_dT = estate.dedT * rho * boltzmann_constant_ / C::k_B;
		}
		return dEint_dT;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		amrex::Real deint_dRho = NAN;
		amrex::Real deint_dP = NAN;
		amrex::Real dRho_dP = NAN;
		amrex::Real dP_dRho_s = NAN;
		amrex::Real G = NAN;
		amrex::ignore_unused(massScalars);

		if constexpr (gamma_ != 1.0) {
			GasState estate;
			estate.rho = rho;
			estate.p = P;
			estate.mu = mean_molecular_weight_ / C::m_u;
			Solve(Input::RhoP, estate);
			deint_dRho = estate.dedr;
			deint_dP = 1.0 / estate.dpde;
			dRho_dP = 1.0 / (estate.dpdr * C::k_B / boltzmann_constant_);
			dP_dRho_s = estate.cs * estate.cs;
			G = estate.G;
		}
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		amrex::Real P = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(Eint);
			amrex::ignore_unused(massScalars);
			P = rho * EOS_Traits<problem_t>::cs_isothermal * EOS_Traits<problem_t>::cs_isothermal;
			return P;
		}
		amrex::ignore_unused(massScalars);

		GasState estate;
		estate.rho = rho;
		if (rho == 0.0) {
			estate.e = 0;
		} else {
			estate.e = Eint / rho;
		}
		estate.mu = mean_molecular_weight_ / C::m_u;
		Solve(Input::RhoE, estate);
		P = estate.p;
		return P;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::Real cs = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(rho);
			amrex::ignore_unused(Pressure);
			amrex::ignore_unused(massScalars);
			cs = EOS_Traits<problem_t>::cs_isothermal;
			return cs;
		}
		amrex::ignore_unused(massScalars);

		GasState estate;
		estate.rho = rho;
		estate.p = Pressure;
		estate.mu = mean_molecular_weight_ / C::m_u;
		Solve(Input::RhoP, estate);
		cs = estate.cs;
		return cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		amrex::Real cs = NAN;

		if constexpr (gamma_ == 1.0) {
			static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
			amrex::ignore_unused(rho);
			amrex::ignore_unused(Pressure);
			cs = EOS_Traits<problem_t>::cs_isothermal;
		} else {
			cs = std::sqrt(Pressure / rho);
		}
		return cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}

      private:
	/// The gas state of one evaluation, in CGS: the inputs (rho and one of T, e, p, with mu) and the outputs read above.
	struct GasState {
		amrex::Real rho{};
		amrex::Real T{};
		amrex::Real p{};
		amrex::Real e{}; ///< specific internal energy
		amrex::Real mu{};
		amrex::Real dpdT{};
		amrex::Real dpdr{};
		amrex::Real dpde{};
		amrex::Real dedT{};
		amrex::Real dedr{};
		amrex::Real cs{};
		amrex::Real G{};
	};

	/// Which of T, e and p is given with rho.
	enum class Input : std::uint8_t { RhoT, RhoE, RhoP };

	/// Complete the state from rho and the given quantity, as Microphysics' eos() did with its gamma_law backend: rho is
	/// clamped to the floors, and so is T when T is given; an e or p outside its bounds resets the state to its clamped
	/// rho and T (0 unless set, so the temperature floor).
	AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static void Solve(Input input, GasState &state)
	{
		bool has_been_reset = false;
		state.rho = amrex::min(eos_floors::max_dens, amrex::max(eos_floors::min_dens, state.rho));
		switch (input) {
			case Input::RhoT:
				state.T = amrex::min(eos_floors::max_temp, amrex::max(eos_floors::min_temp, state.T));
				break;
			case Input::RhoE:
				has_been_reset = state.e < eos_floors::min_e || state.e > eos_floors::max_e;
				break;
			case Input::RhoP:
				has_been_reset = state.p < eos_floors::min_p || state.p > eos_floors::max_p;
				break;
		}
		if (has_been_reset) {
			state.T = amrex::min(eos_floors::max_temp, amrex::max(eos_floors::min_temp, state.T));
			state.rho = amrex::min(eos_floors::max_dens, amrex::max(eos_floors::min_dens, state.rho));
			input = Input::RhoT;
		}

		const amrex::Real m_nucleon = C::m_u;

		switch (input) {
			case Input::RhoT:
				break;
			case Input::RhoE:
				state.T = state.e * state.mu * m_nucleon * (gamma_ - 1.0) / C::k_B;
				break;
			case Input::RhoP:
				state.T = state.p * state.mu * m_nucleon / (C::k_B * state.rho);
				break;
		}

		const amrex::Real Tinv = 1.0 / state.T;
		const amrex::Real rhoinv = 1.0 / state.rho;
		const amrex::Real pressure = state.rho * state.T * C::k_B / (state.mu * m_nucleon);
		const amrex::Real energy = pressure / (gamma_ - 1.0) * rhoinv;

		state.p = pressure;
		state.e = energy;
		state.dpdT = state.p * Tinv;
		state.dpdr = state.p * rhoinv;
		state.dedT = state.e * Tinv;
		state.dedr = 0.0;
		state.dpde = state.dpdT * (1.0 / state.dedT);
		state.cs = std::sqrt(gamma_ * state.p * rhoinv);
		state.G = 0.5 * (1.0 + gamma_);
	}
};

// ==================== EOSMultigamma ====================
// The EOS of the problems with a network: an ideal gas of the network's species, each with its own mass and adiabatic
// index (Network::SpeciesMass(), Network::SpeciesGamma()). Ported from Microphysics' photoionization EOS.

template <typename problem_t> struct EOSMultigamma : EOSBase {
	using Network = typename Microphysics_Traits<problem_t>::Network;

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = false;
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	static_assert(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS, "EOSMultigamma works in CGS: the species masses are in grams");
	static_assert(nmscalars_ == Network::NumSpec, "EOSMultigamma reads the species from the mass scalars, so the problem needs one per species");

	/// Which of T, e and p is given with rho and the species.
	enum class Input : std::uint8_t { RhoT, RhoE, RhoP };

	/// f_i / 2 = 1 / (gamma_i - 1) of each species, its degrees of freedom over 2, so that Solve() multiplies by it
	/// instead of dividing by gamma_i - 1.
	static constexpr auto SpeciesFiOver2() -> amrex::GpuArray<amrex::Real, Network::NumSpec>
	{
		amrex::GpuArray<amrex::Real, Network::NumSpec> f{};
		for (int n = 0; n < Network::NumSpec; ++n) {
			f.arr[n] = 1.0 / (Network::SpeciesGamma().arr[n] - 1.0);
		}
		return f;
	}

	/// A gas state with every quantity Solve() sets: the hydro uses it, and so can a problem that needs a pressure.
	struct GasState {
		amrex::Real rho{};
		amrex::Real T{};
		amrex::Real p{};
		amrex::Real e{};				     ///< specific internal energy
		amrex::GpuArray<amrex::Real, Network::NumSpec> xn{}; ///< number densities
		amrex::Real mu{};
		amrex::Real dpdT{};
		amrex::Real dpdr{};
		amrex::Real dpde{};
		amrex::Real dedT{};
		amrex::Real dedr{};
		amrex::Real cs{};
		amrex::Real G{};
	};

	/// Complete a state from rho, the species' number densities xn and one of T, e (specific) and p, as Microphysics'
	/// eos() did with its photoionization backend: rho is clamped to the floors (quokka::eos_floors), and so is T when T
	/// is given; an e or p outside its bounds resets the state to its clamped rho and T. Works on any state with rho, T,
	/// e and xn (for example quokka::microphysics::BurnState); also sets mu, dedr, p, dpdT, dpdr, dpde, cs and G if the
	/// state has them.
	template <typename State> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static void Solve(Input input, State &state)
	{
		constexpr bool has_pressure = requires { state.p; };

		bool has_been_reset = false;
		state.rho = amrex::min(eos_floors::max_dens, amrex::max(eos_floors::min_dens, state.rho));
		switch (input) {
			case Input::RhoT:
				state.T = amrex::min(eos_floors::max_temp, amrex::max(eos_floors::min_temp, state.T));
				break;
			case Input::RhoE:
				has_been_reset = state.e < eos_floors::min_e || state.e > eos_floors::max_e;
				break;
			case Input::RhoP:
				if constexpr (has_pressure) {
					has_been_reset = state.p < eos_floors::min_p || state.p > eos_floors::max_p;
				}
				break;
		}
		if (has_been_reset) {
			state.T = amrex::min(eos_floors::max_temp, amrex::max(eos_floors::min_temp, state.T));
			state.rho = amrex::min(eos_floors::max_dens, amrex::max(eos_floors::min_dens, state.rho));
			input = Input::RhoT;
		}

		constexpr amrex::Real protonmass = C::m_p;
		constexpr auto spmasses = Network::SpeciesMass();
		constexpr auto fi_over_2 = SpeciesFiOver2();

		amrex::Real sum_Abarinv = 0.0;
		amrex::Real sum_ni_fi_over_2 = 0.0;
		amrex::Real rhotot = 0.0;

		for (int n = 0; n < Network::NumSpec; ++n) {
			rhotot += state.xn[n] * spmasses[n];
			sum_ni_fi_over_2 += state.xn[n] * fi_over_2[n];
			sum_Abarinv += state.xn[n];
		}

		const amrex::Real fi_over_2_avg = sum_ni_fi_over_2 / sum_Abarinv;

		sum_Abarinv *= protonmass / rhotot;
		if constexpr (requires { state.mu; }) {
			state.mu = 1.0 / sum_Abarinv;
		}

		amrex::Real temp = NAN;
		amrex::Real dens = NAN;
		amrex::Real eint = NAN;

		switch (input) {
			case Input::RhoT:
				temp = state.T;
				dens = state.rho;
				eint = sum_ni_fi_over_2 * C::k_B * temp / rhotot;
				break;
			case Input::RhoE:
				dens = state.rho;
				AMREX_ASSERT(state.e > 0.);
				temp = state.e * rhotot / (sum_ni_fi_over_2 * C::k_B);
				eint = state.e;
				break;
			case Input::RhoP:
				if constexpr (has_pressure) {
					dens = state.rho;
					AMREX_ASSERT(state.p > 0.);
					eint = state.p * fi_over_2_avg / dens;
					temp = eint * rhotot / (sum_ni_fi_over_2 * C::k_B);
				}
				break;
		}

		state.T = temp;
		state.rho = dens;
		state.e = eint;

		if constexpr (has_pressure) {
			const amrex::Real pressure = state.rho * eint / fi_over_2_avg;
			state.p = pressure;
			state.dpdT = pressure / temp;
			state.dpdr = pressure / dens;
			state.cs = std::sqrt((1.0 + 1.0 / fi_over_2_avg) * state.p / state.rho);
			if constexpr (requires { state.G; }) {
				state.G = 0.5 * (1.0 + (1.0 + 1.0 / fi_over_2_avg));
			}
		}

		const amrex::Real dedT = sum_ni_fi_over_2 * C::k_B / rhotot;
		state.dedT = dedT;
		if constexpr (requires { state.dedr; }) {
			state.dedr = 0.0;
		}
		if constexpr (has_pressure) {
			state.dpde = state.dpdT / state.dedT;
		}
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoE, chemstate);
		return chemstate.T;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoT, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoP, chemstate);
		return chemstate.e * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.T = Tgas;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoT, chemstate);
		return chemstate.dedT * chemstate.rho;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.p = P;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoP, chemstate);
		const amrex::Real deint_dRho = chemstate.dedr;
		const amrex::Real deint_dP = 1.0 / chemstate.dpde;
		amrex::Real dRho_dP = 1.0 / (chemstate.dpdr * C::k_B / boltzmann_constant_);
		const amrex::Real dP_dRho_s = chemstate.cs * chemstate.cs;
		const amrex::Real G = chemstate.G;
		return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMultigamma does not support isothermal (gamma=1) problems.");
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.e = Eint / rho;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoE, chemstate);
		return chemstate.p;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		static_assert(gamma_ != 1.0, "EOSMultigamma does not support isothermal (gamma=1) problems.");
		GasState chemstate;
		chemstate.rho = rho;
		chemstate.p = Pressure;
		SetSpecies(chemstate, massScalars);
		Solve(Input::RhoP, chemstate);
		return chemstate.cs;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		static_assert(gamma_ == 1.0, "ComputeIsothermalSoundSpeed does not support general EOS");
		static_assert(EOS_Traits<problem_t>::cs_isothermal > 0.0, "EOS_Traits<problem_t>::cs_isothermal must be set when gamma=1.");
		amrex::ignore_unused(rho);
		amrex::ignore_unused(Pressure);
		return EOS_Traits<problem_t>::cs_isothermal;
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real /*rho*/, amrex::Real /*Eint*/,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/ = {}) -> amrex::Real
	{
		// sizeof(problem_t)==0: fires only on instantiation, not at parse time (C++20 ill-formed NDR workaround)
		static_assert(sizeof(problem_t) == 0, "ComputeEntropyFromRhoEint is only supported by the EOSTabulated backend");
		return 0.0;
	}

      private:
	/// The species' number densities from the mass scalars (partial densities); -1 without them.
	AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static void SetSpecies(GasState &state,
									quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
	{
		for (double &ii : state.xn) {
			ii = -1.0;
		}

		if (massScalars) {
			const auto &massArray = *massScalars;
			for (int nn = 0; nn < nmscalars_; ++nn) {
				state.xn[nn] = massArray[nn] / Network::SpeciesMass()[nn];
			}
		}
	}
};

// ==================== EOSTabulated ====================
// Temperature methods read the resampled table; all other methods delegate to EOSIdeal.

template <typename problem_t> struct EOSTabulated : EOSBase {
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr bool is_tabulated = true;
	static constexpr amrex::Real gamma_ = EOSIdeal<problem_t>::gamma_;
	static constexpr amrex::Real boltzmann_constant_ = EOSIdeal<problem_t>::boltzmann_constant_;

	// Temperature methods — use the resampled table via the global registry
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeTgasFromEgas(rho, Eint, get_tables());
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeEgasFromTgas(rho, Tgas, get_tables());
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
				  quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		// One root-find, then use DataTable::partial_derivative for ∂T/∂eint_specific.
		// Table axes: (rho, eint_specific = Eint/rho). dEint_density/dT = rho / (∂T/∂eint_specific).
		const amrex::Real Eint = ComputeEintFromTgas(rho, Tgas, massScalars);
		auto const &tables = get_tables();
		const amrex::Real dT_deint = tables.all_tables.partial_derivative({rho, Eint / rho}, 1, ResampledCooling::TEMPERATURE_IDX);
		AMREX_ASSERT(dT_deint > amrex::Real(0.0));
		return rho / dT_deint;
	}

	// Non-temperature methods — delegate to EOSIdeal
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeEintFromPres(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	{
		return EOSIdeal<problem_t>::ComputeOtherDerivatives(rho, P, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputePressure(rho, Eint, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeSoundSpeed(rho, Pressure, massScalars);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real
	{
		return EOSIdeal<problem_t>::ComputeIsothermalSoundSpeed(rho, Pressure);
	}

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEntropyFromRhoEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real
	{
		amrex::ignore_unused(massScalars);
		return ResampledCooling::ComputeEntropyFromRhoEint(rho, Eint, get_tables());
	}

      private:
	// Returns the device or host table handle appropriate for the current execution context.
	// The registration invariant (non-null pointer) is checked once at setup; only a
	// debug-mode assert is needed here to avoid per-cell overhead in Release GPU kernels.
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto get_tables() -> ResampledCooling::resampledGpuConstTables const &
	{
		auto *reg = ResampledCooling::getEOSTabulatedRegistry();
		AMREX_ASSERT(reg != nullptr);
		AMREX_IF_ON_DEVICE((return reg->device;))
		AMREX_IF_ON_HOST((return reg->host;))
	}
};

// The EOS of a network. A network names the EOS its equations assume in jaff's words (Network::EOSType, the type of
// jaff's [network.eos]), so that jaff need not know Quokka's EOSs; this is the one place that maps them.
template <typename T> struct NetworkEOS {
	using Network = typename Microphysics_Traits<T>::Network;
	static_assert(Network::EOSType == "ideal" || Network::EOSType == "multi_gamma",
		      "Quokka has no EOS for this network's EOSType: it maps \"ideal\" and \"multi_gamma\" to EOSMultigamma");
	using type = EOSMultigamma<T>;
};

// The EOS a problem gets when its EOS_Traits names none: its network's, or EOSIdeal without a network.
template <typename T, bool has_network = !std::is_same_v<typename Microphysics_Traits<T>::Network, quokka::networks::None>> struct DefaultEOS {
	using type = EOSIdeal<T>;
};

template <typename T> struct DefaultEOS<T, true> {
	using type = typename NetworkEOS<T>::type;
};

namespace detail
{
// EOS_Traits<T>::EOS if the problem names one, else DefaultEOS.
template <typename T, typename = void> struct EOSChoice {
	using type = typename DefaultEOS<T>::type;
};

template <typename T> struct EOSChoice<T, std::void_t<typename EOS_Traits<T>::EOS>> {
	static_assert(std::is_same_v<typename Microphysics_Traits<T>::Network, quokka::networks::None>,
		      "A problem with a network takes its EOS from the network (Network::EOSType); it cannot also name one in EOS_Traits::EOS");
	using type = typename EOS_Traits<T>::EOS;
};
} // namespace detail

// ==================== EOS (public name) ====================
// The problem's EOS. A problem without a network that needs its own EOS writes a struct, usually derived from one of
// the above, and names it in EOS_Traits<problem_t>::EOS.

template <typename problem_t> using EOS = typename detail::EOSChoice<problem_t>::type;

} // namespace quokka

#endif // EOS_HPP_
