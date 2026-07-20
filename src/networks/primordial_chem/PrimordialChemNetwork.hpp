#ifndef QUOKKA_PRIMORDIAL_CHEM_NETWORK_HPP_
#define QUOKKA_PRIMORDIAL_CHEM_NETWORK_HPP_

#include <array>
#include <cmath>
#include <string_view>

#include "AMReX_ParmParse.H"
#include "chemistry/ChemistryNetwork.hpp"
#include "networks/primordial_chem/GeneratedRhs.hpp"

namespace quokka::chemistry
{

struct PrimordialChemParameters {
	amrex::Real redshift = 30.0;
};

[[nodiscard]] inline auto readPrimordialChemParameters() -> PrimordialChemParameters
{
	PrimordialChemParameters values{};
	amrex::ParmParse const parameters("network");
	parameters.query("redshift", values.redshift);
	return values;
}

struct PrimordialChemNetwork {
	static constexpr amrex::Real boltzmann_constant = 1.380649e-16;
	static constexpr int species_count = 14;
	static constexpr int variable_count = species_count + 1;
	static constexpr int energy = species_count;
	static constexpr ChemistryNetworkMetadata metadata = {"primordial_chem", "2024-07", species_count, variable_count};
	static constexpr std::array<VariableMetadata, variable_count> variables = {
	    VariableMetadata{"electron", VariableRole::species, true},
	    VariableMetadata{"H+", VariableRole::species, true},
	    VariableMetadata{"H", VariableRole::species, true},
	    VariableMetadata{"H-", VariableRole::species, true},
	    VariableMetadata{"D+", VariableRole::species, true},
	    VariableMetadata{"D", VariableRole::species, true},
	    VariableMetadata{"H2+", VariableRole::species, true},
	    VariableMetadata{"D-", VariableRole::species, true},
	    VariableMetadata{"H2", VariableRole::species, true},
	    VariableMetadata{"HD+", VariableRole::species, true},
	    VariableMetadata{"HD", VariableRole::species, true},
	    VariableMetadata{"He++", VariableRole::species, true},
	    VariableMetadata{"He+", VariableRole::species, true},
	    VariableMetadata{"He", VariableRole::species, true},
	    VariableMetadata{"specific_internal_energy", VariableRole::energy, true},
	};
	static constexpr std::array<std::string_view, species_count> species_names = {"electron", "H+", "H",   "H-", "D+",   "D",   "H2+",
										      "D-",	  "H2", "HD+", "HD", "He++", "He+", "He"};
	static constexpr amrex::GpuArray<amrex::Real, species_count> species_masses = {
	    9.10938188e-28,    1.67262158e-24,	  1.67353251819e-24, 1.67444345638e-24, 3.34512158e-24, 3.34603251819e-24, 3.34615409819e-24,
	    3.34694345638e-24, 3.34706503638e-24, 5.01865409819e-24, 5.01956503638e-24, 6.69024316e-24, 6.69115409819e-24, 6.69206503638e-24};
	static constexpr amrex::GpuArray<amrex::Real, species_count> species_gammas = {
	    5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 1.4, 5.0 / 3.0, 1.4, 1.4, 1.4, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0};

	PrimordialChemParameters parameters{};

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int variable) noexcept -> VariableRole
	{
		return variable == energy ? VariableRole::energy : VariableRole::species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int /*variable*/) noexcept -> bool { return true; }

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto mass_density(IntegratorState<variable_count> const &state) noexcept -> amrex::Real
	{
		amrex::Real density = 0.0;
		for (int species = 0; species < species_count; ++species) {
			density += state.values[species] * species_masses[species];
		}
		return density;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto heat_capacity(IntegratorState<variable_count> const &state) noexcept -> amrex::Real
	{
		amrex::Real capacity = 0.0;
		for (int species = 0; species < species_count; ++species) {
			capacity += state.values[species] / (species_gammas[species] - 1.0);
		}
		return capacity;
	}

	AMREX_GPU_HOST_DEVICE static void update_thermodynamics(IntegratorState<variable_count> &state) noexcept
	{
		const amrex::Real species_density = mass_density(state);
		const amrex::Real capacity = heat_capacity(state);
		if (species_density > 0.0 && capacity > 0.0) {
			state.temperature = state.values[energy] * species_density / (capacity * boltzmann_constant);
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto temperature(IntegratorState<variable_count> state) noexcept -> amrex::Real
	{
		update_thermodynamics(state);
		return state.temperature;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto specific_energy_from_temperature(amrex::GpuArray<amrex::Real, species_count> const &number_density,
											 amrex::Real temperature) noexcept -> amrex::Real
	{
		amrex::Real density = 0.0;
		amrex::Real capacity = 0.0;
		for (int species = 0; species < species_count; ++species) {
			density += number_density[species] * species_masses[species];
			capacity += number_density[species] / (species_gammas[species] - 1.0);
		}
		return capacity * boltzmann_constant * temperature / density;
	}

	AMREX_GPU_HOST_DEVICE static void set_specific_energy_from_temperature(IntegratorState<variable_count> &state, amrex::Real temperature) noexcept
	{
		amrex::GpuArray<amrex::Real, species_count> number_density{};
		for (int species = 0; species < species_count; ++species) {
			number_density[species] = state.values[species];
		}
		state.values[energy] = specific_energy_from_temperature(number_density, temperature);
		update_thermodynamics(state);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto generated_state(IntegratorState<variable_count> state) noexcept -> primordial_detail::PrimordialRhsState
	{
		update_thermodynamics(state);
		primordial_detail::PrimordialRhsState generated{};
		generated.density = state.density;
		generated.temperature = state.temperature;
		const amrex::Real capacity = heat_capacity(state);
		generated.temperature_derivative = capacity * boltzmann_constant / state.density;
		for (int species = 0; species < species_count; ++species) {
			generated.species[species] = state.values[species];
		}
		return generated;
	}

	AMREX_GPU_HOST_DEVICE void rhs(IntegratorState<variable_count> const &state, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		auto generated = generated_state(state);
		amrex::Array1D<amrex::Real, 1, variable_count> generated_derivative{};
		primordial_detail::actual_rhs(generated, generated_derivative, parameters.redshift);
		for (int variable = 0; variable < variable_count; ++variable) {
			derivative[variable] = generated_derivative(variable + 1);
		}
	}

	template <int N> struct OneBasedMatrix {
		DenseMatrix<N> &matrix;
		AMREX_GPU_HOST_DEVICE void zero() noexcept { matrix.zero(); }
		[[nodiscard]] AMREX_GPU_HOST_DEVICE auto operator()(int row, int column) noexcept -> amrex::Real & { return matrix(row - 1, column - 1); }
	};

	AMREX_GPU_HOST_DEVICE void jacobian(IntegratorState<variable_count> const &state, amrex::Real /*time*/,
					    DenseMatrix<variable_count> &matrix) const noexcept
	{
		auto generated = generated_state(state);
		OneBasedMatrix<variable_count> adapter{matrix};
		primordial_detail::actual_jac(generated, adapter, parameters.redshift);
	}

	AMREX_GPU_HOST_DEVICE static void clean(IntegratorState<variable_count> &state, IntegratorOptions const &options) noexcept
	{
		for (int species = 0; species < species_count; ++species) {
			state.values[species] = amrex::max(state.values[species], options.small_state);
		}
		update_thermodynamics(state);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(IntegratorState<variable_count> const &state, IntegratorOptions const &options) noexcept -> bool
	{
		for (int variable = 0; variable < variable_count; ++variable) {
			if (!std::isfinite(state.values[variable])) {
				return false;
			}
		}
		for (int species = 0; species < species_count; ++species) {
			if (state.values[species] < -options.atol_species) {
				return false;
			}
		}
		return state.values[energy] > 0.0;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_update(IntegratorState<variable_count> const &old_state,
								     IntegratorState<variable_count> const &new_state,
								     IntegratorOptions const &options) noexcept -> bool
	{
		for (int species = 0; species < species_count; ++species) {
			if (std::abs(old_state.values[species]) > options.rejection_buffer * options.atol_species &&
			    std::abs(new_state.values[species]) > options.rejection_buffer * options.atol_species &&
			    (std::abs(new_state.values[species]) > 4.0 * std::abs(old_state.values[species]) ||
			     std::abs(new_state.values[species]) < 0.25 * std::abs(old_state.values[species]))) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(IntegratorState<variable_count> const &state, IntegratorOptions const &options) noexcept
	    -> bool
	{
		for (int species = 0; species < species_count; ++species) {
			if (state.values[species] < -options.species_failure_tolerance) {
				return false;
			}
		}
		return true;
	}

	AMREX_GPU_HOST_DEVICE static void balance_charge(IntegratorState<variable_count> &state) noexcept
	{
		state.values[0] = -state.values[3] - state.values[7] + state.values[1] + state.values[12] + state.values[6] + state.values[4] +
				  state.values[9] + 2.0 * state.values[11];
	}
};

} // namespace quokka::chemistry

#endif
