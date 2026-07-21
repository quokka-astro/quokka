#ifndef QUOKKA_PHOTOIONIZATION_NETWORK_HPP_
#define QUOKKA_PHOTOIONIZATION_NETWORK_HPP_

#include <cmath>
#include <string_view>

#include "AMReX_ParmParse.H"
#include "chemistry/ChemistryNetwork.hpp"

namespace quokka::chemistry
{

struct PhotoionizationParameters {
	int recombination = 1;
	int temperature_dependent_recombination = 1;
	int collisional_ionization = 1;
	int energy = 1;
	int photoheating = 1;
	int ki_heating = 1;
	int recombination_cooling = 1;
	int ki_cooling = 1;
	int ion_free_free_cooling = 1;
	int allow_mixed_cell_cooling = 1;
	amrex::Real mixed_cell_lower_bound = 0.01;
	amrex::Real mixed_cell_upper_bound = 0.99;
};

[[nodiscard]] inline auto readPhotoionizationParameters() -> PhotoionizationParameters
{
	PhotoionizationParameters values{};
	amrex::ParmParse const parameters("network");
	parameters.query("recombination_switch", values.recombination);
	parameters.query("recombination_temperature_dependent", values.temperature_dependent_recombination);
	parameters.query("collisional_ionization_switch", values.collisional_ionization);
	parameters.query("energy_switch", values.energy);
	parameters.query("photoheating_switch", values.photoheating);
	parameters.query("KI_heating_switch", values.ki_heating);
	parameters.query("recombination_cooling_switch", values.recombination_cooling);
	parameters.query("KI_cooling_switch", values.ki_cooling);
	parameters.query("ion_ff_cooling_switch", values.ion_free_free_cooling);
	parameters.query("allow_mixed_cell_cooling", values.allow_mixed_cell_cooling);
	parameters.query("mixed_cell_lb", values.mixed_cell_lower_bound);
	parameters.query("mixed_cell_ub", values.mixed_cell_upper_bound);
	return values;
}

struct PhotoionizationNetwork {
	static constexpr amrex::Real boltzmann_constant = 1.380649e-16;
	static constexpr int species_count = 3;
	static constexpr int variable_count = 6;
	static constexpr int electron = 0;
	static constexpr int neutral_hydrogen = 1;
	static constexpr int ionized_hydrogen = 2;
	static constexpr int energy = 3;
	static constexpr int photon_number = 4;
	static constexpr int photon_flux_factor = 5;
	static constexpr int radiation_variables_per_group = 2;
	static constexpr int chemistry_band_count = 1;
	static constexpr amrex::GpuArray<amrex::Real, 2> chemistry_band_edges = {3.29e15, 1.50e16};
	static constexpr amrex::GpuArray<amrex::Real, species_count> species_masses = {9.10938291e-28, 1.673532715291e-24, 1.672621777e-24};
	static constexpr amrex::GpuArray<amrex::Real, species_count> species_gammas = {5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0};
	static constexpr std::array<std::string_view, species_count> species_names = {"electron", "neutral_hydrogen", "ionized_hydrogen"};
	static constexpr std::array<VariableMetadata, variable_count> variables = {
	    VariableMetadata{"electron", VariableRole::species, true},
	    VariableMetadata{"neutral_hydrogen", VariableRole::species, true},
	    VariableMetadata{"ionized_hydrogen", VariableRole::species, true},
	    VariableMetadata{"specific_internal_energy", VariableRole::energy, true},
	    VariableMetadata{"photon_number", VariableRole::radiation_number, true},
	    VariableMetadata{"photon_flux_attenuation", VariableRole::passive, false},
	};
	static constexpr ChemistryNetworkMetadata metadata = {"photoionization", "1", species_count, variable_count};

	PhotoionizationParameters parameters{};

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int variable) noexcept -> VariableRole
	{
		if (variable < species_count) {
			return VariableRole::species;
		}
		if (variable == energy) {
			return VariableRole::energy;
		}
		return variable == photon_number ? VariableRole::radiation_number : VariableRole::passive;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int variable) noexcept -> bool { return variable != photon_flux_factor; }

	AMREX_GPU_HOST_DEVICE static void update_thermodynamics(IntegratorState<variable_count> &state) noexcept
	{
		amrex::Real mass_density = 0.0;
		amrex::Real heat_capacity = 0.0;
		for (int species = 0; species < species_count; ++species) {
			mass_density += state.values[species] * species_masses[species];
			heat_capacity += state.values[species] / (species_gammas[species] - 1.0);
		}
		if (mass_density > 0.0 && heat_capacity > 0.0) {
			state.temperature = state.values[energy] * mass_density / (heat_capacity * boltzmann_constant);
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
		amrex::Real mass_density = 0.0;
		amrex::Real heat_capacity = 0.0;
		for (int species = 0; species < species_count; ++species) {
			mass_density += number_density[species] * species_masses[species];
			heat_capacity += number_density[species] / (species_gammas[species] - 1.0);
		}
		return heat_capacity * boltzmann_constant * temperature / mass_density;
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto recombination_coefficient(amrex::Real temperature, int temperature_dependent) noexcept -> amrex::Real
	{
		return temperature_dependent == 0 ? 2.6e-13 : 2.6e-13 * std::pow(temperature / 1.0e4, -0.7);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto recombination_derivative(amrex::Real temperature, int temperature_dependent) noexcept -> amrex::Real
	{
		return temperature_dependent == 0 ? 0.0 : -2.6e-13 * 0.7 * std::pow(temperature / 1.0e4, -1.7) / 1.0e4;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto collisional_coefficient(amrex::Real temperature) noexcept -> amrex::Real
	{
		return 5.84e-11 * std::sqrt(temperature) * std::exp(-2.18e-11 / (boltzmann_constant * temperature));
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto collisional_derivative(amrex::Real temperature) noexcept -> amrex::Real
	{
		const amrex::Real exponential = std::exp(-2.18e-11 / (boltzmann_constant * temperature));
		return 5.84e-11 * 0.5 / std::sqrt(temperature) * exponential +
		       5.84e-11 * std::sqrt(temperature) * exponential * 2.18e-11 / (boltzmann_constant * temperature * temperature);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ki_cooling_coefficient(amrex::Real temperature) noexcept -> amrex::Real
	{
		return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (temperature + 1.0e3)) + 1.4e-2 * std::sqrt(temperature) * std::exp(-92.0 / temperature));
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto ki_cooling_derivative(amrex::Real temperature) noexcept -> amrex::Real
	{
		return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (temperature + 1.0e3)) * 118400.0 / ((temperature + 1.0e3) * (temperature + 1.0e3)) +
				  1.4e-2 * (0.5 / std::sqrt(temperature) * std::exp(-92.0 / temperature) +
					    std::sqrt(temperature) * std::exp(-92.0 / temperature) * 92.0 / (temperature * temperature)));
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto recombination_cooling_coefficient(amrex::Real temperature) noexcept -> amrex::Real
	{
		return temperature < 100.0 ? 0.0 : 6.1e-10 * boltzmann_constant * temperature * std::pow(temperature, -0.89);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto recombination_cooling_derivative(amrex::Real temperature) noexcept -> amrex::Real
	{
		return temperature < 100.0 ? 0.0
					   : 6.1e-10 * boltzmann_constant * (std::pow(temperature, -0.89) - 0.89 * temperature * std::pow(temperature, -1.89));
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto free_free_coefficient(amrex::Real temperature) noexcept -> amrex::Real
	{
		return 1.4e-27 * std::sqrt(temperature) + 1.0e-19 * std::exp(-118348.0 / temperature);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto free_free_derivative(amrex::Real temperature) noexcept -> amrex::Real
	{
		return 1.4e-27 * 0.5 / std::sqrt(temperature) + 1.0e-19 * std::exp(-118348.0 / temperature) * 118348.0 / (temperature * temperature);
	}

	AMREX_GPU_HOST_DEVICE void rhs(IntegratorState<variable_count> state, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		update_thermodynamics(state);
		const amrex::Real temperature = state.temperature;
		const amrex::Real photo_rate = state.reduced_speed_of_light * 1.5e-18;
		const amrex::Real electron_density = state.values[electron];
		const amrex::Real neutral_density = state.values[neutral_hydrogen];
		const amrex::Real ionized_density = state.values[ionized_hydrogen];
		const amrex::Real photons = amrex::max(state.values[photon_number], 0.0);
		const amrex::Real photoionization = photo_rate * neutral_density * photons;
		const amrex::Real collisional =
		    parameters.collisional_ionization != 0 ? electron_density * neutral_density * collisional_coefficient(temperature) : 0.0;
		const amrex::Real recombination =
		    parameters.recombination != 0
			? recombination_coefficient(temperature, parameters.temperature_dependent_recombination) * electron_density * ionized_density
			: 0.0;
		derivative[electron] = photoionization + collisional - recombination;
		derivative[neutral_hydrogen] = -derivative[electron];
		derivative[ionized_hydrogen] = derivative[electron];
		derivative[energy] = 0.0;
		const amrex::Real total_hydrogen = neutral_density + ionized_density;
		const amrex::Real ionized_fraction = total_hydrogen > 0.0 ? ionized_density / total_hydrogen : 0.0;
		const bool molecular = parameters.allow_mixed_cell_cooling != 0 || ionized_fraction <= parameters.mixed_cell_lower_bound ||
				       ionized_fraction >= parameters.mixed_cell_upper_bound;
		if (parameters.energy != 0) {
			if (parameters.photoheating != 0) {
				derivative[energy] += 6.4e-12 * photoionization;
			}
			if (molecular && parameters.ki_heating != 0) {
				derivative[energy] += 2.0e-26 * neutral_density;
			}
			if (molecular && parameters.ki_cooling != 0) {
				derivative[energy] -= ki_cooling_coefficient(temperature) * neutral_density * neutral_density;
			}
			if (parameters.recombination_cooling != 0) {
				derivative[energy] -= recombination_cooling_coefficient(temperature) * electron_density * ionized_density;
			}
			if (parameters.ion_free_free_cooling != 0) {
				derivative[energy] -= free_free_coefficient(temperature) * electron_density * ionized_density;
			}
			derivative[energy] /= state.density;
		}
		derivative[photon_number] = -photoionization;
		derivative[photon_flux_factor] = -photo_rate * neutral_density * state.values[photon_flux_factor];
	}

	AMREX_GPU_HOST_DEVICE void jacobian(IntegratorState<variable_count> state, amrex::Real /*time*/, DenseMatrix<variable_count> &matrix) const noexcept
	{
		update_thermodynamics(state);
		matrix.zero();
		const amrex::Real temperature = state.temperature;
		const amrex::Real photo_rate = state.reduced_speed_of_light * 1.5e-18;
		const amrex::Real ne = state.values[electron];
		const amrex::Real nhi = state.values[neutral_hydrogen];
		const amrex::Real nhii = state.values[ionized_hydrogen];
		const amrex::Real photons = amrex::max(state.values[photon_number], 0.0);
		const amrex::Real photon_active = state.values[photon_number] >= 0.0 ? 1.0 : 0.0;
		const amrex::Real collisional = parameters.collisional_ionization != 0 ? collisional_coefficient(temperature) : 0.0;
		const amrex::Real dcollisional = parameters.collisional_ionization != 0 ? collisional_derivative(temperature) : 0.0;
		const amrex::Real recombination =
		    parameters.recombination != 0 ? recombination_coefficient(temperature, parameters.temperature_dependent_recombination) : 0.0;
		const amrex::Real drecombination =
		    parameters.recombination != 0 ? recombination_derivative(temperature, parameters.temperature_dependent_recombination) : 0.0;
		const amrex::Real mass_density =
		    ne * species_masses[electron] + nhi * species_masses[neutral_hydrogen] + nhii * species_masses[ionized_hydrogen];
		const amrex::Real heat_capacity =
		    ne / (species_gammas[electron] - 1.0) + nhi / (species_gammas[neutral_hydrogen] - 1.0) + nhii / (species_gammas[ionized_hydrogen] - 1.0);
		const amrex::Real dtemperature_denergy = mass_density / (heat_capacity * boltzmann_constant);
		matrix(electron, electron) = nhi * collisional - recombination * nhii;
		matrix(electron, neutral_hydrogen) = photo_rate * photons + ne * collisional;
		matrix(electron, ionized_hydrogen) = -recombination * ne;
		matrix(electron, energy) = (ne * nhi * dcollisional - drecombination * ne * nhii) * dtemperature_denergy;
		matrix(electron, photon_number) = photon_active * photo_rate * nhi;
		for (int column = 0; column < variable_count; ++column) {
			matrix(neutral_hydrogen, column) = -matrix(electron, column);
			matrix(ionized_hydrogen, column) = matrix(electron, column);
		}

		const amrex::Real total_hydrogen = nhi + nhii;
		const amrex::Real ionized_fraction = total_hydrogen > 0.0 ? nhii / total_hydrogen : 0.0;
		const bool molecular = parameters.allow_mixed_cell_cooling != 0 || ionized_fraction <= parameters.mixed_cell_lower_bound ||
				       ionized_fraction >= parameters.mixed_cell_upper_bound;
		const amrex::Real ki_cooling = parameters.ki_cooling != 0 ? ki_cooling_coefficient(temperature) : 0.0;
		const amrex::Real dki_cooling = parameters.ki_cooling != 0 ? ki_cooling_derivative(temperature) : 0.0;
		const amrex::Real rec_cooling = parameters.recombination_cooling != 0 ? recombination_cooling_coefficient(temperature) : 0.0;
		const amrex::Real drec_cooling = parameters.recombination_cooling != 0 ? recombination_cooling_derivative(temperature) : 0.0;
		const amrex::Real ff_cooling = parameters.ion_free_free_cooling != 0 ? free_free_coefficient(temperature) : 0.0;
		const amrex::Real dff_cooling = parameters.ion_free_free_cooling != 0 ? free_free_derivative(temperature) : 0.0;
		if (parameters.energy != 0) {
			matrix(energy, electron) =
			    (-parameters.recombination_cooling * rec_cooling * nhii - parameters.ion_free_free_cooling * ff_cooling * nhii) / state.density;
			matrix(energy, neutral_hydrogen) =
			    (parameters.photoheating * 6.4e-12 * photo_rate * photons + (molecular ? parameters.ki_heating * 2.0e-26 : 0.0) -
			     (molecular ? 2.0 * parameters.ki_cooling * ki_cooling * nhi : 0.0)) /
			    state.density;
			matrix(energy, ionized_hydrogen) =
			    (-parameters.recombination_cooling * rec_cooling * ne - parameters.ion_free_free_cooling * ff_cooling * ne) / state.density;
			matrix(energy, energy) =
			    (-parameters.recombination_cooling * drec_cooling * ne * nhii - parameters.ion_free_free_cooling * dff_cooling * ne * nhii -
			     (molecular ? parameters.ki_cooling * dki_cooling * nhi * nhi : 0.0)) *
			    dtemperature_denergy / state.density;
			matrix(energy, photon_number) = parameters.photoheating * 6.4e-12 * photo_rate * nhi * photon_active / state.density;
		}
		matrix(photon_number, neutral_hydrogen) = -photo_rate * photons;
		matrix(photon_number, photon_number) = -photon_active * photo_rate * nhi;
		matrix(photon_flux_factor, neutral_hydrogen) = -photo_rate * state.values[photon_flux_factor];
		matrix(photon_flux_factor, photon_flux_factor) = -photo_rate * nhi;
	}

	AMREX_GPU_HOST_DEVICE static void clean(IntegratorState<variable_count> &state, IntegratorOptions const &options) noexcept
	{
		for (int variable = 0; variable < species_count; ++variable) {
			state.values[variable] = amrex::max(state.values[variable], options.small_state);
		}
		state.values[photon_number] = amrex::max(state.values[photon_number], options.small_state);
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
		return state.values[energy] > 0.0 && state.values[photon_number] >= -options.atol_radiation;
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
		return state.values[photon_number] >= -options.radiation_failure_tolerance;
	}
};

} // namespace quokka::chemistry

#endif
