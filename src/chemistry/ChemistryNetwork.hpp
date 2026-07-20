#ifndef QUOKKA_CHEMISTRY_NETWORK_HPP_
#define QUOKKA_CHEMISTRY_NETWORK_HPP_

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::chemistry
{

enum class VariableRole : std::uint8_t { species, energy, radiation_number, passive };

struct VariableMetadata {
	std::string_view name;
	VariableRole role;
	bool controls_error;
};

struct IntegratorOptions {
	amrex::Real rtol_species = 1.0e-10;
	amrex::Real atol_species = 1.0e-10;
	amrex::Real rtol_energy = 1.0e-10;
	amrex::Real atol_energy = 1.0e-25;
	amrex::Real rtol_radiation = 1.0e-10;
	amrex::Real atol_radiation = 1.0e-10;
	amrex::Real species_failure_tolerance = 1.0e-2;
	amrex::Real radiation_failure_tolerance = 1.0e-2;
	amrex::Real small_state = 1.0e-30;
	amrex::Real maximum_timestep = 1.0e30;
	amrex::Real minimum_temperature = 0.0;
	amrex::Real maximum_temperature = 1.0e11;
	amrex::Real rejection_buffer = 1.0;
	amrex::Real controller_minimum = 0.2;
	amrex::Real controller_maximum = 6.0;
	amrex::Real controller_reduction = 0.5;
	amrex::Real controller_b = 4.0;
	amrex::Real controller_k = 2.5;
	int maximum_steps = 150000;
	int tableau = 0;
	bool analytic_jacobian = true;
	bool pivoting = true;
	bool retry_enabled = false;
	bool retry_swap_jacobian = true;
	amrex::Real retry_rtol_species = -1.0;
	amrex::Real retry_atol_species = -1.0;
	amrex::Real retry_rtol_energy = -1.0;
	amrex::Real retry_atol_energy = -1.0;
	amrex::Real retry_rtol_radiation = -1.0;
	amrex::Real retry_atol_radiation = -1.0;
};

enum class IntegratorStatus : std::int8_t {
	success = 1,
	bad_inputs = -1,
	timestep_underflow = -2,
	too_many_steps = -4,
	accuracy_unattainable = -5,
	linear_solve_failure = -7,
	invalid_state = -8,
};

struct IntegratorDiagnostics {
	IntegratorStatus status = IntegratorStatus::success;
	int steps = 0;
	int rhs_evaluations = 0;
	int jacobian_evaluations = 0;
	int rejected_steps = 0;
	amrex::Real reached_time = 0.0;
	amrex::Real suggested_timestep = 0.0;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto succeeded() const noexcept -> bool { return status == IntegratorStatus::success; }
};

template <int N> struct IntegratorState {
	amrex::GpuArray<amrex::Real, N> values{};
	amrex::Real density = 0.0;
	amrex::Real temperature = 0.0;
	amrex::Real energy_scale = 1.0;
	amrex::Real reduced_speed_of_light = 0.0;
};

template <int N> struct DenseMatrix {
	amrex::GpuArray<amrex::Real, N * N> values{};

	AMREX_GPU_HOST_DEVICE void zero() noexcept
	{
		for (auto &value : values) {
			value = 0.0;
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto operator()(int row, int column) noexcept -> amrex::Real & { return values[row * N + column]; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto operator()(int row, int column) const noexcept -> amrex::Real { return values[row * N + column]; }
};

struct ChemistryNetworkMetadata {
	std::string_view name;
	std::string_view version;
	int species_count;
	int variable_count;
};

class ChemistryNetworkRegistry
{
      public:
	template <typename Network> void add()
	{
		if (find(Network::metadata.name) != nullptr) {
			throw std::invalid_argument("chemistry network is already registered");
		}
		networks_.push_back(Network::metadata);
	}

	[[nodiscard]] auto find(std::string_view name) const noexcept -> ChemistryNetworkMetadata const *
	{
		for (auto const &network : networks_) {
			if (network.name == name) {
				return &network;
			}
		}
		return nullptr;
	}

	[[nodiscard]] auto available() const noexcept -> std::vector<ChemistryNetworkMetadata> const & { return networks_; }

      private:
	std::vector<ChemistryNetworkMetadata> networks_{};
};

} // namespace quokka::chemistry

#endif
