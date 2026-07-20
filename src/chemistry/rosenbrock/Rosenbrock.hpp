#ifndef QUOKKA_CHEMISTRY_ROSENBROCK_HPP_
#define QUOKKA_CHEMISTRY_ROSENBROCK_HPP_

#include <cmath>
#include <limits>

#include "AMReX_Algorithm.H"
#include "chemistry/ChemistryNetwork.hpp"
#include "chemistry/rosenbrock/LinearSolver.hpp"
#include "chemistry/rosenbrock/Tableau.hpp"

namespace quokka::chemistry::rosenbrock
{

template <typename Network> struct Workspace {
	static constexpr int variable_count = Network::variable_count;
	IntegratorState<variable_count> state{};
	amrex::GpuArray<amrex::GpuArray<amrex::Real, variable_count>, 8> stages{};
	amrex::GpuArray<amrex::Real, variable_count> candidate{};
	amrex::GpuArray<amrex::Real, variable_count> work{};
	DenseMatrix<variable_count> jacobian{};
	amrex::GpuArray<short, variable_count> pivots{};
};

template <typename Network>
AMREX_GPU_HOST_DEVICE auto tolerance(Network const &network, IntegratorOptions const &options, int variable, bool relative) noexcept -> amrex::Real
{
	switch (network.variable_role(variable)) {
		case VariableRole::species:
			return relative ? options.rtol_species : options.atol_species;
		case VariableRole::energy:
			return relative ? options.rtol_energy : options.atol_energy;
		case VariableRole::radiation_number:
			return relative ? options.rtol_radiation : options.atol_radiation;
		case VariableRole::passive:
			return 1.0e30;
	}
	return 1.0e30;
}

template <typename Network>
AMREX_GPU_HOST_DEVICE auto exceeds_maximum_temperature(Network const &network, IntegratorState<Network::variable_count> const &state,
						       IntegratorOptions const &options) noexcept -> bool
{
	return network.temperature(state) >= options.maximum_temperature;
}

template <typename Network>
AMREX_GPU_HOST_DEVICE void evaluate_rhs(Network const &network, IntegratorState<Network::variable_count> const &state, amrex::Real time,
					IntegratorOptions const &options, amrex::GpuArray<amrex::Real, Network::variable_count> &derivative) noexcept
{
	derivative.fill(0.0);
	if (!exceeds_maximum_temperature(network, state, options)) {
		network.rhs(state, time, derivative);
	}
}

template <typename Network>
AMREX_GPU_HOST_DEVICE auto error_norm(Network const &network, IntegratorOptions const &options, Workspace<Network> const &workspace) noexcept -> amrex::Real
{
	amrex::Real sum = 0.0;
	for (int n = 0; n < Network::variable_count; ++n) {
		if (!network.controls_error(n)) {
			continue;
		}
		const amrex::Real scale =
		    tolerance(network, options, n, false) +
		    tolerance(network, options, n, true) * amrex::max(std::abs(workspace.state.values[n]), std::abs(workspace.candidate[n]));
		const amrex::Real term = workspace.work[n] / scale;
		sum += term * term;
	}
	// Preserve the upstream norm convention: passive variables contribute a
	// zero term but remain part of the system-size normalization.
	return std::sqrt(sum / static_cast<amrex::Real>(Network::variable_count));
}

template <typename Network>
AMREX_GPU_HOST_DEVICE void numerical_jacobian(Network const &network, IntegratorOptions const &options, Workspace<Network> &workspace, amrex::Real time,
					      IntegratorDiagnostics &diagnostics) noexcept
{
	if (exceeds_maximum_temperature(network, workspace.state, options)) {
		workspace.jacobian.zero();
		++diagnostics.jacobian_evaluations;
		return;
	}
	constexpr amrex::Real roundoff = std::numeric_limits<amrex::Real>::epsilon();
	auto const baseline_state = workspace.state;
	amrex::GpuArray<amrex::Real, Network::variable_count> baseline_rhs{};
	evaluate_rhs(network, baseline_state, time, options, baseline_rhs);
	++diagnostics.rhs_evaluations;
	for (int column = 0; column < Network::variable_count; ++column) {
		workspace.state = baseline_state;
		const amrex::Real delta =
		    amrex::max(std::sqrt(roundoff) * std::abs(baseline_state.values[column]), tolerance(network, options, column, false) * std::sqrt(roundoff));
		workspace.state.values[column] += delta;
		amrex::GpuArray<amrex::Real, Network::variable_count> perturbed_rhs{};
		evaluate_rhs(network, workspace.state, time, options, perturbed_rhs);
		++diagnostics.rhs_evaluations;
		for (int row = 0; row < Network::variable_count; ++row) {
			workspace.jacobian(row, column) = (perturbed_rhs[row] - baseline_rhs[row]) / delta;
		}
	}
	workspace.state = baseline_state;
	++diagnostics.jacobian_evaluations;
}

template <typename Tableau, typename Network>
AMREX_GPU_HOST_DEVICE auto integrate_with_tableau(Network const &network, IntegratorState<Network::variable_count> &state, amrex::Real const timestep,
						  IntegratorOptions const &options) noexcept -> IntegratorDiagnostics
{
	IntegratorDiagnostics diagnostics{};
	if (!(timestep >= 0.0) || !(options.maximum_timestep > 0.0) || !(options.maximum_temperature > 0.0) || options.maximum_steps <= 0 ||
	    !(options.controller_b > 0.0) || !(options.controller_k > 0.0)) {
		diagnostics.status = IntegratorStatus::bad_inputs;
		return diagnostics;
	}
	for (int variable = 0; variable < Network::variable_count; ++variable) {
		if (network.controls_error(variable) && (!(tolerance(network, options, variable, false) > 0.0) ||
							 !(tolerance(network, options, variable, true) > 10.0 * std::numeric_limits<amrex::Real>::epsilon()))) {
			diagnostics.status = IntegratorStatus::accuracy_unattainable;
			return diagnostics;
		}
	}
	if (timestep == 0.0) {
		return diagnostics;
	}

	Workspace<Network> workspace{};
	workspace.state = state;
	amrex::Real time = 0.0;
	amrex::Real step = amrex::min(timestep, options.maximum_timestep);
	amrex::Real previous_error = 1.0;
	amrex::Real previous_factor = 1.0;
	bool rejected = false;
	int linear_solve_failures = 0;

	while (time < timestep) {
		if (diagnostics.steps >= options.maximum_steps) {
			diagnostics.status = IntegratorStatus::too_many_steps;
			break;
		}
		constexpr amrex::Real endpointSafetyFactor = 1.0e-4;
		const amrex::Real remaining = timestep - time;
		const bool reachesEndpoint = step * (1.0 + endpointSafetyFactor) >= remaining;
		step = reachesEndpoint ? remaining : amrex::min(step, remaining);
		if (0.1 * std::abs(step) <= std::abs(time) * std::numeric_limits<amrex::Real>::epsilon()) {
			diagnostics.status = IntegratorStatus::timestep_underflow;
			break;
		}

		if (options.analytic_jacobian) {
			if (exceeds_maximum_temperature(network, workspace.state, options)) {
				workspace.jacobian.zero();
			} else {
				network.jacobian(workspace.state, time, workspace.jacobian);
			}
			++diagnostics.jacobian_evaluations;
		} else {
			numerical_jacobian(network, options, workspace, time, diagnostics);
		}
		for (int row = 0; row < Network::variable_count; ++row) {
			for (int column = 0; column < Network::variable_count; ++column) {
				workspace.jacobian(row, column) = -workspace.jacobian(row, column);
			}
			workspace.jacobian(row, row) += 1.0 / (step * Tableau::gamma);
		}
		if (!factor<Network::variable_count>(workspace.jacobian, workspace.pivots, options.pivoting)) {
			step *= 0.5;
			rejected = true;
			++diagnostics.rejected_steps;
			++linear_solve_failures;
			if (linear_solve_failures >= 5) {
				diagnostics.status = IntegratorStatus::linear_solve_failure;
				break;
			}
			continue;
		}

		evaluate_rhs(network, workspace.state, time, options, workspace.stages[0]);
		++diagnostics.rhs_evaluations;
		solve<Network::variable_count>(workspace.jacobian, workspace.pivots, workspace.stages[0]);
		bool valid_stage = true;
		for (int stage = 1; stage < Tableau::stages; ++stage) {
			auto stage_state = workspace.state;
			workspace.stages[stage].fill(0.0);
			for (int n = 0; n < Network::variable_count; ++n) {
				for (int prior = 0; prior < stage; ++prior) {
					stage_state.values[n] += Tableau::a(stage, prior) * workspace.stages[prior][n];
					workspace.stages[stage][n] += Tableau::c(stage, prior) * workspace.stages[prior][n] / step;
				}
			}
			if (!network.valid(stage_state, options)) {
				valid_stage = false;
				break;
			}
			network.clean(stage_state, options);
			amrex::GpuArray<amrex::Real, Network::variable_count> stage_rhs{};
			evaluate_rhs(network, stage_state, time + Tableau::ctime(stage) * step, options, stage_rhs);
			++diagnostics.rhs_evaluations;
			for (int n = 0; n < Network::variable_count; ++n) {
				workspace.stages[stage][n] += stage_rhs[n];
			}
			solve<Network::variable_count>(workspace.jacobian, workspace.pivots, workspace.stages[stage]);
		}
		if (!valid_stage) {
			step *= 0.25;
			rejected = true;
			++diagnostics.rejected_steps;
			continue;
		}

		for (int n = 0; n < Network::variable_count; ++n) {
			workspace.candidate[n] = workspace.state.values[n];
			workspace.work[n] = 0.0;
			for (int stage = 0; stage < Tableau::stages; ++stage) {
				workspace.candidate[n] += Tableau::b(stage) * workspace.stages[stage][n];
				workspace.work[n] += Tableau::error(stage) * workspace.stages[stage][n];
			}
		}
		++diagnostics.steps;
		auto candidate_state = workspace.state;
		candidate_state.values = workspace.candidate;
		const amrex::Real error = error_norm(network, options, workspace);
		const bool valid = network.valid(candidate_state, options) && network.valid_update(workspace.state, candidate_state, options);
		const amrex::Real bounded_error = amrex::max(error, 1.0e-10);
		const amrex::Real controller =
		    amrex::Clamp(std::pow(1.0 / bounded_error, 1.0 / (options.controller_b * options.controller_k)) *
				     std::pow(1.0 / amrex::max(previous_error, 1.0e-10), 1.0 / (options.controller_b * options.controller_k)) *
				     std::pow(previous_factor, -1.0 / options.controller_b),
				 options.controller_minimum, options.controller_maximum);
		previous_factor = controller;
		previous_error = error;
		amrex::Real next_step = step * controller;
		if (error <= 1.0 && valid) {
			workspace.state = candidate_state;
			time += step;
			if (reachesEndpoint) {
				time = timestep;
				break;
			}
			if (rejected) {
				next_step = amrex::min(next_step, step);
			}
			rejected = false;
			step = amrex::min(next_step, options.maximum_timestep);
			continue;
		}
		rejected = true;
		++diagnostics.rejected_steps;
		step = valid ? amrex::min(next_step, options.controller_reduction * step) : 0.25 * step;
	}

	diagnostics.reached_time = time;
	diagnostics.suggested_timestep = step;
	if (diagnostics.status == IntegratorStatus::success) {
		state = workspace.state;
		if (!network.valid_final(state, options)) {
			diagnostics.status = IntegratorStatus::invalid_state;
		} else {
			network.clean(state, options);
		}
	}
	return diagnostics;
}

template <typename Network>
AMREX_GPU_HOST_DEVICE auto integrate(Network const &network, IntegratorState<Network::variable_count> &state, amrex::Real const timestep,
				     IntegratorOptions const &options) noexcept -> IntegratorDiagnostics
{
	if (options.tableau == 0) {
		return integrate_with_tableau<Rodas5p>(network, state, timestep, options);
	}
	if (options.tableau == 3) {
		return integrate_with_tableau<Ros2s>(network, state, timestep, options);
	}
	IntegratorDiagnostics diagnostics{};
	diagnostics.status = IntegratorStatus::bad_inputs;
	return diagnostics;
}

template <typename Network>
AMREX_GPU_HOST_DEVICE auto integrate_with_retry(Network const &network, IntegratorState<Network::variable_count> &state, amrex::Real const timestep,
						IntegratorOptions const &options) noexcept -> IntegratorDiagnostics
{
	auto const original_state = state;
	auto diagnostics = integrate(network, state, timestep, options);
	if (diagnostics.succeeded() || !options.retry_enabled) {
		return diagnostics;
	}
	state = original_state;
	auto retry = options;
	retry.retry_enabled = false;
	if (retry.retry_swap_jacobian) {
		retry.analytic_jacobian = !retry.analytic_jacobian;
	}
	if (retry.retry_rtol_species > 0.0) {
		retry.rtol_species = retry.retry_rtol_species;
	}
	if (retry.retry_atol_species > 0.0) {
		retry.atol_species = retry.retry_atol_species;
	}
	if (retry.retry_rtol_energy > 0.0) {
		retry.rtol_energy = retry.retry_rtol_energy;
	}
	if (retry.retry_atol_energy > 0.0) {
		retry.atol_energy = retry.retry_atol_energy;
	}
	if (retry.retry_rtol_radiation > 0.0) {
		retry.rtol_radiation = retry.retry_rtol_radiation;
	}
	if (retry.retry_atol_radiation > 0.0) {
		retry.atol_radiation = retry.retry_atol_radiation;
	}
	return integrate(network, state, timestep, retry);
}

} // namespace quokka::chemistry::rosenbrock

#endif
