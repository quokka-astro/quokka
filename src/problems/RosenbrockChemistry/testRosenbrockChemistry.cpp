#include <cmath>
#include <limits>

#include "AMReX.H"
#include "chemistry/ChemistryNetwork.hpp"
#include "chemistry/rosenbrock/Rosenbrock.hpp"
#include "networks/photoionization/PhotoionizationNetwork.hpp"
#include "networks/primordial_chem/PrimordialChemNetwork.hpp"

namespace
{

struct DecayNetwork {
	static constexpr int variable_count = 2;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int variable) noexcept -> quokka::chemistry::VariableRole
	{
		return variable == 0 ? quokka::chemistry::VariableRole::species : quokka::chemistry::VariableRole::passive;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int variable) noexcept -> bool { return variable == 0; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto temperature(quokka::chemistry::IntegratorState<variable_count> const & /*state*/) noexcept
	    -> amrex::Real
	{
		return 0.0;
	}

	AMREX_GPU_HOST_DEVICE void rhs(quokka::chemistry::IntegratorState<variable_count> const &state, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		derivative[0] = -state.values[0];
		derivative[1] = -100.0 * state.values[1];
	}

	AMREX_GPU_HOST_DEVICE void jacobian(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real /*time*/,
					    quokka::chemistry::DenseMatrix<variable_count> &matrix) const noexcept
	{
		matrix.zero();
		matrix(0, 0) = -1.0;
		matrix(1, 1) = -100.0;
	}

	AMREX_GPU_HOST_DEVICE static void clean(quokka::chemistry::IntegratorState<variable_count> & /*state*/,
						quokka::chemistry::IntegratorOptions const & /*options*/) noexcept
	{
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(quokka::chemistry::IntegratorState<variable_count> const &state,
							      quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return std::isfinite(state.values[0]) && std::isfinite(state.values[1]);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_update(quokka::chemistry::IntegratorState<variable_count> const & /*old_state*/,
								     quokka::chemistry::IntegratorState<variable_count> const & /*new_state*/,
								     quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return true;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(quokka::chemistry::IntegratorState<variable_count> const &state,
								    quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return valid(state, options);
	}
};

struct StageValidationNetwork {
	static constexpr int variable_count = 1;
	// Records whether the integrator cleaned a stage that should have been rejected.
	mutable bool cleaned_invalid_stage = false;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int /*variable*/) noexcept -> quokka::chemistry::VariableRole
	{
		return quokka::chemistry::VariableRole::species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int /*variable*/) noexcept -> bool { return true; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto temperature(quokka::chemistry::IntegratorState<variable_count> const & /*state*/) noexcept
	    -> amrex::Real
	{
		return 0.0;
	}

	AMREX_GPU_HOST_DEVICE void rhs(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		derivative[0] = -10.0;
	}

	AMREX_GPU_HOST_DEVICE void jacobian(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real /*time*/,
					    quokka::chemistry::DenseMatrix<variable_count> &matrix) const noexcept
	{
		matrix.zero();
	}

	AMREX_GPU_HOST_DEVICE void clean(quokka::chemistry::IntegratorState<variable_count> &state,
					 quokka::chemistry::IntegratorOptions const &options) const noexcept
	{
		if (!valid(state, options)) {
			cleaned_invalid_stage = true;
		}
		state.values[0] = amrex::max(state.values[0], options.small_state);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(quokka::chemistry::IntegratorState<variable_count> const &state,
							      quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return std::isfinite(state.values[0]) && state.values[0] >= -options.atol_species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto valid_update(quokka::chemistry::IntegratorState<variable_count> const & /*old_state*/,
									       quokka::chemistry::IntegratorState<variable_count> const & /*new_state*/,
									       quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return true;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(quokka::chemistry::IntegratorState<variable_count> const &state,
								    quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return valid(state, options);
	}
};

struct RetryAccountingNetwork {
	static constexpr int variable_count = 1;
	// The first four calls produce invalid stages; the fifth produces a singular
	// factorization that must have its own retry budget.
	mutable int jacobian_calls = 0;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int /*variable*/) noexcept -> quokka::chemistry::VariableRole
	{
		return quokka::chemistry::VariableRole::species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int /*variable*/) noexcept -> bool { return true; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto temperature(quokka::chemistry::IntegratorState<variable_count> const & /*state*/) noexcept
	    -> amrex::Real
	{
		return 0.0;
	}

	AMREX_GPU_HOST_DEVICE void rhs(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real time,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		derivative[0] = time == 0.0 ? -200.0 : 0.0;
	}

	AMREX_GPU_HOST_DEVICE void jacobian(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real /*time*/,
					    quokka::chemistry::DenseMatrix<variable_count> &matrix) const noexcept
	{
		matrix.zero();
		++jacobian_calls;
		if (jacobian_calls == 5) {
			constexpr amrex::Real singular_step = 1.0 / 256.0;
			matrix(0, 0) = 1.0 / (singular_step * quokka::chemistry::rosenbrock::Ros2s::gamma);
		}
	}

	AMREX_GPU_HOST_DEVICE static void clean(quokka::chemistry::IntegratorState<variable_count> & /*state*/,
						quokka::chemistry::IntegratorOptions const & /*options*/) noexcept
	{
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(quokka::chemistry::IntegratorState<variable_count> const &state,
							      quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return std::isfinite(state.values[0]) && state.values[0] >= 0.0;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto valid_update(quokka::chemistry::IntegratorState<variable_count> const & /*old_state*/,
									       quokka::chemistry::IntegratorState<variable_count> const & /*new_state*/,
									       quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return true;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(quokka::chemistry::IntegratorState<variable_count> const &state,
								    quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return valid(state, options);
	}
};

struct TemperatureGateNetwork {
	static constexpr int variable_count = 1;
	mutable int rhs_calls = 0;
	mutable int jacobian_calls = 0;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int /*variable*/) noexcept -> quokka::chemistry::VariableRole
	{
		return quokka::chemistry::VariableRole::species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int /*variable*/) noexcept -> bool { return true; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto temperature(quokka::chemistry::IntegratorState<variable_count> const &state) noexcept
	    -> amrex::Real
	{
		return state.temperature;
	}

	AMREX_GPU_HOST_DEVICE void rhs(quokka::chemistry::IntegratorState<variable_count> const &state, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		++rhs_calls;
		derivative[0] = -state.values[0];
	}

	AMREX_GPU_HOST_DEVICE void jacobian(quokka::chemistry::IntegratorState<variable_count> const & /*state*/, amrex::Real /*time*/,
					    quokka::chemistry::DenseMatrix<variable_count> &matrix) const noexcept
	{
		++jacobian_calls;
		matrix.zero();
		matrix(0, 0) = -1.0;
	}

	AMREX_GPU_HOST_DEVICE static void clean(quokka::chemistry::IntegratorState<variable_count> & /*state*/,
						quokka::chemistry::IntegratorOptions const & /*options*/) noexcept
	{
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(quokka::chemistry::IntegratorState<variable_count> const &state,
							      quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return std::isfinite(state.values[0]);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto valid_update(quokka::chemistry::IntegratorState<variable_count> const & /*old_state*/,
									       quokka::chemistry::IntegratorState<variable_count> const & /*new_state*/,
									       quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return true;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(quokka::chemistry::IntegratorState<variable_count> const &state,
								    quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return valid(state, options);
	}
};

struct AcceptedStateCleaningNetwork {
	static constexpr int variable_count = 1;
	mutable bool evaluated_negative_state = false;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto variable_role(int /*variable*/) noexcept -> quokka::chemistry::VariableRole
	{
		return quokka::chemistry::VariableRole::species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto controls_error(int /*variable*/) noexcept -> bool { return true; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto temperature(quokka::chemistry::IntegratorState<variable_count> const & /*state*/) noexcept
	    -> amrex::Real
	{
		return 1.0;
	}

	AMREX_GPU_HOST_DEVICE void rhs(quokka::chemistry::IntegratorState<variable_count> const &state, amrex::Real /*time*/,
				       amrex::GpuArray<amrex::Real, variable_count> &derivative) const noexcept
	{
		evaluated_negative_state = evaluated_negative_state || state.values[0] < 0.0;
		derivative[0] = -0.51;
	}

	AMREX_GPU_HOST_DEVICE void jacobian(quokka::chemistry::IntegratorState<variable_count> const &state, amrex::Real /*time*/,
					    quokka::chemistry::DenseMatrix<variable_count> &matrix) const noexcept
	{
		evaluated_negative_state = evaluated_negative_state || state.values[0] < 0.0;
		matrix.zero();
	}

	AMREX_GPU_HOST_DEVICE static void clean(quokka::chemistry::IntegratorState<variable_count> &state,
						quokka::chemistry::IntegratorOptions const &options) noexcept
	{
		state.values[0] = amrex::max(state.values[0], options.small_state);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid(quokka::chemistry::IntegratorState<variable_count> const &state,
							      quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return std::isfinite(state.values[0]) && state.values[0] >= -options.atol_species;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static constexpr auto valid_update(quokka::chemistry::IntegratorState<variable_count> const & /*old_state*/,
									       quokka::chemistry::IntegratorState<variable_count> const & /*new_state*/,
									       quokka::chemistry::IntegratorOptions const & /*options*/) noexcept -> bool
	{
		return true;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE static auto valid_final(quokka::chemistry::IntegratorState<variable_count> const &state,
								    quokka::chemistry::IntegratorOptions const &options) noexcept -> bool
	{
		return valid(state, options);
	}
};

template <int N> [[nodiscard]] auto same_values(quokka::chemistry::IntegratorState<N> const &left, quokka::chemistry::IntegratorState<N> const &right) -> bool
{
	for (int variable = 0; variable < N; ++variable) {
		if (left.values[variable] != right.values[variable]) {
			return false;
		}
	}
	return true;
}

} // namespace

auto problem_main() -> int
{
	quokka::chemistry::ChemistryNetworkRegistry registry{};
	registry.add<quokka::chemistry::PhotoionizationNetwork>();
	registry.add<quokka::chemistry::PrimordialChemNetwork>();
	auto const *photoionization = registry.find("photoionization");
	if (photoionization == nullptr || photoionization->species_count != 3 || photoionization->variable_count != 6) {
		return 5;
	}
	if (quokka::chemistry::PhotoionizationNetwork::variables[quokka::chemistry::PhotoionizationNetwork::photon_flux_factor].controls_error) {
		return 6;
	}
	quokka::chemistry::PrimordialChemNetwork const primordial{};
	quokka::chemistry::IntegratorState<quokka::chemistry::PrimordialChemNetwork::variable_count> primordialState{};
	amrex::GpuArray<amrex::Real, quokka::chemistry::PrimordialChemNetwork::species_count> primordialNumberDensities{};
	primordialNumberDensities.fill(1.0);
	for (int species = 0; species < quokka::chemistry::PrimordialChemNetwork::species_count; ++species) {
		primordialState.values[species] = primordialNumberDensities[species];
	}
	primordialState.values[quokka::chemistry::PrimordialChemNetwork::energy] =
	    quokka::chemistry::PrimordialChemNetwork::specific_energy_from_temperature(primordialNumberDensities, 1000.0);
	const amrex::Real primordialSpeciesDensity = quokka::chemistry::PrimordialChemNetwork::mass_density(primordialState);
	primordialState.density = 2.0 * primordialSpeciesDensity;
	quokka::chemistry::PrimordialChemNetwork::update_thermodynamics(primordialState);
	if (primordialState.density != 2.0 * primordialSpeciesDensity || std::abs(primordialState.temperature - 1000.0) > 1.0e-12) {
		return 16;
	}
	primordialState.density = primordialSpeciesDensity;
	primordialState.values[2] *= 2.0;
	quokka::chemistry::PrimordialChemNetwork::set_specific_energy_from_temperature(primordialState, 750.0);
	if (primordialState.density != primordialSpeciesDensity || std::abs(primordialState.temperature - 750.0) > 1.0e-12) {
		return 17;
	}
	primordialState.values[2] = primordialNumberDensities[2];
	primordialState.values[quokka::chemistry::PrimordialChemNetwork::energy] =
	    quokka::chemistry::PrimordialChemNetwork::specific_energy_from_temperature(primordialNumberDensities, 1000.0);
	amrex::GpuArray<amrex::Real, quokka::chemistry::PrimordialChemNetwork::variable_count> primordialDerivative{};
	primordial.rhs(primordialState, 0.0, primordialDerivative);
	quokka::chemistry::DenseMatrix<quokka::chemistry::PrimordialChemNetwork::variable_count> primordialJacobian{};
	primordial.jacobian(primordialState, 0.0, primordialJacobian);
	if (!std::isfinite(primordialDerivative[0]) || !std::isfinite(primordialJacobian(0, 0))) {
		return 7;
	}
	quokka::chemistry::IntegratorOptions temperatureOptions{};
	temperatureOptions.maximum_temperature = 1000.0;
	temperatureOptions.rtol_species = 1.0e-6;
	temperatureOptions.rtol_energy = 1.0e-6;
	temperatureOptions.analytic_jacobian = false;
	primordialState.values[quokka::chemistry::PrimordialChemNetwork::energy] =
	    quokka::chemistry::PrimordialChemNetwork::specific_energy_from_temperature(primordialNumberDensities, 2000.0);
	auto const primordialHotInitial = primordialState;
	const auto primordialHotDiagnostics = quokka::chemistry::rosenbrock::integrate(primordial, primordialState, 1.0e6, temperatureOptions);
	if (!primordialHotDiagnostics.succeeded() || !same_values(primordialState, primordialHotInitial)) {
		return 10;
	}
	temperatureOptions.analytic_jacobian = true;
	quokka::chemistry::PhotoionizationNetwork const photoionizationNetwork{};
	quokka::chemistry::IntegratorState<quokka::chemistry::PhotoionizationNetwork::variable_count> photoionizationState{};
	amrex::GpuArray<amrex::Real, quokka::chemistry::PhotoionizationNetwork::species_count> photoionizationNumberDensities = {1.0, 100.0, 1.0};
	for (int species = 0; species < quokka::chemistry::PhotoionizationNetwork::species_count; ++species) {
		photoionizationState.values[species] = photoionizationNumberDensities[species];
		photoionizationState.density += photoionizationNumberDensities[species] * quokka::chemistry::PhotoionizationNetwork::species_masses[species];
	}
	photoionizationState.values[quokka::chemistry::PhotoionizationNetwork::energy] =
	    quokka::chemistry::PhotoionizationNetwork::specific_energy_from_temperature(photoionizationNumberDensities, 2000.0);
	photoionizationState.values[quokka::chemistry::PhotoionizationNetwork::photon_number] = 1.0e5;
	photoionizationState.values[quokka::chemistry::PhotoionizationNetwork::photon_flux_factor] = 1.0;
	photoionizationState.reduced_speed_of_light = 1.0e10;
	auto const photoionizationHotInitial = photoionizationState;
	const auto photoionizationHotDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(photoionizationNetwork, photoionizationState, 1.0, temperatureOptions);
	if (!photoionizationHotDiagnostics.succeeded() || !same_values(photoionizationState, photoionizationHotInitial)) {
		return 11;
	}
	StageValidationNetwork const stageValidationNetwork{};
	quokka::chemistry::IntegratorState<StageValidationNetwork::variable_count> stageValidationState{};
	stageValidationState.values[0] = 1.0;
	auto stageValidationOptions = temperatureOptions;
	stageValidationOptions.maximum_steps = 1;
	const auto stageValidationDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(stageValidationNetwork, stageValidationState, 1.0, stageValidationOptions);
	if (stageValidationDiagnostics.rejected_steps == 0 || stageValidationNetwork.cleaned_invalid_stage) {
		return 12;
	}
	RetryAccountingNetwork const retryAccountingNetwork{};
	quokka::chemistry::IntegratorState<RetryAccountingNetwork::variable_count> retryAccountingState{};
	retryAccountingState.values[0] = 1.0;
	auto retryAccountingOptions = temperatureOptions;
	retryAccountingOptions.atol_species = 1.0;
	retryAccountingOptions.rtol_species = 1.0;
	retryAccountingOptions.tableau = 3;
	const auto retryAccountingDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(retryAccountingNetwork, retryAccountingState, 1.0, retryAccountingOptions);
	if (!retryAccountingDiagnostics.succeeded() || retryAccountingNetwork.jacobian_calls <= 5) {
		return 13;
	}
	quokka::chemistry::IntegratorOptions lowerTemperatureOptions{};
	lowerTemperatureOptions.minimum_temperature = 10.0;
	lowerTemperatureOptions.maximum_temperature = 100.0;
	lowerTemperatureOptions.maximum_timestep = 0.5;
	lowerTemperatureOptions.rtol_species = 1.0e-6;
	lowerTemperatureOptions.atol_species = 1.0e-8;
	TemperatureGateNetwork const analyticTemperatureGate{};
	quokka::chemistry::IntegratorState<TemperatureGateNetwork::variable_count> analyticTemperatureState{};
	analyticTemperatureState.values[0] = 1.0;
	analyticTemperatureState.temperature = 1.0;
	const auto analyticTemperatureDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(analyticTemperatureGate, analyticTemperatureState, 1.0, lowerTemperatureOptions);
	lowerTemperatureOptions.analytic_jacobian = false;
	TemperatureGateNetwork const numericalTemperatureGate{};
	quokka::chemistry::IntegratorState<TemperatureGateNetwork::variable_count> numericalTemperatureState{};
	numericalTemperatureState.values[0] = 1.0;
	numericalTemperatureState.temperature = 1.0;
	const auto numericalTemperatureDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(numericalTemperatureGate, numericalTemperatureState, 1.0, lowerTemperatureOptions);
	if (!analyticTemperatureDiagnostics.succeeded() || !numericalTemperatureDiagnostics.succeeded() || analyticTemperatureState.values[0] != 1.0 ||
	    numericalTemperatureState.values[0] != 1.0 || analyticTemperatureGate.rhs_calls != 0 || analyticTemperatureGate.jacobian_calls != 0 ||
	    numericalTemperatureGate.rhs_calls != 0 || numericalTemperatureGate.jacobian_calls != 0) {
		return 14;
	}
	AcceptedStateCleaningNetwork const acceptedStateCleaningNetwork{};
	quokka::chemistry::IntegratorState<AcceptedStateCleaningNetwork::variable_count> acceptedStateCleaningState{};
	acceptedStateCleaningState.values[0] = 0.05;
	auto acceptedStateCleaningOptions = lowerTemperatureOptions;
	acceptedStateCleaningOptions.minimum_temperature = 0.0;
	acceptedStateCleaningOptions.maximum_timestep = 0.1;
	acceptedStateCleaningOptions.atol_species = 1.0;
	acceptedStateCleaningOptions.rtol_species = 1.0;
	acceptedStateCleaningOptions.small_state = 0.0;
	acceptedStateCleaningOptions.tableau = 3;
	acceptedStateCleaningOptions.analytic_jacobian = true;
	const auto acceptedStateCleaningDiagnostics =
	    quokka::chemistry::rosenbrock::integrate(acceptedStateCleaningNetwork, acceptedStateCleaningState, 0.2, acceptedStateCleaningOptions);
	if (!acceptedStateCleaningDiagnostics.succeeded() || acceptedStateCleaningNetwork.evaluated_negative_state) {
		return 15;
	}
	DecayNetwork const network{};
	quokka::chemistry::IntegratorState<DecayNetwork::variable_count> state{};
	state.values = {1.0, 1.0};
	quokka::chemistry::IntegratorOptions options{};
	options.rtol_species = 1.0e-10;
	options.atol_species = 1.0e-12;
	options.tableau = 0;
	const auto rodas5pDiagnostics = quokka::chemistry::rosenbrock::integrate(network, state, 1.0, options);
	if (!rodas5pDiagnostics.succeeded() || std::abs(state.values[0] - std::exp(-1.0)) > 1.0e-8) {
		return 8;
	}
	state.values = {1.0, 1.0};
	options.tableau = 3;
	const auto diagnostics = quokka::chemistry::rosenbrock::integrate(network, state, 1.0, options);
	if (!diagnostics.succeeded()) {
		return 1;
	}
	if (std::abs(state.values[0] - std::exp(-1.0)) > 1.0e-8) {
		return 2;
	}
	// The second component is integrated but deliberately excluded from the
	// error norm, exercising the contract needed by photon flux attenuation.
	if (!(state.values[1] >= 0.0 && state.values[1] < 1.0e-6)) {
		return 3;
	}
	options.tableau = 1;
	const auto unsupported = quokka::chemistry::rosenbrock::integrate(network, state, 1.0, options);
	if (unsupported.status != quokka::chemistry::IntegratorStatus::bad_inputs) {
		return 4;
	}
	options.tableau = 0;
	options.rtol_species = std::numeric_limits<amrex::Real>::epsilon();
	const auto excessiveAccuracy = quokka::chemistry::rosenbrock::integrate(network, state, 1.0, options);
	return excessiveAccuracy.status == quokka::chemistry::IntegratorStatus::accuracy_unattainable ? 0 : 9;
}
