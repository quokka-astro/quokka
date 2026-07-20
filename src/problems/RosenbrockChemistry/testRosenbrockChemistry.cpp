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
	amrex::GpuArray<amrex::Real, quokka::chemistry::PrimordialChemNetwork::variable_count> primordialDerivative{};
	primordial.rhs(primordialState, 0.0, primordialDerivative);
	quokka::chemistry::DenseMatrix<quokka::chemistry::PrimordialChemNetwork::variable_count> primordialJacobian{};
	primordial.jacobian(primordialState, 0.0, primordialJacobian);
	if (!std::isfinite(primordialDerivative[0]) || !std::isfinite(primordialJacobian(0, 0))) {
		return 7;
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
