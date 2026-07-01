/// \file testRadhydroEquilibrationAnomaly.cpp
/// \brief Demonstrates anomalous acceleration during moving radiation-matter equilibration.

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#include <cmath>
#include <fstream>
#include <vector>

struct RadhydroEquilibrationAnomaly {
};

namespace
{
constexpr double c_light_value = 100.0;
constexpr double beta0 = 1.0e-2;
constexpr double v0 = beta0 * c_light_value;
constexpr double rho0 = 1.0;
constexpr double Tgas0 = 1.0;
constexpr double radiation_constant_value = 1.0;
constexpr double Erad0 = 4.0;
constexpr double kappa0 = 1.0;
constexpr double dt = 1.0e-4;
double initialErad = Erad0;			       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace

template <> struct SimulationData<RadhydroEquilibrationAnomaly> {
	std::vector<double> time;
	std::vector<double> velocity;
};

template <> struct quokka::EOS_Traits<RadhydroEquilibrationAnomaly> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<RadhydroEquilibrationAnomaly> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<RadhydroEquilibrationAnomaly> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double c_light = c_light_value;
	static constexpr double radiation_constant = radiation_constant_value;
	static constexpr double gravitational_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<RadhydroEquilibrationAnomaly>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
							       quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return std::pow(Egas, 1. / 4.);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<RadhydroEquilibrationAnomaly>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
							       quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return std::pow(Tgas, 4);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<RadhydroEquilibrationAnomaly>::ComputeEintTempDerivative(
    const double /*rho*/, const double Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
{
	return 4.0 * std::pow(Tgas, 3);
}

namespace
{
[[nodiscard]] auto initialGasInternalEnergy() -> double
{
	return quokka::EOS<RadhydroEquilibrationAnomaly>::ComputeEintFromTgas(rho0, Tgas0);
}
} // namespace

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<RadhydroEquilibrationAnomaly>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<RadhydroEquilibrationAnomaly>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<RadhydroEquilibrationAnomaly>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double Egas0 = initialGasInternalEnergy();
	const double x1GasMomentum0 = rho0 * v0;
	const double gasEnergy0 = quokka::EOS<RadhydroEquilibrationAnomaly>::ComputeEgasFromEint(rho0, x1GasMomentum0, 0.0, 0.0, Egas0, 0.0);
	const double localErad0 = initialErad;
	const double initialRadFlux = (4.0 / 3.0) * v0 * localErad0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x1GasMomentum_index) = x1GasMomentum0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x2GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x3GasMomentum_index) = 0.0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::gasEnergy_index) = gasEnergy0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::gasInternalEnergy_index) = Egas0;

		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::radEnergy_index) = localErad0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x1RadFlux_index) = initialRadFlux;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x2RadFlux_index) = 0.0;
		state_cc(i, j, k, RadSystem<RadhydroEquilibrationAnomaly>::x3RadFlux_index) = 0.0;
	});
}

template <> void QuokkaSimulation<RadhydroEquilibrationAnomaly>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.0);
	static_cast<void>(position);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const double rho = values.at(RadSystem<RadhydroEquilibrationAnomaly>::gasDensity_index)[0];
		const double x1GasMomentum = values.at(RadSystem<RadhydroEquilibrationAnomaly>::x1GasMomentum_index)[0];
		userData_.time.push_back(tNew_[0]);
		userData_.velocity.push_back(x1GasMomentum / rho);
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<RadhydroEquilibrationAnomaly> sim;
	double stop_time = dt;
	double constant_dt = dt;
	int write_history = 0;
	int check_short_step = 1;
	amrex::ParmParse pp("anomaly");
	pp.query("initial_erad", initialErad);
	pp.query("stop_time", stop_time);
	pp.query("constant_dt", constant_dt);
	pp.query("write_history", write_history);
	pp.query("check_short_step", check_short_step);

	sim.cflNumber_ = 0.8;
	sim.radiationCflNumber_ = 0.8;
	sim.constantDt_ = constant_dt;
	sim.stopTime_ = stop_time;
	sim.maxTimesteps_ = static_cast<int>(std::ceil(stop_time / constant_dt)) + 1;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.time.push_back(0.0);
		sim.userData_.velocity.push_back(v0);
	}
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	static_cast<void>(position);
	const double rho = values.at(RadSystem<RadhydroEquilibrationAnomaly>::gasDensity_index)[0];
	const double x1GasMomentum = values.at(RadSystem<RadhydroEquilibrationAnomaly>::x1GasMomentum_index)[0];
	const double v = x1GasMomentum / rho;
	const double delta_v = v - v0;
	const double equilibriumErad = radiation_constant_value * std::pow(Tgas0, 4);
	const double expected_delta_v = constant_dt * kappa0 * v0 * (initialErad - equilibriumErad) / c_light_value;
	const double rel_error = std::abs(delta_v - expected_delta_v) / std::abs(expected_delta_v);

	amrex::Print() << "initial velocity = " << v0 << '\n';
	amrex::Print() << "final velocity = " << v << '\n';
	amrex::Print() << "measured anomalous delta_v = " << delta_v << '\n';
	amrex::Print() << "small-step Newtonian prediction = " << expected_delta_v << '\n';
	amrex::Print() << "relative error = " << rel_error << '\n';

	if ((write_history != 0) && amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream history("radhydro_equilibration_anomaly_velocity.csv");
		history << "time,velocity\n";
		for (std::size_t i = 0; i < sim.userData_.time.size(); ++i) {
			history << sim.userData_.time[i] << ',' << sim.userData_.velocity[i] << '\n';
		}
	}

	int status = 0;
	if ((check_short_step != 0) && (!(delta_v > 0.0) || rel_error > 5.0e-2 || std::isnan(rel_error))) {
		status = 1;
	}
	return status;
}
