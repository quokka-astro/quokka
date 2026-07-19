/// \file testDustHallPedersenDrift.cpp
/// \brief Stiff-limit Hall/Pedersen drift test for coupled gas-dust dynamics with Lorentz force.
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct DustHallPedersenDrift {
};

namespace
{
constexpr double rho_gas = 1.0;
constexpr double epsilon = 1.0;
constexpr double rho_dust = epsilon * rho_gas;
constexpr double sound_speed = 1.0;
constexpr double alpha_d = 1.0;
constexpr double charge_to_mass_ratio = 1.0;
constexpr double magnetic_field_z = 1.0;
constexpr double external_force = 1.0;

constexpr double constant_dt = 5.0;
constexpr double stop_time = 20.0;

constexpr double omega_L = charge_to_mass_ratio * magnetic_field_z;
constexpr double alpha_rel = (1.0 + epsilon) * alpha_d;
constexpr double omega_rel = (1.0 + epsilon) * omega_L;
constexpr double g_rel_x = -external_force / rho_gas;

struct DriftState {
	double ux;
	double uy;
};
} // namespace

template <> struct SimulationData<DustHallPedersenDrift> {
	std::vector<double> t_vec_;
	std::vector<double> ux_vec_;
	std::vector<double> uy_vec_;
	std::vector<double> uz_vec_;
	std::vector<double> center_momentum_x_vec_;
	std::vector<double> center_momentum_y_vec_;
	std::vector<double> center_momentum_z_vec_;
};

template <> struct quokka::EOS_Traits<DustHallPedersenDrift> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = sound_speed;
};

template <> struct Physics_Traits<DustHallPedersenDrift> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustHallPedersenDrift>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
											     amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
											     amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/,
											     double /*cs*/) -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> alpha{};
	alpha[0] = alpha_d;
	return alpha;
}

template <> AMREX_GPU_HOST_DEVICE auto DustSources<DustHallPedersenDrift>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> q_over_m{};
	q_over_m[0] = charge_to_mass_ratio;
	return q_over_m;
}

template <> void QuokkaSimulation<DustHallPedersenDrift>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<DustHallPedersenDrift>::nvarTotal_cc;
	const double magnetic_energy = 0.5 * magnetic_field_z * magnetic_field_z;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::energy_index) = magnetic_energy;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x3Momentum_index) = 0.0;

		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::dustDensity_index) = rho_dust;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x1DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x2DustMomentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustHallPedersenDrift>::x3DustMomentum_index) = 0.0;
	});
}

template <> void QuokkaSimulation<DustHallPedersenDrift>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_fc = Physics_Indices<DustHallPedersenDrift>::nvarPerDim_fc;
	const double bfield = (grid_elem.dir_ == quokka::direction::z) ? magnetic_field_z : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0;
		}
		state_fc(i, j, k, Physics_Indices<DustHallPedersenDrift>::mhdFirstIndex) = bfield;
	});
}

void recordHistory(QuokkaSimulation<DustHallPedersenDrift> &sim)
{
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5);

	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}

	auto &data = sim.userData_;
	data.t_vec_.push_back(sim.tNew_[0]);

	const double gas_density = values.at(HydroSystem<DustHallPedersenDrift>::density_index)[0];
	const double gas_momentum_x = values.at(HydroSystem<DustHallPedersenDrift>::x1Momentum_index)[0];
	const double gas_momentum_y = values.at(HydroSystem<DustHallPedersenDrift>::x2Momentum_index)[0];
	const double gas_momentum_z = values.at(HydroSystem<DustHallPedersenDrift>::x3Momentum_index)[0];
	const double gas_vx = gas_momentum_x / gas_density;
	const double gas_vy = gas_momentum_y / gas_density;
	const double gas_vz = gas_momentum_z / gas_density;

	const double dust_density = values.at(HydroSystem<DustHallPedersenDrift>::dustDensity_index)[0];
	const double dust_momentum_x = values.at(HydroSystem<DustHallPedersenDrift>::x1DustMomentum_index)[0];
	const double dust_momentum_y = values.at(HydroSystem<DustHallPedersenDrift>::x2DustMomentum_index)[0];
	const double dust_momentum_z = values.at(HydroSystem<DustHallPedersenDrift>::x3DustMomentum_index)[0];
	const double dust_vx = dust_momentum_x / dust_density;
	const double dust_vy = dust_momentum_y / dust_density;
	const double dust_vz = dust_momentum_z / dust_density;

	data.ux_vec_.push_back(dust_vx - gas_vx);
	data.uy_vec_.push_back(dust_vy - gas_vy);
	data.uz_vec_.push_back(dust_vz - gas_vz);
	data.center_momentum_x_vec_.push_back(gas_momentum_x + dust_momentum_x);
	data.center_momentum_y_vec_.push_back(gas_momentum_y + dust_momentum_y);
	data.center_momentum_z_vec_.push_back(gas_momentum_z + dust_momentum_z);
}

template <> void QuokkaSimulation<DustHallPedersenDrift>::computeAfterTimestep() { recordHistory(*this); }

template <> void QuokkaSimulation<DustHallPedersenDrift>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	amrex::ignore_unused(lev);
	amrex::ignore_unused(time);

	const double magnetic_energy = 0.5 * magnetic_field_z * magnetic_field_z;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<DustHallPedersenDrift>::density_index);
			const amrex::Real x1mom = state(i, j, k, HydroSystem<DustHallPedersenDrift>::x1Momentum_index);
			const amrex::Real x2mom = state(i, j, k, HydroSystem<DustHallPedersenDrift>::x2Momentum_index);
			const amrex::Real x3mom = state(i, j, k, HydroSystem<DustHallPedersenDrift>::x3Momentum_index);

			const amrex::Real x1mom_new = x1mom + dt_lev * external_force;
			const amrex::Real gas_kinetic_energy_new = (x1mom_new * x1mom_new + x2mom * x2mom + x3mom * x3mom) / (2.0 * rho);
			const amrex::Real Egas_new = magnetic_energy + gas_kinetic_energy_new;

			AMREX_ASSERT(!std::isnan(x1mom_new));
			AMREX_ASSERT(!std::isnan(Egas_new));

			state(i, j, k, HydroSystem<DustHallPedersenDrift>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<DustHallPedersenDrift>::energy_index) = Egas_new;
			state(i, j, k, HydroSystem<DustHallPedersenDrift>::internalEnergy_index) = 0.0;
		});
	}
}

auto runHallPedersenSimulation() -> SimulationData<DustHallPedersenDrift>
{
	using quokka::BCType::mathematicalBndryTypes;
	auto BCs_cc = quokka::BC<DustHallPedersenDrift>(quokka::BCType::int_dir);
	auto BCs_fc =
	    quokka::BC_fc<DustHallPedersenDrift>(mathematicalBndryTypes::periodic, mathematicalBndryTypes::periodic, mathematicalBndryTypes::periodic);
	QuokkaSimulation<DustHallPedersenDrift> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = 1000000.0; // large CFL number to avoid CFL violation
	sim.constantDt_ = constant_dt;
	sim.stopTime_ = stop_time;

	sim.setInitialConditions();
	recordHistory(sim);
	sim.evolve();

	return sim.userData_;
}

auto steadyRelativeDrift() -> DriftState
{
	const double denom = alpha_rel * alpha_rel + omega_rel * omega_rel;
	return {.ux = alpha_rel * g_rel_x / denom, .uy = -omega_rel * g_rel_x / denom};
}

auto analyticRelativeDrift(double t) -> DriftState
{
	const DriftState steady = steadyRelativeDrift();
	const double delta_ux0 = -steady.ux;
	const double delta_uy0 = -steady.uy;
	const double decay = std::exp(-alpha_rel * t);
	const double cos_term = std::cos(omega_rel * t);
	const double sin_term = std::sin(omega_rel * t);

	return {.ux = steady.ux + decay * (delta_ux0 * cos_term + delta_uy0 * sin_term),
		.uy = steady.uy + decay * (-delta_ux0 * sin_term + delta_uy0 * cos_term)};
}

auto relativeDriftL2Error(SimulationData<DustHallPedersenDrift> const &data) -> double
{
	double err_sq = 0.0;
	double ref_sq = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		DriftState const exact = analyticRelativeDrift(data.t_vec_[i]);
		const double dux = data.ux_vec_[i] - exact.ux;
		const double duy = data.uy_vec_[i] - exact.uy;
		err_sq += dux * dux + duy * duy;
		ref_sq += exact.ux * exact.ux + exact.uy * exact.uy;
	}
	return (ref_sq > 0.0) ? std::sqrt(err_sq / ref_sq) : 1.0;
}

auto finalRelativeDriftError(SimulationData<DustHallPedersenDrift> const &data) -> double
{
	if (data.t_vec_.empty()) {
		return 1.0;
	}

	const size_t i = data.t_vec_.size() - 1;
	DriftState const exact = analyticRelativeDrift(data.t_vec_[i]);
	const double dux = data.ux_vec_[i] - exact.ux;
	const double duy = data.uy_vec_[i] - exact.uy;
	return std::sqrt(dux * dux + duy * duy);
}

auto maxMomentumResidual(SimulationData<DustHallPedersenDrift> const &data) -> double
{
	double max_residual = 0.0;
	for (size_t i = 0; i < data.t_vec_.size(); ++i) {
		const double px_exact = external_force * data.t_vec_[i];
		max_residual = std::max(max_residual, std::abs(data.center_momentum_x_vec_[i] - px_exact));
		max_residual = std::max(max_residual, std::abs(data.center_momentum_y_vec_[i]));
		max_residual = std::max(max_residual, std::abs(data.center_momentum_z_vec_[i]));
		max_residual = std::max(max_residual, std::abs(data.uz_vec_[i]));
	}
	return max_residual;
}

#ifdef HAVE_PYTHON
void plotRelativeDrift(SimulationData<DustHallPedersenDrift> const &data)
{
	const size_t n_dense = 1000;
	std::vector<double> t_dense;
	std::vector<double> ux_dense;
	std::vector<double> uy_dense;
	const double t_max = data.t_vec_.empty() ? 0.0 : data.t_vec_.back();
	t_dense.resize(n_dense);
	ux_dense.resize(n_dense);
	uy_dense.resize(n_dense);

	for (size_t i = 0; i < n_dense; ++i) {
		const double t = t_max * static_cast<double>(i) / static_cast<double>(n_dense - 1);
		DriftState const exact = analyticRelativeDrift(t);
		t_dense[i] = t;
		ux_dense[i] = exact.ux;
		uy_dense[i] = exact.uy;
	}

	matplotlibcpp::clf();
	matplotlibcpp::plot(t_dense, ux_dense, {{"label", "u_x analytic"}, {"color", "C0"}, {"linestyle", "--"}, {"linewidth", "1.0"}});
	matplotlibcpp::plot(data.t_vec_, data.ux_vec_,
			    {{"label", "u_x numerical"}, {"color", "C0"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "4"}});
	matplotlibcpp::plot(t_dense, uy_dense, {{"label", "u_y analytic"}, {"color", "C1"}, {"linestyle", "--"}, {"linewidth", "1.0"}});
	matplotlibcpp::plot(data.t_vec_, data.uy_vec_,
			    {{"label", "u_y numerical"}, {"color", "C1"}, {"linestyle", "None"}, {"marker", "s"}, {"markersize", "4"}});
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("t");
	matplotlibcpp::ylabel(R"($u = v_d - v_g$)");
	matplotlibcpp::title("Stiff Hall/Pedersen Drift");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./dust_hall_pedersen_stiff.pdf");
}
#endif

auto problem_main() -> int
{
	auto data = runHallPedersenSimulation();

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const DriftState steady = steadyRelativeDrift();
		const double rel_error = relativeDriftL2Error(data);
		const double final_error = finalRelativeDriftError(data);
		const double momentum_residual = maxMomentumResidual(data);

		amrex::Print() << "\nHall/Pedersen drift analytic steady state:\n";
		amrex::Print() << "  u_x* = " << steady.ux << "\n";
		amrex::Print() << "  u_y* = " << steady.uy << "\n";
		amrex::Print() << "\nStiff-case diagnostics:\n";
		amrex::Print() << "  relative drift L2 error = " << rel_error << "\n";
		amrex::Print() << "  final drift error       = " << final_error << "\n";
		amrex::Print() << "  max momentum residual   = " << momentum_residual << "\n";

		const double rel_error_tol = 0.006;
		const double final_error_tol = 0.002;
		const double momentum_tol = 1.0e-14;
		const bool passed =
		    (rel_error <= rel_error_tol) && (final_error <= final_error_tol) && (momentum_residual <= momentum_tol) && std::isfinite(rel_error);

		if (!passed) {
			status = 1;
			amrex::Print() << "\nTest FAILED: stiff Hall/Pedersen drift errors exceeded tolerance.\n";
		} else {
			amrex::Print() << "\nTest PASSED: stiff Hall/Pedersen drift matches the analytic solution.\n";
		}

#ifdef HAVE_PYTHON
		plotRelativeDrift(data);
#endif
	}

	return status;
}
