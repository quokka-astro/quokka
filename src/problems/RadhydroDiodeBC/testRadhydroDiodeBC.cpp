/// \file testRadhydroDiodeBC.cpp
/// \brief Tests the diode boundary condition (setDiodeBCLo/Hi) with multigroup radiation enabled.
///
/// The gas moves uniformly in +x, so the lower x boundary takes the inflow (reflecting) branch of the diode BC
/// and the upper x boundary takes the outflow (copy) branch. Radiation is decoupled from the gas (tiny opacity,
/// beta_order = 0). Group 0 is a uniform field with zero flux: a boundary that neither injects nor drains radiation
/// must leave it unchanged. Group 1 is a pulse that streams out of both boundaries: its total energy must not grow.

#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#include <cmath>

struct DiodeProblem {}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 2;
constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1.0e-3, 1.0, 1.0e3};

constexpr double c = 10.0;	   // speed of light (dimensionless)
constexpr double kappa0 = 1.0e-10; // opacity, small enough that radiation is decoupled from the gas
constexpr double rho0 = 1.0;
constexpr double T0 = 1.0;
constexpr double v0 = 0.1; // gas velocity in +x
constexpr double a_rad = 1.0;
constexpr double k_B = 1.0;
constexpr double mu = 1.0;

constexpr double Erad0 = 1.0;	    // uniform radiation energy density of group 0
constexpr double Erad1_bg = 1.0e-6; // background radiation energy density of group 1
constexpr double Erad1_pk = 1.0;    // peak of the group 1 pulse
constexpr double pulse_width = 0.05;
constexpr double max_time = 0.5; // five light-crossing times

template <> struct quokka::EOS_Traits<DiodeProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DiodeProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<DiodeProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DiodeProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<DiodeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<DiodeProblem>::ComputeEintFromTgas(rho0, T0);
	constexpr int nv = Physics_NumVars::numRadVarsPerGroup;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Erad1 = Erad1_bg + Erad1_pk * std::exp(-(x - 0.5) * (x - 0.5) / (2.0 * pulse_width * pulse_width));
		state_cc(i, j, k, RadSystem<DiodeProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<DiodeProblem>::radEnergy_index + nv) = Erad1;
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<DiodeProblem>::x1RadFlux_index + nv * g) = 0.;
			state_cc(i, j, k, RadSystem<DiodeProblem>::x2RadFlux_index + nv * g) = 0.;
			state_cc(i, j, k, RadSystem<DiodeProblem>::x3RadFlux_index + nv * g) = 0.;
		}
		state_cc(i, j, k, RadSystem<DiodeProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<DiodeProblem>::x1GasMomentum_index) = rho0 * v0;
		state_cc(i, j, k, RadSystem<DiodeProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DiodeProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DiodeProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<DiodeProblem>::gasInternalEnergy_index) = Egas;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DiodeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							 amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							 int /*bcomp*/, int /*orig_comp*/)
{
	setDiodeBCLo<0>(iv, consVar, geom);
	setDiodeBCHi<0>(iv, consVar, geom);
}

auto problem_main() -> int
{
	constexpr int nvars = RadSystem<DiodeProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // diode
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // diode
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<DiodeProblem> sim(BCs_cc);
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.cflNumber_ = 0.3;
	sim.radiationCflNumber_ = 0.3;
	sim.maxTimesteps_ = 10000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	sim.evolve();
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	constexpr int nv = Physics_NumVars::numRadVarsPerGroup;
	const int E0 = RadSystem<DiodeProblem>::radEnergy_index;
	const int F0 = RadSystem<DiodeProblem>::x1RadFlux_index;
	const int E1 = E0 + nv;

	bool all_finite = true;
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < nx; ++i) {
			all_finite = all_finite && std::isfinite(values.at(n)[i]);
		}
	}

	// group 0: uniform, zero-flux field must stay uniform with zero flux
	double max_dE0 = 0.;
	double max_F0 = 0.;
	// group 1: the pulse must leave the domain, and total energy must not grow
	double sum_E1_init = 0.;
	double sum_E1 = 0.;
	for (int i = 0; i < nx; ++i) {
		max_dE0 = std::max(max_dE0, std::abs(values.at(E0)[i] / Erad0 - 1.0));
		max_F0 = std::max(max_F0, std::abs(values.at(F0)[i]) / (c * Erad0));
		sum_E1_init += values0.at(E1)[i];
		sum_E1 += values.at(E1)[i];
	}
	const double sum_E1_bg = Erad1_bg * nx;
	const double excess_ratio = (sum_E1 - sum_E1_bg) / (sum_E1_init - sum_E1_bg);

	amrex::Print() << "all values finite = " << all_finite << '\n';
	amrex::Print() << "group 0: max |E/E0 - 1| = " << max_dE0 << ", max |F|/(cE0) = " << max_F0 << '\n';
	amrex::Print() << "group 1: total E final / initial = " << sum_E1 / sum_E1_init << ", remaining pulse fraction = " << excess_ratio << '\n';

	const double tol = 1.0e-8;
	int status = 0;
	if (!all_finite || !(max_dE0 < tol) || !(max_F0 < tol) || !(sum_E1 <= sum_E1_init * (1.0 + tol)) || !(excess_ratio < 0.05)) {
		status = 1;
	}
	return status;
}
