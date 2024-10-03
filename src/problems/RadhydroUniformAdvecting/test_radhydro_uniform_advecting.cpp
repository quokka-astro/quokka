/// \file test_radhydro_uniform_advecting.cpp
/// \brief Defines a test problem for radiation advection in a uniform medium with grey radiation.
///

#include "test_radhydro_uniform_advecting.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double T_exact = 0.768032502191; // equilibrium temperature
constexpr int beta_order_ = 1; // order of beta in the radiation four-force
constexpr double v0 = 0.0;
constexpr double kappa0 = 1.0;
constexpr double c = 1.0;
constexpr double chat = 1.0;
constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

constexpr double max_time = 10.0;

constexpr double erad_floor = 1.0e-20;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = beta_order_;
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};






// template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void rhs_specie(const burn_t& state, Array1D<Real, 1, neqs>& ydot, const Array1D<Real, 0, NumSpec-1>& X) {
	Real const Tdust = state.T;
	Real const rho = state.rho;

	// // Radiation
	// const auto fourPiBoverc = RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(Tdust);
	// const auto kappa_B = RadSystem<problem_t>::ComputePlanckOpacity(rho, Tdust);
	// const auto kappa_E = RadSystem<problem_t>::ComputeEnergyMeanOpacity(rho, Tdust);

	// // <ydot>
	// for (int g = 0; g < n_group_in_rhs; ++g) {
	// 	ydot(g + 1) = RadSystem<problem_t>::c_hat_ * rho * (kappa_B[g] * fourPiBoverc[g] - kappa_E[g] * X(g)); // X = Erad
	// }

	const Real fourPiBoverc = a_rad * Tdust * Tdust * Tdust * Tdust;

	ydot(1) = chat * rho * kappa0 * (fourPiBoverc - X(0));
}

// template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real rhs_eint(const burn_t& state, const Array1D<Real, 0, NumSpec-1>& X) {
	Real Tdust = state.T;
	Real rho = state.rho;

	// // Assuming NumSpec - 1 = neqs
	// const amrex::GpuArray<Real, n_group_in_rhs> fourPiBoverc = RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(Tdust);
	// const amrex::GpuArray<Real, n_group_in_rhs> kappa_B = RadSystem<problem_t>::ComputePlanckOpacity(rho, Tdust);
	// const amrex::GpuArray<Real, n_group_in_rhs> kappa_E = RadSystem<problem_t>::ComputeEnergyMeanOpacity(rho, Tdust);

	// Real edot = 0.0;
	// for (int g = 0; g < n_group_in_rhs; ++g) {
	// 	edot += - RadSystem<problem_t>::c_hat_ * rho * (kappa_B[g] * fourPiBoverc[g] - kappa_E[g] * X(g)); // X = Erad
	// }

	const Real fourPiBoverc = a_rad * Tdust * Tdust * Tdust * Tdust;

	const Real edot = - c * rho * kappa0 * (fourPiBoverc - X(0));

  return edot;
}

// template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_rhs(burn_t& state, Array1D<Real, 1, neqs>& ydot)
{
	Array1D<Real, 0, NumSpec-1> X;
	for (int i = 0; i < NumSpec; ++i) {
		X(i) = state.xn[i];
	}

	// YDOTS and Edot

	rhs_specie(state, ydot, X);
	Real edot = rhs_eint(state, X);

	// Append the energy equation (this is erg/g/s)
	ydot(net_ienuc) = edot;
}












template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	double erad = NAN;
	double frad = NAN;
	erad = erad_floor;
	frad = 0.0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = frad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 8.0;

	const double max_dt = 1.0;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> Vgas_exact(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> Trad_exact{};
	std::vector<double> Tgas_exact{};
	std::vector<double> Erad_exact{};

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		rhogas.at(i) = rho_t;
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas.at(i));
		Tgas_exact.push_back(T_exact);
		Vgas.at(i) = v_t;
		Vgas_exact.at(i) = 0.0;
	}

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_exact[i]);
		sol_norm += std::abs(Tgas_exact[i]);
	}
	const double error_tol = 1.0e-4; // This is a very very stringent test (to machine accuracy!)
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Texact_args;
	std::map<std::string, std::string> Tradexact_args;
	Trad_args["label"] = "radiation (numerical)";
	Trad_args["linestyle"] = "--";
	Tgas_args["label"] = "gas (numerical)";
	Tgas_args["linestyle"] = "--";
	Texact_args["label"] = "radiation/gas (exact)";
	Texact_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Tgas_exact, Texact_args);
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::ylim(-0.1, 1.1);
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./gas-cooling-down-T.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	vgas_args["label"] = "gas velocity (exact)";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas_exact, vgas_args);
	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("v (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./gas-cooling-down-velocity.pdf");
#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
