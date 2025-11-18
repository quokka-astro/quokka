#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"
#include <fmt/format.h>

struct ParticleRadiationProblem {
};

constexpr int ngroups_ = 4;
constexpr amrex::GpuArray<double, ngroups_ + 1> radBoundaries_{1.e-04, 1.00778140e-01, 1.00778140e+00, 5.53817071e+00, 1.e+2};

constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 0.2;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;
constexpr double chat_over_c = 1.0;

template <> struct quokka::EOS_Traits<ParticleRadiationProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<ParticleRadiationProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = ngroups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<ParticleRadiationProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = 1.;
	static constexpr amrex::GpuArray<double, Physics_Traits<ParticleRadiationProblem>::nGroups + 1> radBoundaries = radBoundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

// template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
// {
// 	return kappa0;
// }

// template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleRadiationProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
// {
// 	return kappa0;
// }

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleRadiationProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, ngroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
									  const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	constexpr double gas_to_dust_ratio = 1.0e-3;
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0; // power-law slopes
	}
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}


template <> void QuokkaSimulation<ParticleRadiationProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<ParticleRadiationProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<ParticleRadiationProblem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) =
			    Erad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleRadiationProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const double tmax = 1.0;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<ParticleRadiationProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<ParticleRadiationProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// Total radiation energy in the field
	amrex::Real total_Erad_init = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad_init +=
		    sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;


	// evolve
	sim.evolve();


	// Total radiation energy in the field
	amrex::Real total_Erad = 0.0;
	for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
		total_Erad += sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
	}

	// total gas energy
	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(RadSystem<ParticleRadiationProblem>::gasEnergy_index) * vol;


	if (amrex::ParallelDescriptor::IOProcessor()) {

		// print total gas energy
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		amrex::Print() << "Total radiation energy (initial): " << total_Erad_init / chat_over_c << "\n";
		amrex::Print() << "Total radiation energy (final): " << total_Erad / chat_over_c << "\n";

		const double total_energy_init = total_Erad_init / chat_over_c + total_gas_energy_init;
		const double total_energy = total_Erad / chat_over_c + total_gas_energy;
		const double change_of_total_energy = total_energy - total_energy_init;
		amrex::Print() << "Change of total energy: " << change_of_total_energy << "\n";

		const double lum_mean = change_of_total_energy / sim.tNew_[0]; // mean luminosity, erg/s
		amrex::Print() << "Mean luminosity: " << lum_mean << " erg/s\n";
	}

	const int status = 0; // Initialize to success
	return status;


// 	// read output variables
// 	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
// 	const int nx = static_cast<int>(position.size());

// 	// compute error norm
// 	std::vector<double> erad(nx);
// 	std::vector<double> erad_exact(nx);
// 	std::vector<double> xs(nx);
// 	for (int i = 0; i < nx; ++i) {
// 		amrex::Real const x = position[i];
// 		xs.at(i) = x;
// 		erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
// 		double erad_sim = 0.0;
// 		for (int g = 0; g < Physics_Traits<ParticleRadiationProblem>::nGroups; ++g) {
// 			erad_sim += values.at(RadSystem<ParticleRadiationProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
// 		}
// 		erad.at(i) = erad_sim;
// 	}

// 	double err_norm = 0.;
// 	double sol_norm = 0.;
// 	for (int i = 0; i < nx; ++i) {
// 		err_norm += std::abs(erad[i] - erad_exact[i]);
// 		sol_norm += std::abs(erad_exact[i]);
// 	}

// 	const double rel_err_norm = err_norm / sol_norm;
// 	const double rel_err_tol = 0.01;
// 	int status = 1;
// 	if (rel_err_norm < rel_err_tol) {
// 		status = 0;
// 	}
// 	amrex::Print() << "Relative L1 norm = " << rel_err_norm << '\n';

// #ifdef HAVE_PYTHON
// 	// Plot results
// 	matplotlibcpp::clf();
// 	matplotlibcpp::ylim(0.0, 1.1);

// 	std::map<std::string, std::string> erad_args;
// 	std::map<std::string, std::string> erad_exact_args;
// 	erad_args["label"] = "numerical solution";
// 	erad_exact_args["label"] = "exact solution";
// 	erad_exact_args["linestyle"] = "--";
// 	matplotlibcpp::plot(xs, erad, erad_args);
// 	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
// 	matplotlibcpp::save("./radiation_streaming.pdf");
// #endif // HAVE_PYTHON

// 	// Cleanup and exit
// 	amrex::Print() << "Finished." << '\n';
// 	return status;
}
