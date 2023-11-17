/// \file test_linear_diffusion_multigroup.cpp
/// \brief Defines a test problem for multigroup radiation in the diffusion regime with a frequecy-dependant opacity.
///

#include <limits>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "fextract.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"
#include "test_linear_diffusion_multigroup.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double a_rad = C::a_rad; // radiation constant
constexpr double c = C::c_light;   // speed of light

constexpr double rho0 = 1.8212111e-5;	   // g cm^-3 (matter density)
constexpr double T_ref = 1.1604448449e7; // K (1 keV)
constexpr double T0 = 0.1 * T_ref;
constexpr double x0 = 100448.496;	       // cm 
constexpr double C_v = 9.9968637e7;      // erg g^-1 K^-1
constexpr double mu = 1. / (5./3. - 1) * C::k_B / C_v; // mean molecular weight
constexpr double c_k = 4.0628337e43;     // cm^-1 Hz^3
constexpr double Egas0 = C_v * rho0 * T0;
constexpr double Egas_floor = 1e-10 * Egas0;
constexpr double Erad_floor_ = 1e-10 * a_rad * T0 * T0 * T0 * T0;

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 8; // number of radiation groups
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = mu;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr bool compute_v_over_c_terms = false;
	static constexpr double energy_unit = C::hplanck;
	static constexpr amrex::GpuArray<double, Physics_Traits<TheProblem>::nGroups + 1> radBoundaries{ 0.0,
    1.2089946e13,
    2.53888866e13,
    4.001772126e13,
    5.610943938600001e13,
    7.381032932460002e13,
    9.328130825706003e13,
    1.1469938508276605e14,
    1.3825926959104266e14};
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaVec{};
  double nu_g = NAN;
	for (int g = 0; g < nGroups_; ++g) {
    if (g == 0) {
      nu_g = 0.5 * RadSystem_Traits<TheProblem>::radBoundaries[1];
    } else {
      // take the geometrical mean
      nu_g = std::sqrt(RadSystem_Traits<TheProblem>::radBoundaries[g] * RadSystem_Traits<TheProblem>::radBoundaries[g + 1]);
    }
    auto kappa = c_k * std::pow(nu_g, -3);
		kappaVec[g] = kappa / rho;
	}
	return kappaVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

// const auto initial_Egas = 1e-10 * quokka::EOS<TheProblem>::ComputeEintFromTgas(rho0, T_hohlraum);
// const auto initial_Erad = 1e-10 * (a_rad * (T_hohlraum * T_hohlraum * T_hohlraum * T_hohlraum));

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto radBoundaries_g = RadSystem_Traits<TheProblem>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
    // calculate radEnergyFractions based on the boundary conditions
    auto radEnergyFractions = RadSystem<TheProblem>::ComputePlanckEnergyFractions(radBoundaries_g, T0);
    double Egas = Egas_floor;
    if (x < x0) {
      Egas = Egas0;
    } 

		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// TODO(ben): disable v/c terms for this problem!

	// Problem parameters

	// const int nx = 1500;
	const int max_timesteps = 200;
	const double CFL_number = 0.8;
	const double initial_dt = 5.8034112e-8;
	const double max_dt = 5.8034112e-8;
	const double max_time = 1.0;	// not used. We stop based on max_timesteps

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TheProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.maxDt_ = max_dt;
	sim.initDt_ = initial_dt;
	sim.plotfileInterval_ = -1;

	// evolve
	sim.setInitialConditions();
	// sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	int nx = static_cast<int>(position.size());

	// compare with exact solution
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			xs.at(i) = position[i];
			const auto Erad_t = values.at(RadSystem<TheProblem>::radEnergy_index)[i];
			const auto Etot_t = values.at(RadSystem<TheProblem>::gasEnergy_index)[i];
			const auto rho = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
			const auto x1GasMom = values.at(RadSystem<TheProblem>::x1GasMomentum_index)[i];

			Erad.at(i) = Erad_t;
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);
			const auto Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);
			const auto Egas_t = Etot_t - Ekin;
			Egas.at(i) = Egas_t;
			Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas_t);
		}

		std::vector<double> xs_exact = {0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5, 0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228, 5.62341};

		std::vector<double> Erad_diffusion_exact_0p1 = {0.09403, 0.09326, 0.09128, 0.08230, 0.06086, 0.04766, 0.03171,
								0.00755, 0.00064, 0.,	   0.,	    0.,	     0.};
		std::vector<double> Erad_transport_exact_0p1 = {0.09531, 0.09531, 0.09532, 0.09529, 0.08823, 0.04765, 0.00375, 0., 0., 0., 0., 0., 0.};

		std::vector<double> Erad_diffusion_exact_1p0 = {0.50359, 0.49716, 0.48302, 0.43743, 0.36656, 0.33271, 0.29029,
								0.18879, 0.10150, 0.04060, 0.01011, 0.00003, 0.};
		std::vector<double> Erad_transport_exact_1p0 = {0.64308, 0.63585, 0.61958, 0.56187, 0.44711, 0.35801, 0.25374,
								0.11430, 0.03648, 0.00291, 0.,	    0.,	     0.};

		std::vector<double> Egas_transport_exact_1p0 = {0.27126, 0.26839, 0.26261, 0.23978, 0.18826, 0.14187, 0.08838,
								0.03014, 0.00625, 0.00017, 0.,	    0.,	     0.};

		std::vector<double> Erad_diffusion_exact_3p1 = {0.95968, 0.95049, 0.93036, 0.86638, 0.76956, 0.72433,
								0.66672, 0.51507, 0.35810, 0.21309, 0.10047, 0.00634};
		std::vector<double> Erad_transport_exact_3p1 = {1.20052, 1.18869, 1.16190, 1.07175, 0.90951, 0.79902,
								0.66678, 0.44675, 0.27540, 0.14531, 0.05968, 0.00123};

		std::vector<double> Erad_diffusion_exact_10p0 = {1.86585, 1.85424, 1.82889, 1.74866, 1.62824, 1.57237, 1.50024,
								 1.29758, 1.06011, 0.79696, 0.52980, 0.12187, 0.00445};
		std::vector<double> Erad_transport_exact_10p0 = {2.23575, 2.21944, 2.18344, 2.06448, 1.86072, 1.73178, 1.57496,
								 1.27398, 0.98782, 0.70822, 0.45016, 0.09673, 0.00375};

		std::vector<double> Egas_transport_exact_10p0 = {2.11186, 2.09585, 2.06052, 1.94365, 1.74291, 1.61536, 1.46027,
								 1.16591, 0.88992, 0.62521, 0.38688, 0.07642, 0.00253};

		std::vector<double> Trad_exact_10(Erad_transport_exact_10p0);
		std::vector<double> Trad_exact_1(Erad_transport_exact_1p0);

		std::vector<double> Tgas_exact_10(Egas_transport_exact_10p0);
		std::vector<double> Tgas_exact_1(Egas_transport_exact_10p0);

		for (size_t i = 0; i < xs_exact.size(); ++i) {
			Trad_exact_10.at(i) = std::pow(Erad_transport_exact_10p0.at(i) / a_rad, 1. / 4.);
			Trad_exact_1.at(i) = std::pow(Erad_transport_exact_1p0.at(i) / a_rad, 1. / 4.);

			Tgas_exact_10.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho0, Egas_transport_exact_10p0.at(i));
			Tgas_exact_1.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho0, Egas_transport_exact_1p0.at(i));
		}

		// interpolate numerical solution onto exact solution tabulated points

		std::vector<double> Tgas_numerical_interp(xs_exact.size());
		interpolate_arrays(xs_exact.data(), Tgas_numerical_interp.data(), static_cast<int>(xs_exact.size()), xs.data(), Tgas.data(),
				   static_cast<int>(xs.size()));

		// compute L1 error norm
		double err_norm = 0.;
		double sol_norm = 0.;
		for (size_t i = 0; i < xs_exact.size(); ++i) {
			err_norm += std::abs(Tgas_numerical_interp[i] - Tgas_exact_10[i]);
			sol_norm += std::abs(Tgas_exact_10[i]);
		}
		const double rel_error = err_norm / sol_norm;
		const double error_tol = 0.03; // this will not agree to better than this, due to
					       // not being able to capture fEdd < 1/3 behavior
		amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;
		// if (rel_error > error_tol) {
		// 	status = 1;
		// }

#ifdef HAVE_PYTHON
		// Plot solution

		matplotlibcpp::clf();
		matplotlibcpp::xlim(0.2, 8.0); // cm

		std::map<std::string, std::string> Erad_args;
		Erad_args["label"] = "numerical solution";
		Erad_args["color"] = "black";
		matplotlibcpp::plot(xs, Erad, Erad_args);

		std::map<std::string, std::string> diffusion_args;
		diffusion_args["label"] = "exact diffusion solution";
		diffusion_args["color"] = "black";
		diffusion_args["linestyle"] = "none";
		diffusion_args["marker"] = "o";
		// diffusion_args["edgecolors"] = "k";
		// matplotlibcpp::plot(xs_exact, Erad_diffusion_exact_10p0, diffusion_args);

		std::map<std::string, std::string> transport_args;
		transport_args["label"] = "exact transport solution";
		transport_args["color"] = "black";
		transport_args["linestyle"] = "none";
		transport_args["marker"] = "x";
		// transport_args["edgecolors"] = "k";
		// matplotlibcpp::plot(xs_exact, Erad_transport_exact_10p0, transport_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::ylabel("radiation energy density");
		matplotlibcpp::xlim(0.0, 3.0); // cm
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./SuOlsonTest.pdf");

		// matplotlibcpp::xscale("log");
		// matplotlibcpp::yscale("log");
		// matplotlibcpp::xlim(0.2, 8.0); // cm
		// matplotlibcpp::ylim(1e-3, 3.0);
		// matplotlibcpp::save("./SuOlsonTest_loglog.pdf");
#endif
	}

	return status;
}
