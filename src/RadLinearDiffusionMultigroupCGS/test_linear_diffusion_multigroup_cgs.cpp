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
#include "test_linear_diffusion_multigroup_cgs.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double a_rad = C::a_rad; // radiation constant
constexpr double c = C::c_light;   // speed of light
constexpr double h = C::hplanck;
constexpr double k_B = C::k_B;

constexpr double rho0 = 1.8212111e-5;	   // g cm^-3 (matter density)
constexpr double T_keV = 1.1604448449e7; // K (1 keV)
constexpr double T_ref = 0.1 * T_keV;     // 0.1 keV
constexpr double T_f = 0.01 * T_keV;     // 0.1 keV
constexpr double T0 = T_ref;
constexpr double x0 = 100448.496;	       // cm 
constexpr double x_ref = x0 * 2.;
constexpr double C_v = 9.9968637e7;      // erg g^-1 K^-1
constexpr double mu = 1. / (5./3. - 1) * C::k_B / C_v; // mean molecular weight, = 1.238548409 m_p
constexpr double c_k = 4.0628337e43;     // cm^-1 Hz^3
constexpr double Egas0 = C_v * rho0 * T0;
constexpr double Egas_floor = 1e-8 * Egas0;
constexpr double Erad_floor_ = 1e-8 * a_rad * T0 * T0 * T0 * T0;    // Erad_floor_ / (a_r * T^4) has to be > 1e-10 for the Newton-Raphson iteration to work
// constexpr double Erad_floor_ = 1e-10 * a_rad;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

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
	// static constexpr int nGroups = 1; // number of radiation groups
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = mu;
	static constexpr double boltzmann_constant = C::k_B;
	// static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr bool compute_v_over_c_terms = true;
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
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeThermalRadiation(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<double, nGroups_> B_g{};
	for (int g = 0; g < nGroups_; ++g) {
	  B_g[g] = 2. * M_PI * h / (c * c) * std::pow(boundaries[g], 3) * (std::exp(- h * boundaries[g] / (k_B * T_f)) - std::exp(- h * boundaries[g + 1] / (k_B * T_f))) * temperature;
	}
  auto power = 4. * M_PI / c * B_g;   // = a_r * T^4
	return power;
}

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

// template <>
// AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
// AMRSimulation<TheProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
// 							   amrex::GeometryData const & /*geom*/, const amrex::Real /*time*/, const amrex::BCRec *bcr,
// 							   int /*bcomp*/, int /*orig_comp*/)
// {
// 	if (!((bcr->lo(0) == amrex::BCType::ext_dir) || (bcr->hi(0) == amrex::BCType::ext_dir))) {
// 		return;
// 	}

// #if (AMREX_SPACEDIM == 1)
// 	auto i = iv.toArray()[0];
// 	int j = 0;
// 	int k = 0;
// #endif
// #if (AMREX_SPACEDIM == 2)
// 	auto [i, j] = iv.toArray();
// 	int k = 0;
// #endif
// #if (AMREX_SPACEDIM == 3)
// 	auto [i, j, k] = iv.toArray();
// #endif

// 	if (i < 0) {
// 		// Marshak boundary condition
// 		const double T_H = T0;
// 		const double E_inc = a_rad * std::pow(T_H, 4);
// 		const double E_0 = consVar(0, j, k, RadSystem<TheProblem>::radEnergy_index);
// 		const double F_0 = consVar(0, j, k, RadSystem<TheProblem>::x1RadFlux_index);
//     const auto Egas = Egas0;

// 		// use value at interface to solve for F_rad in the ghost zones
// 		const double F_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * F_0);

// 		AMREX_ASSERT(std::abs(F_bdry / (c * E_inc)) < 1.0);

// 		// x1 left side boundary (Marshak)
// 		consVar(i, j, k, RadSystem<TheProblem>::radEnergy_index) = E_inc;
// 		consVar(i, j, k, RadSystem<TheProblem>::x1RadFlux_index) = F_bdry;
// 		consVar(i, j, k, RadSystem<TheProblem>::x2RadFlux_index) = 0.;
// 		consVar(i, j, k, RadSystem<TheProblem>::x3RadFlux_index) = 0.;
//     consVar(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
//     consVar(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho0;
//     consVar(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
//     consVar(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
//     consVar(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
//     consVar(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
// 	} else {
// 		// right-side boundary -- reflecting

//     const auto Egas = Egas_floor;

// 		consVar(i, j, k, RadSystem<TheProblem>::radEnergy_index) = Erad;
// 		consVar(i, j, k, RadSystem<TheProblem>::x1RadFlux_index) = 0;
// 		consVar(i, j, k, RadSystem<TheProblem>::x2RadFlux_index) = 0;
// 		consVar(i, j, k, RadSystem<TheProblem>::x3RadFlux_index) = 0;
//     consVar(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
//     consVar(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho0;
//     consVar(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
//     consVar(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
//     consVar(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
//     consVar(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
// 	}
// }


auto problem_main() -> int
{
	// TODO(ben): disable v/c terms for this problem!

	// Problem parameters

	// const int nx = 1500;
	const int max_timesteps = 100;
	// const int max_timesteps = 0;
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
	sim.evolve();

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
			xs.at(i) = position[i] / x_ref;
			const auto Erad_t = values.at(RadSystem<TheProblem>::radEnergy_index)[i];
			const auto Etot_t = values.at(RadSystem<TheProblem>::gasEnergy_index)[i];
			const auto rho = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
			const auto x1GasMom = values.at(RadSystem<TheProblem>::x1GasMomentum_index)[i];

			Erad.at(i) = Erad_t;
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.) / T_ref;
			const auto Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);
			const auto Egas_t = Etot_t - Ekin;
			Egas.at(i) = Egas_t;
			Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas_t) / T_ref;
		}

    // dummy print for debugging
    std::cout << "Hello! " << std::endl;

#ifdef HAVE_PYTHON
		// Plot solution

		matplotlibcpp::clf();
		matplotlibcpp::xlim(0.0, 1.0);    // cm
		// matplotlibcpp::ylim(-0.05, 1.05);   // dimensionless

		std::map<std::string, std::string> Trad_args;
		Trad_args["label"] = "numerical solution";
		Trad_args["color"] = "C1";
		matplotlibcpp::plot(xs, Trad, Trad_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x (cm)");
		matplotlibcpp::ylabel("radiation temperature");
    // title
    std::string title = "ct = " + std::to_string(sim.tNew_[0] * c / x_ref);
    matplotlibcpp::title(title);
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./LinearDiffusionMP_Trad.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::xlim(0.0, 1.0);    // cm
		matplotlibcpp::ylim(-0.05, 1.05);   // dimensionless

		Trad_args["label"] = "numerical solution";
		Trad_args["color"] = "C2";
		matplotlibcpp::plot(xs, Tgas, Trad_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x (cm)");
		matplotlibcpp::ylabel("gas temperature");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./LinearDiffusionMP_Tgas.pdf");
#endif
	}

	return status;
}
