/// \file test_radhydro_pulse_grey.cpp
/// \brief Defines a 2D test problem for radiation in the diffusion regime with advection in medium with variable opacity under grey approximation.
///

#include "test_static_sphere_MG.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct TheProblem {
};

constexpr int variable_kappa = 0;
static constexpr bool export_csv = true;

constexpr double kappa_scale_up = 1.0e6;
constexpr double kappa_P_coeff = 3063.96 * kappa_scale_up;
constexpr double kappa_R_coeff = 101.248 * kappa_scale_up;
constexpr double T0 = 1.0e7; // K (temperature)
constexpr double T1 = 2.0e7; // K (temperature)
constexpr double rho0 = 1.2; // g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;
constexpr double width = 24.0; // cm, width of the pulse
constexpr double erad_floor = a_rad * T0 * T0 * T0 * T0 * 1.0e-10;
constexpr double mu = 2.33 * C::m_u;
constexpr double k_B = C::k_B;

// Default parameters: static diffusion, tau = 2e3, beta = 3e-5, beta tau = 6e-2
AMREX_GPU_MANAGED double kappa0 = 100. * kappa_scale_up;	 // NOLINT
AMREX_GPU_MANAGED double v0_adv = 1.0e6; // NOLINT
// AMREX_GPU_MANAGED double max_time = 4.8e-5; // max_time = 2.0 * width / v1;

// dynamic diffusion: tau = 2e4, beta = 3e-3, beta tau = 60
// constexpr double kappa0 = 1000.; // cm^2 g^-1
// constexpr double v0_adv = 1.0e8;    // advecting pulse
// constexpr double max_time = 1.2e-4; // max_time = 2.0 * width / v1;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

AMREX_GPU_HOST_DEVICE
auto compute_initial_Tgas(const double x) -> double
{
	// compute temperature profile for Gaussian radiation pulse
	const double sigma = width;
	return T0 + (T1 - T0) * std::exp(-x * x / (2.0 * sigma * sigma));
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_rho(const double x) -> double
{
	// compute density profile for Gaussian radiation pulse
	auto T = compute_initial_Tgas(x);
	return rho0 * T0 / T + (a_rad * mu / 3. / k_B) * (std::pow(T0, 4) / T - std::pow(T, 3));
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa{};
  if constexpr (variable_kappa) {
    const double sigma = kappa_P_coeff * std::pow(Tgas / T0, -3.5);
    kappa.fillin(sigma / rho);
  } else {
    kappa.fillin(kappa0);
  }
	return kappa;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa{};
  if constexpr (variable_kappa) {
    const double sigma = kappa_P_coeff * std::pow(Tgas / T0, -3.5);
    kappa.fillin(sigma / rho);
  } else {
    kappa.fillin(kappa0);
  }
	return kappa;
}

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
    auto const r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
		const double Trad = compute_initial_Tgas(r);
		const double Erad = a_rad * std::pow(Trad, 4);
		const double rho = compute_exact_rho(r);
		const double Egas = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho, Trad);
		const double v0 = v0_adv;

		// state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index) = (1. + 4. / 3. * (v0 * v0) / (c * c)) * Erad;
		state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index) = 4. / 3. * v0 * Erad;
		state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a test of grey radiation diffusion plus advection by gas.
	// This makes this problem a stringent test of the radiation advection
	// in the diffusion limit under grey approximation.

	// Problem parameters
	const double CFL_number = 0.8;
	// const int nx = 32;

	const double max_dt = 1e-3; // t_cr = 2 cm / cs = 7e-8 s

	// Boundary conditions
	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	amrex::ParmParse pp; // NOLINT
	pp.query("kappa0", kappa0);
	pp.query("v0_adv", v0_adv);

	// Problem 2: advecting radiation

	// Problem initialization
	RadhydroSimulation<TheProblem> sim2(BCs_cc);

	sim2.radiationReconstructionOrder_ = 3; // PPM
	sim2.radiationCflNumber_ = CFL_number;
	sim2.maxDt_ = max_dt;

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();

	// read output variables
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0, true);
	auto [position2y, values2y] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 1, 0.0, true);
	const int nx = static_cast<int>(position2.size());
	const int ny = static_cast<int>(position2y.size());
	auto prob_lo = sim2.geom[0].ProbLoArray();
	auto prob_hi = sim2.geom[0].ProbHiArray();
	// compute the pixel size
	const double dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx);
	const double move = v0_adv * sim2.tNew_[0];
	const int n_p = static_cast<int>(move / dx);
	const int half = static_cast<int>(nx / 2.0);
	const double drift = move - static_cast<double>(n_p) * dx;
	const int shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

	std::vector<double> xs2(nx);
	std::vector<double> Trad2(nx);
	std::vector<double> Tgas2(nx);
	std::vector<double> Vgas2(nx);
	std::vector<double> rhogas2(nx);

	std::vector<double> xs2y(nx);
	std::vector<double> Trad2y(nx);
	std::vector<double> Tgas2y(nx);
	std::vector<double> Vgas2y(nx);
	std::vector<double> rhogas2y(nx);

	for (int i = 0; i < nx; ++i) {
		int index_ = 0;
		if (shift >= 0) {
			if (i < shift) {
				index_ = nx - shift + i;
			} else {
				index_ = i - shift;
			}
		} else {
			if (i <= nx - 1 + shift) {
				index_ = i - shift;
			} else {
				index_ = i - (nx + shift);
			}
		}
		const amrex::Real x = position2[i];
		const auto Erad_t = values2.at(RadSystem<TheProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values2.at(RadSystem<TheProblem>::gasDensity_index)[i];
		const auto v_t = values2.at(RadSystem<TheProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values2.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
		xs2.at(i) = x - drift;
		rhogas2.at(index_) = rho_t;
		Trad2.at(index_) = Trad_t;
		Tgas2.at(index_) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas2.at(index_) = 1e-5 * (v_t - v0_adv);
	}

	for (int i = 0; i < ny; ++i) {
		const double x = position2y[i];
		const auto Erad_t = values2y.at(RadSystem<TheProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values2y.at(RadSystem<TheProblem>::gasDensity_index)[i];
		const auto v_t = values2y.at(RadSystem<TheProblem>::x2GasMomentum_index)[i] / rho_t;
		const auto Egas = values2y.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
		xs2y.at(i) = x;
		rhogas2y.at(i) = rho_t;
		Trad2y.at(i) = Trad_t;
		Tgas2y.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas2y.at(i) = 1e-5 * (v_t);
	}

	// Save xs, Trad, Tgas, rhogas, Vgas, xs_mg, Trad_mg, Tgas_mg, rhogas_mg, Vgas_mg, xs2, Trad2, Tgas2, rhogas2, Vgas2
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (export_csv) {
			std::ofstream file;
			// file.open("static_sphere.csv");
			file.open(fmt::format("./static_sphere_t{:.5g}.csv", sim2.tNew_[0]));
			file << "xs,rho,Trad,Tgas,Vgas\n";
			for (size_t i = 0; i < xs2.size(); ++i) {
				file << std::scientific << std::setprecision(12) << xs2[i] << "," << rhogas2[i] << "," << Trad2[i] << "," << Tgas2[i] << "," << Vgas2[i] << "\n";
			}
			file.close();
		}

#ifdef HAVE_PYTHON
	// plot velocity
	int const s = nx / 64; // stride
	std::map<std::string, std::string> args;
	args["label"] = "vx";
	args["color"] = "C0";
	matplotlibcpp::clf();
	matplotlibcpp::plot(xs2, Vgas2, args);
	args["label"] = "vy";
	args["linestyle"] = "--";
	args["color"] = "C1";
	matplotlibcpp::plot(xs2y, Vgas2y, args);
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("vx");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./static_sphere_vel_t{:.4g}.pdf", sim2.tNew_[0]));

	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "Tgas along x";
	Tgas_args["linestyle"] = "-";
	Tgas_args["color"] = "C0";
	matplotlibcpp::plot(xs2, Tgas2, Tgas_args);
	Tgas_args["label"] = "Tgas along y";
	Tgas_args["linestyle"] = "--";
	Tgas_args["color"] = "C1";
	matplotlibcpp::plot(xs2y, Tgas2y, Tgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::ylim(0.98e7, 2.02e7);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./radhydro_pulse_temperature_greynew.pdf");
	matplotlibcpp::save(fmt::format("./static_sphere_temperature_t{:.5g}.pdf", sim2.tNew_[0]));

	// plot gas density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["label"] = "density along x";
	rho_args["linestyle"] = "-";
	rho_args["color"] = "C0";
	matplotlibcpp::plot(xs2, rhogas2, rho_args);
	rho_args["label"] = "density along y";
	rho_args["linestyle"] = "--";
	rho_args["color"] = "C1";
	matplotlibcpp::plot(xs2y, rhogas2y, rho_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("density (g cm^-3)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
	matplotlibcpp::tight_layout();
	// save to file: density_{tNew_[0]}
	matplotlibcpp::save(fmt::format("./static_sphere_density_t{:.5g}.pdf", sim2.tNew_[0]));
#endif
	}

	// Cleanup and exit
	int status = 0;
	// if ((rel_error > error_tol) || std::isnan(rel_error) || (symm_rel_error > symm_err_tol) || std::isnan(symm_rel_error)) {
	// 	status = 1;
	// }
	return status;
}
