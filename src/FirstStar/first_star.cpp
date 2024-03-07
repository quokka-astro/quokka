/// \file first_star.cpp
/// \brief 
///

#include "first_star.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct TheProblem {
};

constexpr double pi = M_PI;

constexpr double mu = 1.0;
constexpr double k_B = 1.0;
constexpr double gamma_ = 5.0 / 3.0;
constexpr double c_iso = 1.0;
constexpr double Cv = 1.0 / (gamma_ - 1.0) * k_B / mu;

// EOS parameters
constexpr double rho_core = 1.0;
constexpr double rho_1 = 1.2;
constexpr double jump = 1.2;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = gamma_;
	// static constexpr double cs_isothermal = 1.0;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

// redefine EOS::ComputePressure
template <> 
AMREX_GPU_HOST_DEVICE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputePressure(amrex::Real rho, amrex::Real /*Eint*/, 
											const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/) -> amrex::Real
{
	const double p = std::log(jump) / std::log(rho_1 / rho_core);
	const double phase12 = 1.0 / (1.0 + std::pow(rho / rho_core, p));
	const double phase3 = std::pow(rho_core / rho_1, p) * std::pow(rho / rho_1, gamma_);
	return c_iso * c_iso * (phase12 + phase3);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure,
										const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	if (rho <= rho_core) {
		return c_iso;
	}
	return std::sqrt(Pressure / rho);
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
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const distxy = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		double A = 1.0;
		double G = 1.0;
		double h = 0.1;
		const double r_star = 0.5;
		const double rho_core = A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);

		const double rho_bg = 1.0e-3;
		const double r1 = 2.0;
		const double T0 = 1.0;

		// compute density
		double rho = NAN;
		if (r <= r_star) {
			rho = rho_core;
		} else if (r <= r1) {
			rho = rho_core * std::pow(r_star / r, 2);
		} else {
			rho = rho_bg;
		}
		const auto E_int = Cv * rho * T0;

		// compute azimuthal velocity
		double v_phi = 2 * A * c_iso * h;
		if (distxy <= r_star) {
			v_phi *= distxy / r_star;
		}

		// compute x, y, z velocity
		const double v_x = -v_phi * (y - y0) / distxy;
		const double v_y = v_phi * (x - x0) / distxy;
		const double v_z = 0.0;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = rho * v_x;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = rho * v_y;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = rho * v_z;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = E_int;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = E_int + 0.5 * rho * (v_x * v_x + v_y * v_y + v_z * v_z);
	});
}

template <> void RadhydroSimulation<TheProblem>::computeAfterTimestep()
{
	// read output variables
	// Extract the data at the final time at the center of the y-z plane (center=true) 
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> pressure(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto rho_t = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<TheProblem>::x2GasMomentum_index)[i] / rho_t;
		const auto Egas_t = values.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
		const auto pressure_t = (gamma_ - 1.0) * Egas_t;
		rhogas.at(i) = rho_t;
		Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas_t);
		Vgas.at(i) = v_t;
		pressure.at(i) = pressure_t;
	}

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("temperature");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./first-star-T.pdf");
	matplotlibcpp::save(fmt::format("./first-star-T-{:.4g}.pdf", tNew_[0]));

	// plot pressure
	matplotlibcpp::clf();
	std::map<std::string, std::string> pressure_args;
	pressure_args["label"] = "gas pressure";
	pressure_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, pressure, pressure_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("pressure");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./first-star-P.pdf");
	matplotlibcpp::save(fmt::format("./first-star-P-{:.4g}.pdf", tNew_[0]));

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("v_y");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./first-star-v.pdf");
	matplotlibcpp::save(fmt::format("./first-star-v-{:.4g}.pdf", tNew_[0]));

	// plot density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rhogas_args;
	rhogas_args["label"] = "gas density";
	rhogas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, rhogas, rhogas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("density");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./first-star-rho.pdf");
	matplotlibcpp::save(fmt::format("./first-star-rho-{:.4g}.pdf", tNew_[0]));
#endif

}

auto problem_main() -> int
{

	// // get the current date and time in the format YYYY-MM-DD.HH_mm_ss
	// const std::string currentDateTime()
	// {
	// 	time_t now = time(0);
	// 	struct tm tstruct;
	// 	char buf[80];
	// 	tstruct = *localtime(&now);
	// 	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
	// 	return buf;
	// }

	// Problem parameters

	const double max_dt = 1e0;

	// Boundary conditions
	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			// BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.doPoissonSolve_ = 1; // enable self-gravity

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.maxDt_ = max_dt;

	// initialize
	sim.setInitialConditions();

	// evolve
	// sim.evolve();

	// read output variables
	// Extract the data at the final time at the center of the y-z plane (center=true) 
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> pressure(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto rho_t = values.at(HydroSystem<TheProblem>::density_index)[i];
		const auto v_t = values.at(HydroSystem<TheProblem>::x2Momentum_index)[i] / rho_t;
		const auto Egas_t = values.at(HydroSystem<TheProblem>::internalEnergy_index)[i];
		const auto pressure_t = (gamma_ - 1.0) * Egas_t;
		rhogas.at(i) = rho_t;
		Tgas.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas_t);
		Vgas.at(i) = v_t;
		pressure.at(i) = pressure_t;
	}

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("temperature");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-T.pdf");

	// plot pressure
	matplotlibcpp::clf();
	std::map<std::string, std::string> pressure_args;
	pressure_args["label"] = "gas pressure";
	pressure_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, pressure, pressure_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("pressure");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-P.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("velocity");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-v.pdf");

	// plot density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rhogas_args;
	rhogas_args["label"] = "gas density";
	rhogas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, rhogas, rhogas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("density");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./first-star-rho.pdf");
#endif

	// evolve
	sim.evolve();

	// Cleanup and exit
	const int status = 0;
	return status;
}
