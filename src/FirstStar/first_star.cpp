/// \file first_star.cpp
/// \brief 
///

#include "first_star.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

namespace fs = std::filesystem;

struct TheProblem {
};

// constexpr const char* subfolder = __DATE__ " " __TIME__;
constexpr const char* subfolder = "figures";
bool test_passes = false; // if one of the energy checks fails, set to false

// model 1: 1/(1 + n^s) + 1/(1 + n1^s) * (n / n1)^(gamma - 1)
constexpr int model = 1;

constexpr double pi = M_PI;
constexpr double k_B = 1.0;
constexpr double c_iso = 1.0;
constexpr double A = 2.2;	// xi = 1.45 according to Table 1 of Shu77
constexpr double q = 0.2;
constexpr double r_c = 1.0;
constexpr double r_star = q * r_c;
constexpr double G = 1.0;
constexpr double h = 0.1;
constexpr double rho_star = A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);

// EOS parameters
constexpr double mu = 1.0;
constexpr double gamma_ = 5.0 / 3.0;
constexpr double Cv = 1.0 / (gamma_ - 1.0) * k_B / mu; // Specific heat at constant volume in the adiabatic phase
constexpr double T0 = c_iso * c_iso;

// model 1 parameters
constexpr double rho0 = rho_star;
constexpr double rho1 = 3.2 * rho0; // jump ends at this density
constexpr double s = 4.0; // jump slope. The jump equals s * log10(rho1 / rho0)

// model 2 parameters
constexpr double rho_core = 1.0;
constexpr double rho_one = 1000.0;
constexpr double jump_slope = 10.0;

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

AMREX_GPU_HOST_DEVICE
auto compute_T(const double rho) -> double
{
	if constexpr (model == 1) {
		const double scale = 1.0 / (1.0 + std::pow(rho / rho0, s)) + 1.0 / (1.0 + std::pow(rho1 / rho0, s)) * std::pow(rho / rho1, gamma_ - 1.0);
		return scale * T0;
	} else if constexpr (model == 2) {
		const double scale = 1.0 / (1.0 + std::exp(jump_slope * (rho / rho_core - 1.0))) + std::pow(rho / rho_one, gamma_ - 1.0);
		return scale * T0;
	}
}

AMREX_GPU_HOST_DEVICE
auto compute_e(const double rho) -> double
{
	double Tgas = NAN;
	double e = NAN;
	if constexpr (model == 1) {
		Tgas = compute_T(rho);
		e = Cv * rho * Tgas;
		return e;
	} else if constexpr (model == 2) {
		if (rho >= rho_core) {
			Tgas = compute_T(rho);
			return Cv * rho * Tgas;
		}
		Tgas = compute_T(rho_core);
		return Cv * rho_core * Tgas;
	}
}

// redefine EOS::ComputePressure
template <> 
AMREX_GPU_HOST_DEVICE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputePressure(amrex::Real rho, amrex::Real /*Eint*/, 
											const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/) -> amrex::Real
{
	if constexpr (model == 1) {
		const double e = compute_e(rho);
		return (gamma_ - 1.0) * e;
	} else {
		const double T = compute_T(rho);
		const double e = rho * Cv * T;
		return (gamma_ - 1.0) * e;
	}
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeTgasFromEint(amrex::Real rho, amrex::Real /*Egas*/,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_T(rho);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromTgas(amrex::Real rho, amrex::Real /*Tgas*/,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_e(rho);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_e(rho);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeSoundSpeed(amrex::Real rho, amrex::Real /*Pressure*/,
										const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	const double T = compute_T(rho);
	return std::sqrt(k_B * T / mu);
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

		const double rho_bg = rho_star * std::pow(r_star / r_c, 2);
		// const double r1 = 2.0;

		// compute density
		double rho = NAN;
		if (r <= r_star) {
			rho = rho_star;
		} else if (r <= r_c) {
			rho = rho_star * std::pow(r_star / r, 2);
		} else {
			rho = rho_bg;
		}
		const auto E_int = compute_e(rho);

		// compute azimuthal velocity
		double v_phi = 0.0;
		if (distxy <= r_star) {
			v_phi = 2 * A * c_iso * h;
			v_phi *= distxy / r_star;
		} else if (distxy <= r_c) {
			v_phi = 2 * A * c_iso * h;
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

template <> void RadhydroSimulation<TheProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of total energy
	amrex::Real const Egas0 = initSumCons[RadSystem<TheProblem>::gasEnergy_index];
	amrex::Real const Egas = state_new_cc_[0].sum(RadSystem<TheProblem>::gasEnergy_index) * vol;

	// compute kinetic energy
	amrex::MultiFab Ekin_mf(boxArray(0), DistributionMap(0), 1, 0);
	for (amrex::MFIter iter(state_new_cc_[0]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = state_new_cc_[0].const_array(iter);
		auto const &ekin = Ekin_mf.array(iter);
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			// compute kinetic energy
			Real rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
			Real px = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
			Real py = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
			Real pz = state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
			Real psq = px * px + py * py + pz * pz;
			ekin(i, j, k) = psq / (2.0 * rho) * vol;
		});
	}
	amrex::Real const Ekin = Ekin_mf.sum(0);

	amrex::Real const abs_err = (Egas - Egas0);
	amrex::Real const rel_err = abs_err / Egas0;

	amrex::Print() << "\nInitial energy = " << Egas0 << std::endl;
	amrex::Print() << "Final energy = " << Egas << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err << std::endl;
	amrex::Print() << std::endl;

	bool E_test_passes = false;  // does total energy test pass?

	if ((std::abs(rel_err) > 2.0e-13) || std::isnan(rel_err)) {
		// note that 2.0e-15 is appropriate for a 256^3 grid
		// it may need to be modified for coarser resolutions
		amrex::Print() << "Energy not conserved to machine precision!\n";
		E_test_passes = false;
	} else {
		amrex::Print() << "Energy conservation is OK.\n";
		E_test_passes = true;
	}

	// if both tests pass, then overall pass
  test_passes = E_test_passes;
}

template <> void RadhydroSimulation<TheProblem>::computeAfterTimestep(const int step)
{
  const int skip_plot = 100;
  if (step % skip_plot > 0) {
    return;
  }

	// read output variables
	// Extract the data at the final time at the center of the y-z plane (center=true) 
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Egas(nx);
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
		Egas.at(i) = Egas_t;
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
	// matplotlibcpp::save("./first-star-T.png");
	matplotlibcpp::save(fmt::format("./{}/first-star-T-s{:06d}-t{:.4g}.png", subfolder, step, tNew_[0]));

	// plot internal energy
	matplotlibcpp::clf();
	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas internal energy";
	Egas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Egas, Egas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("internal energy");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/first-star-E-s{:06d}-t{:.4g}.png", subfolder, step, tNew_[0]));

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
	// matplotlibcpp::save("./first-star-P.png");
	matplotlibcpp::save(fmt::format("./{}/first-star-P-s{:06d}-t{:.4g}.png", subfolder, step, tNew_[0]));

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
	// matplotlibcpp::save("./first-star-v.png");
	matplotlibcpp::save(fmt::format("./{}/first-star-v-s{:06d}-t{:.4g}.png", subfolder, step, tNew_[0]));

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
	// matplotlibcpp::save("./first-star-rho.png");
	matplotlibcpp::save(fmt::format("./{}/first-star-rho-s{:06d}-t{:.4g}.png", subfolder, step, tNew_[0]));
#endif

}

auto problem_main() -> int
{

	// Problem parameters

	const double max_dt = 1e0;

#if 1
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
#else
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<TheProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<TheProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<TheProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if constexpr (simulate_full_box) { // periodic boundaries
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			} else { // octant symmetry
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			}
		}
	}
#endif

	// Problem initialization
	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.doPoissonSolve_ = 1; // enable self-gravity

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.maxDt_ = max_dt;

	// initialize
	sim.setInitialConditions();

	// Check if the directory already exists
	const std::string directory_name = subfolder;
	if (fs::exists(directory_name)) {
		std::cout << "Directory already exists." << std::endl;
	} else {
		// Create the directory
		if (fs::create_directory(directory_name)) {
			std::cout << "Directory created successfully." << std::endl;
		} else {
			std::cerr << "Failed to create directory." << std::endl;
		}
	}

	// evolve
	// sim.evolve();

	// read output variables
	// Extract the data at the final time at the center of the y-z plane (center=true) 
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> xs(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Egas(nx);
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
		Egas.at(i) = Egas_t;
		Vgas.at(i) = v_t;
		pressure.at(i) = pressure_t;
	}

#if 0 // #ifdef HAVE_PYTHON
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
	matplotlibcpp::save(fmt::format("./first-star-T-{:.4g}.pdf", sim.tNew_[0]));
	// matplotlibcpp::save(fmt::format("./first-star-T-step{:d}.pdf", sim.nStep_));

	// plot internal energy
	matplotlibcpp::clf();
	std::map<std::string, std::string> Egas_args;
	Egas_args["label"] = "gas internal energy";
	Egas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Egas, Egas_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("internal energy");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./first-star-E-{:.4g}.pdf", sim.tNew_[0]));

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
	matplotlibcpp::save(fmt::format("./first-star-P-{:.4g}.pdf", sim.tNew_[0]));

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
	matplotlibcpp::save(fmt::format("./first-star-v-{:.4g}.pdf", sim.tNew_[0]));

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
	matplotlibcpp::save(fmt::format("./first-star-rho-{:.4g}.pdf", sim.tNew_[0]));
#endif

	// evolve
	sim.evolve();

	// Cleanup and exit
	int status = 1;
	if (test_passes) {
		status = 0;
	} else {
		status = 1;
	}
	return status;
}
