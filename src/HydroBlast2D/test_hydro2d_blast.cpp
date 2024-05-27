//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "test_hydro2d_blast.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

struct TheProblem {
};

AMREX_GPU_MANAGED double kappa0 = 100.;
AMREX_GPU_MANAGED double v0_adv = 0.0;

constexpr double mu = 2.33 * C::m_u;
constexpr double rho0 = 1.2; // g cm^-3 (matter density)
constexpr double T0 = 1.0e7; // K (temperature)
constexpr double Cv = 3. / 2. * C::k_B / mu; // erg g^-1 K^-1 (specific heat capacity)
constexpr double E0 = Cv * rho0 * T0;		// erg g^-1 (internal energy density)
constexpr double a_rad = C::a_rad;
constexpr double erad_floor = a_rad * T0 * T0 * T0 * T0 * 1.0e-20;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = C::c_light;
	static constexpr double c_hat = C::c_light;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 1;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa{};
  kappa.fillin(kappa0);
	return kappa;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa{};
  kappa.fillin(kappa0);
	return kappa;
}

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		double vx = v0_adv;
		double vy = 0.;
		double vz = 0.;
		double rho = rho0;
		double Egas = 0.;

		if (R < 0.1) { // inside circle
			Egas = E0;
		} else {
			Egas = 0.1 * E0;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas + 0.5 * rho * v_sq;

		// initialize radiation variables to zero
		state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index) = erad_floor;
		state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index) = 0;
	});
}

template <> void RadhydroSimulation<TheProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = HydroSystem<TheProblem>::ComputePressure(state, i, j, k);
			amrex::Real const P_xplus = HydroSystem<TheProblem>::ComputePressure(state, i + 1, j, k);
			amrex::Real const P_xminus = HydroSystem<TheProblem>::ComputePressure(state, i - 1, j, k);
			amrex::Real const P_yplus = HydroSystem<TheProblem>::ComputePressure(state, i, j + 1, k);
			amrex::Real const P_yminus = HydroSystem<TheProblem>::ComputePressure(state, i, j - 1, k);

			amrex::Real const del_x = std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
			amrex::Real const del_y = std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));

			amrex::Real const gradient_indicator = std::max(del_x, del_y) / std::max(P, P_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	constexpr bool reflecting_boundary = true;

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

	const int ncomp_cc = RadhydroSimulation<TheProblem>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (reflecting_boundary) {
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			} else {
				// periodic
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			}
		}
	}

	// Problem initialization
	RadhydroSimulation<TheProblem> sim2(BCs_cc);

	double max_steps = 10;

	// read max_steps from inputs file
	amrex::ParmParse pp;
	pp.query("max_timestep", max_steps);

	sim2.radiationReconstructionOrder_ = 3; // PPM
	sim2.stopTime_ = 0.1; // 1.5;
	sim2.cflNumber_ = 0.3;
	sim2.radiationCflNumber_ = 0.3;
	sim2.maxTimesteps_ = max_steps;
	// sim2.plotfileInterval_ = 2000;

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();



	// read output variables
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0, true);
	// auto [position2y, values2y] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 1, 0.0, true);
	// const int nx = static_cast<int>(position2.size());
	// const int ny = static_cast<int>(position2y.size());
	// auto prob_lo = sim2.geom[0].ProbLoArray();
	// auto prob_hi = sim2.geom[0].ProbHiArray();
	// // compute the pixel size
	// const double dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx);
	// const double move = v0_adv * sim2.tNew_[0];
	// const int n_p = static_cast<int>(move / dx);
	// const int half = static_cast<int>(nx / 2.0);
	// const double drift = move - static_cast<double>(n_p) * dx;
	// const int shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

// 	std::vector<double> xs2(nx);
// 	std::vector<double> Trad2(nx);
// 	std::vector<double> Tgas2(nx);
// 	std::vector<double> Vgas2(nx);
// 	std::vector<double> rhogas2(nx);

// 	std::vector<double> xs2y(nx);
// 	std::vector<double> Trad2y(nx);
// 	std::vector<double> Tgas2y(nx);
// 	std::vector<double> Vgas2y(nx);
// 	std::vector<double> rhogas2y(nx);

// 	for (int i = 0; i < nx; ++i) {
// 		int index_ = 0;
// 		if (shift >= 0) {
// 			if (i < shift) {
// 				index_ = nx - shift + i;
// 			} else {
// 				index_ = i - shift;
// 			}
// 		} else {
// 			if (i <= nx - 1 + shift) {
// 				index_ = i - shift;
// 			} else {
// 				index_ = i - (nx + shift);
// 			}
// 		}
// 		const amrex::Real x = position2[i];
// 		const auto Erad_t = values2.at(RadSystem<TheProblem>::radEnergy_index)[i];
// 		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
// 		const auto rho_t = values2.at(RadSystem<TheProblem>::gasDensity_index)[i];
// 		const auto v_t = values2.at(RadSystem<TheProblem>::x1GasMomentum_index)[i] / rho_t;
// 		const auto Egas = values2.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
// 		xs2.at(i) = x - drift;
// 		rhogas2.at(index_) = rho_t;
// 		Trad2.at(index_) = Trad_t;
// 		Tgas2.at(index_) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas);
// 		Vgas2.at(index_) = 1e-5 * (v_t - v0_adv);
// 	}

// 	for (int i = 0; i < ny; ++i) {
// 		const double x = position2y[i];
// 		const auto Erad_t = values2y.at(RadSystem<TheProblem>::radEnergy_index)[i];
// 		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
// 		const auto rho_t = values2y.at(RadSystem<TheProblem>::gasDensity_index)[i];
// 		const auto v_t = values2y.at(RadSystem<TheProblem>::x2GasMomentum_index)[i] / rho_t;
// 		const auto Egas = values2y.at(RadSystem<TheProblem>::gasInternalEnergy_index)[i];
// 		xs2y.at(i) = x;
// 		rhogas2y.at(i) = rho_t;
// 		Trad2y.at(i) = Trad_t;
// 		Tgas2y.at(i) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas);
// 		Vgas2y.at(i) = 1e-5 * (v_t);
// 	}


// #ifdef HAVE_PYTHON
// 	// plot temperature
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> Trad_args;
// 	std::map<std::string, std::string> Tgas_args;
// 	Trad_args["label"] = "Trad (nonadvecting)";
// 	Trad_args["linestyle"] = "-.";
// 	Tgas_args["label"] = "Tgas (nonadvecting)";
// 	Tgas_args["linestyle"] = "--";
// 	Trad_args["label"] = "Trad (advecting)";
// 	Tgas_args["label"] = "Tgas (advecting)";
// 	matplotlibcpp::plot(xs2, Trad2, Trad_args);
// 	matplotlibcpp::plot(xs2, Tgas2, Tgas_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("temperature (K)");
// 	matplotlibcpp::ylim(0.98e7, 2.02e7);
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	// matplotlibcpp::save("./radhydro_pulse_temperature_greynew.pdf");
// 	matplotlibcpp::save(fmt::format("./blast2d_temperature_t{:.5g}.pdf", sim2.tNew_[0]));

// 	// plot gas density profile
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> rho_args;
// 	rho_args["label"] = "gas density (non-advecting)";
// 	rho_args["linestyle"] = "-";
// 	rho_args["label"] = "gas density (advecting))";
// 	matplotlibcpp::plot(xs2, rhogas2, rho_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("density (g cm^-3)");
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	// save to file: density_{tNew_[0]}
// 	matplotlibcpp::save(fmt::format("./blast2d_density_t{:.5g}.pdf", sim2.tNew_[0]));

// 	// plot gas velocity profile
// 	matplotlibcpp::clf();
// 	std::map<std::string, std::string> vgas_args;
// 	vgas_args["label"] = "gas velocity (non-advecting)";
// 	vgas_args["linestyle"] = "-";
// 	vgas_args["label"] = "gas velocity (advecting)";
// 	matplotlibcpp::plot(xs2, Vgas2, vgas_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("velocity (km s^-1)");
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	matplotlibcpp::save(fmt::format("./blast2d_velocity_t{:.5g}.pdf", sim2.tNew_[0]));

// 	// plot temperature Trad2y and Tgas2y
// 	matplotlibcpp::clf();
// 	Trad_args["label"] = "Trad (nonadvecting)";
// 	Trad_args["linestyle"] = "-.";
// 	Tgas_args["label"] = "Tgas (nonadvecting)";
// 	Tgas_args["linestyle"] = "--";
// 	Trad_args["label"] = "Trad (advecting)";
// 	Tgas_args["label"] = "Tgas (advecting)";
// 	matplotlibcpp::plot(xs2y, Trad2y, Trad_args);
// 	matplotlibcpp::plot(xs2y, Tgas2y, Tgas_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("temperature (K)");
// 	matplotlibcpp::ylim(0.98e7, 2.02e7);
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	// matplotlibcpp::save("./radhydro_pulse_temperature_greynew.pdf");
// 	matplotlibcpp::save(fmt::format("./blast2d_temperature_y_t{:.5g}.pdf", sim2.tNew_[0]));

// 	// plot gas density profile
// 	matplotlibcpp::clf();
// 	rho_args["label"] = "gas density (non-advecting)";
// 	rho_args["linestyle"] = "-";
// 	rho_args["label"] = "gas density (advecting))";
// 	matplotlibcpp::plot(xs2y, rhogas2y, rho_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("density (g cm^-3)");
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	// save to file: density_{tNew_[0]}
// 	matplotlibcpp::save(fmt::format("./blast2d_density_y_t{:.5g}.pdf", sim2.tNew_[0]));

// 	// plot gas velocity profile
// 	matplotlibcpp::clf();
// 	vgas_args["label"] = "gas velocity (non-advecting)";
// 	vgas_args["linestyle"] = "-";
// 	vgas_args["label"] = "gas velocity (advecting)";
// 	matplotlibcpp::plot(xs2y, Vgas2y, vgas_args);
// 	matplotlibcpp::xlabel("length x (cm)");
// 	matplotlibcpp::ylabel("velocity (km s^-1)");
// 	matplotlibcpp::legend();
// 	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
// 	matplotlibcpp::tight_layout();
// 	matplotlibcpp::save(fmt::format("./blast2d_velocity_y_t{:.5g}.pdf", sim2.tNew_[0]));

// #endif




	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
