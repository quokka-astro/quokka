//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroVacuum.cpp
/// \brief Defines a test problem for a shock tube.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <format>
#include <fstream>

#include "AMReX_BLassert.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct ShocktubeProblem {
};

template <> struct quokka::EOS_Traits<ShocktubeProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ShocktubeProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
};

template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		double vx = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 0.5) {
			rho = 1.0;
			vx = -2.0;
			P = 0.4;
		} else {
			rho = 1.0;
			vx = 2.0;
			P = 0.4;
		}

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}

		auto const gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
							     const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// Number of variables (use Physics_Indices which correctly accounts for enabled physics)
	constexpr int nvar = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;

	// Left boundary values
	const double rho_L = 1.0;
	const double vx_L = -2.0;
	const double P_L = 0.4;
	const double E_L = P_L / (gamma - 1.) + 0.5 * rho_L * (vx_L * vx_L);

	// Prepare left boundary values
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	low_bdr_cells[RadSystem<ShocktubeProblem>::gasDensity_index] = rho_L;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x1GasMomentum_index] = rho_L * vx_L;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x3GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<ShocktubeProblem>::gasEnergy_index] = E_L;
	low_bdr_cells[RadSystem<ShocktubeProblem>::gasInternalEnergy_index] = P_L / (gamma - 1.);

	// Right boundary values
	const double rho_R = 1.0;
	const double vx_R = 2.0;
	const double P_R = 0.4;
	const double E_R = P_R / (gamma - 1.) + 0.5 * rho_R * (vx_R * vx_R);

	// Prepare right boundary values
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	for (int n = 0; n < nvar; ++n) {
		high_bdr_cells[n] = 0;
	}
	high_bdr_cells[RadSystem<ShocktubeProblem>::gasDensity_index] = rho_R;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x1GasMomentum_index] = rho_R * vx_R;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x2GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x3GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<ShocktubeProblem>::gasEnergy_index] = E_R;
	high_bdr_cells[RadSystem<ShocktubeProblem>::gasInternalEnergy_index] = P_R / (gamma - 1.);

	// Apply boundary conditions using helper functions (direction 0 = x-axis)
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

template <>
void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{

	auto const box = geom[0].Domain();
	int const nx = (box.hiVect3d()[0] - box.loVect3d()[0]) + 1;
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		xs.at(i) = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
	}

	// read in exact solution
	std::vector<double> xs_exact;
	std::vector<double> density_exact;
	std::vector<double> pressure_exact;
	std::vector<double> velocity_exact;
	std::vector<double> eint_exact;

	std::string const filename = "../extern/Toro/e1rpex.out";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		auto x = values.at(0);
		auto density = values.at(1);
		auto velocity = values.at(2);
		auto pressure = values.at(3);
		auto eint = pressure / ((quokka::EOS_Traits<ShocktubeProblem>::gamma - 1.0) * density);

		xs_exact.push_back(x);
		density_exact.push_back(density);
		pressure_exact.push_back(pressure);
		velocity_exact.push_back(velocity);
		eint_exact.push_back(eint);
	}

	std::vector<double> density_exact_interp(xs.size());
	interpolate_arrays(xs.data(), density_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), density_exact.data(),
			   static_cast<int>(xs_exact.size()));

	std::vector<double> velocity_exact_interp(xs.size());
	interpolate_arrays(xs.data(), velocity_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), velocity_exact.data(),
			   static_cast<int>(xs_exact.size()));

	std::vector<double> pressure_exact_interp(xs.size());
	interpolate_arrays(xs.data(), pressure_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), pressure_exact.data(),
			   static_cast<int>(xs_exact.size()));

	std::vector<double> eint_exact_interp(xs.size());
	interpolate_arrays(xs.data(), eint_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), eint_exact.data(),
			   static_cast<int>(xs_exact.size()));

	amrex::Gpu::AsyncVector<double> rho_g(density_exact_interp.size());
	amrex::Gpu::AsyncVector<double> vx_g(velocity_exact_interp.size());
	amrex::Gpu::AsyncVector<double> P_g(pressure_exact_interp.size());

	// copy exact solution to device
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
	amrex::Gpu::streamSynchronizeAll();

	// fill reference solution multifab
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		auto const &rho_arr = rho_g.data();
		auto const &vx_arr = vx_g.data();
		auto const &P_arr = P_g.data();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}
			amrex::Real const rho = rho_arr[i];
			amrex::Real const vx = vx_arr[i];
			amrex::Real const P = P_arr[i];

			const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
		});
	}

#ifdef HAVE_PYTHON

	// Plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> e(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real const rho = values.at(HydroSystem<ShocktubeProblem>::density_index)[i];
			amrex::Real const xmom = values.at(HydroSystem<ShocktubeProblem>::x1Momentum_index)[i];
			amrex::Real const Egas = values.at(HydroSystem<ShocktubeProblem>::energy_index)[i];

			amrex::Real const xvel = xmom / rho;
			amrex::Real const Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real const eint = Eint / rho;

			d.at(i) = rho;
			vx.at(i) = xvel;
			e.at(i) = eint;
		}

		// Plot results
		int const s = 12; // stride
		matplotlibcpp::clf();
		std::map<std::string, std::string> d_args;
		std::unordered_map<std::string, std::string> dexact_args;
		d_args["label"] = "simulation";
		d_args["color"] = "C0";
		dexact_args["label"] = "exact solution";
		dexact_args["marker"] = "o";
		dexact_args["color"] = "C0";
		// dexact_args["edgecolors"] = "k";
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::scatter(strided_vector_from(xs_exact, s), strided_vector_from(density_exact, s), 5.0, dexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::ylabel("density");
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		// matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::save(std::format("./hydro_vacuum_{:.4f}.pdf", tNew_[0]));

		// internal energy plot
		matplotlibcpp::clf();
		std::map<std::string, std::string> e_args;
		std::unordered_map<std::string, std::string> eexact_args;
		e_args["label"] = "simulation";
		e_args["color"] = "C5";
		eexact_args["label"] = "exact solution";
		eexact_args["marker"] = "o";
		eexact_args["color"] = "C5";
		// eexact_args["edgecolors"] = "k";
		matplotlibcpp::plot(xs, e, e_args);
		matplotlibcpp::scatter(strided_vector_from(xs_exact, s), strided_vector_from(eint_exact, s), 5.0, eexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::ylabel("specific internal energy");
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		// matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::save(std::format("./hydro_vacuum_eint_{:.4f}.pdf", tNew_[0]));
	}
#endif
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 100;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double max_time = 0.15;
	const double max_dt = 1e-3;
	const int max_timesteps = 5000;

	QuokkaSimulation<ShocktubeProblem> sim;

	sim.cflNumber_ = CFL_number;
	sim.maxDt_ = max_dt;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	sim.plotfileInterval_ = -1;

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	// Compute test success condition
	int status = 0;
	const double error_tol = 0.015;
	amrex::Real const error_norm = sim.computeErrorNorm();
	if (error_norm > error_tol) {
		status = 1;
	}

	return status;
}
