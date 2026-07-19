//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroSMS.cpp
/// \brief Defines a test problem for a shock tube.
///

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "AMReX_BC_TYPES.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <format>
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include "util/BC.hpp"

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
		double m = NAN;
		double rho = NAN;
		double E = NAN;

		if (x < 0.5) {
			rho = 3.86;
			m = -3.1266;
			E = 27.0913;
		} else {
			rho = 1.0;
			m = -3.44;
			E = 8.4168;
		}

		double const Eint = E - 0.5 * (m * m) / rho;

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = m;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = E;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = Eint;
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

	// Left state
	const double rho_L = 3.86;
	const double m_L = -3.1266;
	const double E_L = 27.0913;
	const double Eint_L = E_L - 0.5 * (m_L * m_L) / rho_L;

	// Prepare left boundary values
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	low_bdr_cells[RadSystem<ShocktubeProblem>::gasDensity_index] = rho_L;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x1GasMomentum_index] = m_L;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<ShocktubeProblem>::x3GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<ShocktubeProblem>::gasEnergy_index] = E_L;
	low_bdr_cells[HydroSystem<ShocktubeProblem>::internalEnergy_index] = Eint_L;

	// Right state
	const double rho_R = 1.0;
	const double m_R = -3.44;
	const double E_R = 8.4168;
	const double Eint_R = E_R - 0.5 * (m_R * m_R) / rho_R;

	// Prepare right boundary values
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	for (int n = 0; n < nvar; ++n) {
		high_bdr_cells[n] = 0;
	}
	high_bdr_cells[RadSystem<ShocktubeProblem>::gasDensity_index] = rho_R;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x1GasMomentum_index] = m_R;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x2GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<ShocktubeProblem>::x3GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<ShocktubeProblem>::gasEnergy_index] = E_R;
	high_bdr_cells[HydroSystem<ShocktubeProblem>::internalEnergy_index] = Eint_R;

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

	// compute exact solution
	std::vector<double> density_exact;
	std::vector<double> pressure_exact;
	std::vector<double> velocity_exact;

	for (int i = 0; i < nx; ++i) {
		double vx = NAN;
		double rho = NAN;
		double P = NAN;
		const double vshock = 0.1096;
		amrex::Real const x = xs[i];

		if (x < (0.5 + vshock * tNew_[0])) {
			rho = 3.86;
			vx = -0.81;
			P = 10.3334;
		} else {
			rho = 1.0;
			vx = -3.44;
			P = 1.0;
		}
		density_exact.push_back(rho);
		pressure_exact.push_back(P);
		velocity_exact.push_back(vx);
	}

	amrex::Gpu::AsyncVector<double> rho_g(density_exact.size());
	amrex::Gpu::AsyncVector<double> vx_g(velocity_exact.size());
	amrex::Gpu::AsyncVector<double> P_g(pressure_exact.size());

	// copy exact solution to device
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact.begin(), density_exact.end(), rho_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact.begin(), velocity_exact.end(), vx_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact.begin(), pressure_exact.end(), P_g.begin());
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

			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) =
			    quokka::EOS<ShocktubeProblem>::ComputeEintFromPres(rho, P) + 0.5 * rho * (vx * vx);
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = quokka::EOS<ShocktubeProblem>::ComputeEintFromPres(rho, P);
		});
	}

	// Plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> P(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real const rho = values.at(HydroSystem<ShocktubeProblem>::density_index)[i];
			amrex::Real const xmom = values.at(HydroSystem<ShocktubeProblem>::x1Momentum_index)[i];
			amrex::Real const Egas = values.at(HydroSystem<ShocktubeProblem>::energy_index)[i];

			amrex::Real const xvel = xmom / rho;
			amrex::Real const Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real const pressure = quokka::EOS<ShocktubeProblem>::ComputePressure(rho, Eint);

			d.at(i) = rho;
			vx.at(i) = xvel;
			P.at(i) = pressure;
		}

#ifdef HAVE_PYTHON
		// Plot results
		matplotlibcpp::clf();
		int const s = 2; // stride
		std::map<std::string, std::string> d_args;
		std::unordered_map<std::string, std::string> dexact_args;
		d_args["label"] = "simulation";
		d_args["color"] = "C0";
		dexact_args["label"] = "exact solution";
		dexact_args["marker"] = "o";
		dexact_args["color"] = "C0";
		// dexact_args["edgecolors"] = "k";
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(density_exact, s), 3.0, dexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::tight_layout();
		// matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::save(std::format("./hydro_sms_{:.4f}.pdf", tNew_[0]));
#endif
	}
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 100;
	// const double Lx = 1.0;
	const double CFL_number = 0.2;
	const double max_time = 1.0;
	const int max_timesteps = 20000;

	QuokkaSimulation<ShocktubeProblem> sim;

	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.integratorOrder_ = 2;     // use forward Euler
	sim.reconstructionOrder_ = 3; // use donor cell

	sim.plotfileInterval_ = -1;

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	// Compute test success condition
	int status = 0;
	const double error_tol = 0.005;
	amrex::Real const error_norm = sim.computeErrorNorm();
	if (error_norm > error_tol) {
		status = 1;
	}

	return status;
}
