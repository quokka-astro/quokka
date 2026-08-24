//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConduction.cpp
/// \brief Defines a test problem for thermal conduction.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <cmath>
#include <fstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include "util/richardson.hpp"

/** Thermal conduction test problem
The runtime parameter conduction.conduction_type selects between two independent test setups, each
with its own initial condition and analytic reference solution, so a single build can validate both
physics modes (see ElectronConduction.hpp) just by changing that one parameter:
  - "isotropic": kappa = const. Initial condition is a smooth Gaussian temperature profile
    (cell-averaged in x via the erf antiderivative). Reference solution is the exact Gaussian
    diffusion solution (sigma_t^2 = sigma^2 + 2*D*t) -- identical to the original amr2-branch test.
  - "spitzer": kappa = kappa0*T^2.5. Initial condition is an instantaneous point source: only the
    cell(s) at x=0 start above the floor, everywhere else at Efloor. Reference solution is the
    Pattle (1959) self-similar nonlinear-diffusion solution. The total excess energy deposited (M0,
    see below) is fixed across resolutions, not the peak amplitude, so the deposit converges to a
    true Dirac-delta source and the same analytic solution is valid at every resolution (needed for
    the Richardson convergence test in problem_main/runConductionTest).
Physical parameters for the test problem are chosen to satisfy t_hydro / t_conduction >> 1, so that the gas does not have time to move
and the energy evolution is purely due to conduction. */

constexpr double Eint0 = 2.505e-8;		  // "isotropic": Gaussian peak. "spitzer": peak at the reference resolution nx_ref (both equivalent to T = 2.e8 K)
constexpr double Efloor = 5.674216387016754e-11; // equivalent to T = 2.e6 K
const double rho0 = 0.1;			  // 1/cm^3
constexpr double Lref = 7.714e+17;		  // quarter box length
constexpr double sigma = 2.410685615625e+17;	  // "isotropic" only: width of the initial Gaussian, in cm (amr2-branch value)
constexpr double D = 4.396303164750053e+28;	  // "isotropic" only: fixed diffusion coefficient for the Gaussian solution, in cm^2/s (amr2-branch value)
constexpr int nx_ref = 128;			  // "spitzer" only: resolution at which Eint0 is the deposited peak value (matches inputs/ThermalConduction.toml)
constexpr double dx0_ref = 4.0 * Lref / nx_ref;  
constexpr double M0 = (Eint0 - Efloor) * 2.0 * dx0_ref;
struct ThermalConductionProblem {
};

template <> struct quokka::EOS_Traits<ThermalConductionProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
};

template <> void QuokkaSimulation<ThermalConductionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a ThermalConduction test problem using parameters from

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const bool isIsotropic = (conductionType_ == "isotropic");
	const bool isSpitzer = (conductionType_ == "spitzer");
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		/*-------------------------------*/
		const amrex::Real rho = rho0 * C::m_p; // g/cm^3
		amrex::Real Eint = Efloor;
		if (isIsotropic) {
			// Gaussian temperature profile, cell-averaged in x via the erf antiderivative.
			const amrex::Real sigma2 = sigma * sigma;
			const amrex::Real erfx_low = std::erf(xlow / std::sqrt(2.0 * sigma2));
			const amrex::Real erfx_high = std::erf(xhigh / std::sqrt(2.0 * sigma2));
			Eint = Eint0 * (sigma * std::sqrt(M_PI / 2.0)) * (erfx_high - erfx_low) / dx[0] + Efloor;
		} else if (isSpitzer) {
			// Instantaneous point source: only the cell(s) at x=0 start above the floor,
			// everywhere else starts at the floor. The deposited amplitude is set so the total
			// excess energy (M0) is the same at every resolution -- see the comment on M0 above.
			Eint = (std::abs(x) < dx[0]) ? (Efloor + M0 / (2.0 * dx[0])) : Efloor;
		}
		/*-------------------------------*/

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}
		if(i==64 ){
			amrex::Print() << "Initial conditions at the center of the domain: " << std::endl;
			amrex::Print() << "Density: " << rho << std::endl;
			amrex::Print() << "Internal Energy: " << Eint << std::endl;
		}
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within one-sigma of the initial Gaussian profile for refinement
	const double refine_Lmax = Lref;

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real y0 = 0.0;
		amrex::Real y1 = 1.0;
		amrex::Real z0 = 0.0;
		amrex::Real z1 = 1.0;

#if AMREX_SPACEDIM >= 2
		y0 = prob_lo[1] + (j * dx[1]);
		y1 = prob_lo[1] + ((j + 1) * dx[1]);
#endif
#if AMREX_SPACEDIM == 3
		z0 = prob_lo[2] + (k * dx[2]);
		z1 = prob_lo[2] + ((k + 1) * dx[2]);
#endif

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			bool const in_region = (std::abs(x) < refine_Lmax);

			amrex::ignore_unused(y, z); // avoids unused-variable warnings in 1D

			if (in_region) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <>
void QuokkaSimulation<ThermalConductionProblem>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in,
								   amrex::MultiFab const &state_cc,
								   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	const int ncomp = ncomp_in;
	auto const &output = mf.arrays();
	auto const &state = state_cc.const_arrays();
	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "diagnostics require resampled cooling tables.");
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ThermalConductionProblem>::density_index);
			Real const Eint = HydroSystem<ThermalConductionProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
			output[bx](i, j, k, ncomp) = quokka::EOS<ThermalConductionProblem>::ComputeTgasFromEint(rho, Eint);
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

template <>
void QuokkaSimulation<ThermalConductionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real t = tNew_[0];
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3

	const bool isIsotropic = (conductionType_ == "isotropic");
	const bool isSpitzer = (conductionType_ == "spitzer");

	amrex::Real sigma2_t = 0.0;
	amrex::Real A = 0.0;
	amrex::Real q = 0.0;
	amrex::Real r1 = 0.0;
	amrex::Real Tscale = 0.0;

	if (isIsotropic) {
		// Exact Gaussian diffusion solution, cell-averaged in x via the erf antiderivative --
		// identical to the original amr2-branch reference solution, using the fixed sigma/D
		// above (not derived from conductivity_prefactor).
		const amrex::Real sigma2_0 = sigma * sigma;
		sigma2_t = sigma2_0 + 2.0 * D * t;
	} else if (isSpitzer) {
		// Pattle (1959) self-similar solution for nonlinear diffusion with D(T) ~ T^q
		// (q = 5/2 for Spitzer conduction); compactly supported, front at r1. Works in
		// temperature space and converts to Eint at the end: Eint = A*T exactly for this EOS
		// (gamma=2 => (gamma-1)=1, mean_molecular_weight=m_u => mu=1), with A taken from the
		// EOS itself (Eint at T=1 K) rather than hand-derived, so this stays correct even if
		// the EOS backend's internal conventions change. Assumes a zero background; the sim's
		// floor (Efloor) is added back onto the final profile, exact (not approximate) since
		// Eint=A*T has no additive offset. M0 (fixed across resolutions, see above) is what
		// makes this same analytic solution valid at every resolution.
		A = quokka::EOS<ThermalConductionProblem>::ComputeEintFromTgas(rho, 1.0); // dEint/dT
		const amrex::Real D0 = electronConductionKappa0_ / A;			    // D(T) = D0 * T^q
		const amrex::Real Q0 = M0 / A;						    // excess-T quantity released
		q = 2.5;
		const amrex::Real Gamma_num = std::tgamma(1.0 / q + 1.5);
		const amrex::Real Gamma_den = std::tgamma(1.0 / q + 1.0);
		const amrex::Real r0 = (Q0 / std::sqrt(M_PI)) * Gamma_num / Gamma_den;
		const amrex::Real t0 = q * r0 * r0 / (2.0 * (q + 2.0) * D0);
		r1 = r0 * std::pow(t / t0, 1.0 / (q + 2.0));
		Tscale = std::pow(t / t0, -1.0 / (q + 2.0));
	}

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const xlow = prob_lo[0] + i * dx[0];
			amrex::Real const xhigh = prob_lo[0] + (i + 1) * dx[0];
			amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

			amrex::Real Eint_exact = Efloor;
			if (isIsotropic) {
				amrex::Real const erfx_low = std::erf(xlow / std::sqrt(2.0 * sigma2_t));
				amrex::Real const erfx_high = std::erf(xhigh / std::sqrt(2.0 * sigma2_t));
				Eint_exact += Eint0 * (sigma * std::sqrt(M_PI / 2.0)) * (erfx_high - erfx_low) / dx[0];
			} else if (isSpitzer) {
				// Cell-centre evaluation: Pattle's compactly-supported profile has no simple
				// closed-form cell average.
				if (std::abs(x) <= r1) {
					amrex::Real const base = 1.0 - (x / r1) * (x / r1);
					Eint_exact += A * std::pow(base, 1.0 / q) * Tscale;
				}
			}

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			// fill gas components
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::energy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionProblem>::x3Momentum_index) = 0.;
		});
	}
}

auto runConductionTest(int nx, int /*ny*/, int /*nz*/) -> double
{
	// Read problem parameters
	const double max_time = 469054.0075444166; // 1 conduction time

	const double CFL_number = 0.3;
	const int max_timesteps = std::max(2000, nx * 100);

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, nx, nx};
	pp.add("max_level", 1);
	pp.addarr("n_cell", ncells);

	// Set domain bounds using AMReX parameter system
	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<double> const prob_lo = {-1.5428e18, -1.5428e18, -1.5428e18};
	amrex::Vector<double> const prob_hi = {1.5428e+18, 1.5428e+18, 1.5428e+18};
	amrex::Vector<int> const is_periodic = {0, 0, 0};
	pp_geom.addarr("prob_lo", prob_lo);
	pp_geom.addarr("prob_hi", prob_hi);
	pp_geom.addarr("is_periodic", is_periodic);

	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	const int nvars_fc = Physics_Indices<ThermalConductionProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::foextrap); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::foextrap);
		}
	}
	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc, BCs_fc);

	sim.cflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}

auto problem_main() -> int
{
	// boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		BCs_cc[n].setLo(dir, amrex::BCType::foextrap);  
		BCs_cc[n].setHi(dir, amrex::BCType::foextrap); 
		}
	}
	const int nvars_fc = Physics_Indices<ThermalConductionProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::foextrap); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::foextrap);
		}
	}
	// Problem initialization
	QuokkaSimulation<ThermalConductionProblem> sim(BCs_cc, BCs_fc);


	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	amrex::Print() << "Error norm: " << sim.computeErrorNorm() << '\n';

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;

	/***Richardson Extrapolation ****/

	// quokka::richardson::applyQuietDefaults();
	// quokka::richardson::Parameters params{};
	// params.machine_precision_target = 2.0e-9; // limit based on delta_b_magn, smaller values can be used if this is decreased
	// params.nx_initial = 128;
	// params.nx_max = 512;
	// params.expected_rate = 2.0;
	// params.tolerance = 0.3;
	// params.test_name = "Thermal Conduction";
	// params.csv_filename = "thermal_conduction_convergence.csv";

	// return quokka::richardson::run(params, [](int nx) { return runConductionTest(nx); });
}
