//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionSpitzerGaussian.cpp
/// \brief Defines a test problem for Spitzer thermal conduction (kappa = kappa0*T^2.5) with a Gaussian IC.
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
#include <sstream>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

/** Spitzer thermal conduction test problem with Gaussian IC--
While there is no analytic solution for a Gaussian evolving under Spitzer conduction, we can compare against a
 high-resolution numerical reference solution. The reference solution was generated using nx=256 in 1D and compared
 against this the convergence rate should be close to -2.0. For higher convergence rates go upto nx=4096. Also note
 that since there is a natural discontinuity in the solution the convergence rate is expected to be -1.0 for the whole domain,
 but inside the discontinuity the convergence rate is expected to be -2.0. This test proves that the code is
 converges as expected for Spitzer thermal conduction.
 */

constexpr double Eint0 = 2.505e-8; // Gaussian peak (equivalent to T = 2.e8 K)
// gaussian_spitzer_highres.csv was generated with this floor; keep the IC consistent with that table.
constexpr double Efloor = Eint0 / 10.0;
const double rho0 = 0.1;		     // 1/cm^3
constexpr double sigma = 2.410685615625e+17; // width of the initial Gaussian, in cm (amr2-branch value)
struct ThermalConductionSpitzerGaussianProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionSpitzerGaussianProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionSpitzerGaussianProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionSpitzerGaussianProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
};

template <> void QuokkaSimulation<ThermalConductionSpitzerGaussianProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho = rho0 * C::m_p;	  // g/cm^3
	const amrex::Real sigma2 = sigma * sigma; // t = 0

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real erfx_low = std::erf(xlow / std::sqrt(2.0 * sigma2));
		const amrex::Real erfx_high = std::erf(xhigh / std::sqrt(2.0 * sigma2));
		const amrex::Real Eint = Efloor + Eint0 * (sigma * std::sqrt(M_PI / 2.0)) * (erfx_high - erfx_low) / dx[0];

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::internalEnergy_index) = Eint;
	});
}

template <>
void QuokkaSimulation<ThermalConductionSpitzerGaussianProblem>::computeReferenceSolution(amrex::MultiFab &ref,
											 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
											 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	amrex::Real const rho = rho0 * C::m_p; // g/cm^3

	// There is no analytic solution for a Gaussian evolving under Spitzer conduction, so compare against a
	// tabulated high-resolution numerical reference instead.
	std::string const filename = "../extern/problems/ThermalConductionSpitzerGaussian/gaussian_spitzer_highres.csv";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fstream.is_open(), "Could not open gaussian_spitzer_highres.csv");

	std::string header;
	std::getline(fstream, header);

	std::vector<amrex::Real> x_host;
	std::vector<amrex::Real> Eint_host;
	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::string field;
		std::vector<double> vals;
		while (std::getline(iss, field, ',')) {
			vals.push_back(std::stod(field));
		}
		x_host.push_back(vals.at(0));
		Eint_host.push_back(vals.at(1));
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(x_host.size() >= 3, "gaussian_spitzer_highres.csv must contain at least 3 rows");

	amrex::Gpu::DeviceVector<amrex::Real> x_ref(x_host.size());
	amrex::Gpu::DeviceVector<amrex::Real> Eint_ref(Eint_host.size());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, x_host.begin(), x_host.end(), x_ref.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Eint_host.begin(), Eint_host.end(), Eint_ref.begin());
	amrex::Gpu::streamSynchronize();

	amrex::Real const *x_ref_ptr = x_ref.dataPtr();
	amrex::Real const *Eint_ref_ptr = Eint_ref.dataPtr();
	int const n_ref = static_cast<int>(x_ref.size());

	// restrict the error norm to |x| < 0.2 pc: error away from the core is dominated by edge/floor effects
	amrex::Real const error_mask_radius = 0.2 * 3.0856775814913673e18;

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const &state = state_new_cc_[0].const_array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

			if (std::abs(x) < error_mask_radius) {
				// clamp queries outside the tabulated x-range to the nearest tabulated edge value
				// (can occur at test resolutions finer than the table's own grid)
				amrex::Real const Eint_exact = interpolate_value<BoundaryPolicy::Clamp>(x, x_ref_ptr, Eint_ref_ptr, n_ref);

				for (int n = 0; n < ncomp; ++n) {
					stateExact(i, j, k, n) = 0.;
				}

				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::density_index) = rho;
				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::energy_index) = Eint_exact;
				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::internalEnergy_index) = Eint_exact;
				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::x1Momentum_index) = 0.0;
				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::x2Momentum_index) = 0.;
				stateExact(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::x3Momentum_index) = 0.;
			} else {
				// outside the region of interest: copy the simulated state so this cell contributes zero error
				for (int n = 0; n < ncomp; ++n) {
					stateExact(i, j, k, n) = state(i, j, k, n);
				}
			}
		});
	}
	amrex::Gpu::streamSynchronize();
}

auto runConductionTest(int nx) -> double
{
	constexpr double max_time = 660942.2643981476;
	constexpr int max_level = 0;

	// Set grid dimensions using AMReX parameter system
	amrex::ParmParse pp("amr");
	amrex::Vector<int> const ncells = {nx, nx, nx};
	pp.add("max_level", max_level);
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
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionSpitzerGaussianProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionSpitzerGaussianProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();
	return sim.computeErrorNorm();
}

template <>
void QuokkaSimulation<ThermalConductionSpitzerGaussianProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
										  amrex::MultiFab const &state_cc,
										  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<ThermalConductionSpitzerGaussianProblem>::density_index);
				Real const Eint = HydroSystem<ThermalConductionSpitzerGaussianProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<ThermalConductionSpitzerGaussianProblem>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	amrex::Vector<int> const resolutions = {32, 64, 128};
	amrex::Vector<double> errors;
	for (int nx : resolutions) {
		double const error = runConductionTest(nx);
		errors.push_back(error);
		amrex::Print() << std::format("nx = {:4d}  error norm = {:.6e}\n", nx, error);
	}

	// Best-fit slope of log(error) vs log(Nx) via ordinary least squares.
	double sum_x = 0.0;
	double sum_y = 0.0;
	double sum_xx = 0.0;
	double sum_xy = 0.0;
	int const n = static_cast<int>(resolutions.size());
	for (int i = 0; i < n; ++i) {
		double const log_nx = std::log(static_cast<double>(resolutions[i]));
		double const log_err = std::log(errors[i]);
		sum_x += log_nx;
		sum_y += log_err;
		sum_xx += log_nx * log_nx;
		sum_xy += log_nx * log_err;
	}
	double const mean_x = sum_x / n;
	double const mean_y = sum_y / n;
	double const slope = (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x);
	double const intercept = mean_y - slope * mean_x;
	amrex::Print() << std::format("\nBest-fit line: log(error) = {:.4f} * log(Nx) + {:.4f}\n", slope, intercept);

	constexpr double expectedRate = 2.0;
	constexpr double tolerance = 0.3;
	constexpr double passThreshold = -(expectedRate - tolerance);
	amrex::Print() << std::format(
	    "Spitzer+Gaussian conduction convergence: slope = {:.4f} ({:.1f} expected, converging faster is fine, pass threshold = {:.4f})\n", slope,
	    -expectedRate, passThreshold);
	bool const passed = slope <= passThreshold;

	if (passed) {
		amrex::Print() << "\n✓ Thermal conduction (spitzer, Gaussian) test PASSED\n";
		return 0;
	}
	amrex::Print() << "\n✗ Thermal conduction (spitzer, Gaussian) test FAILED\n";
	return 1;
}
