//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file star_cluster.cpp
/// \brief Defines a test problem for pressureless spherical collapse.
///
#include <limits>
#include <memory>
#include <random>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "star_cluster.hpp"

using amrex::Real;

struct StarCluster {
};

template <> struct quokka::EOS_Traits<StarCluster> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 0.1; // dimensionless
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
};

template <> struct HydroSystem_Traits<StarCluster> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StarCluster> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

template <> struct SimulationData<StarCluster> {
	std::unique_ptr<amrex::TableData<Real, 4>> dvx_modes;
	std::unique_ptr<amrex::TableData<Real, 4>> dvy_modes;
	std::unique_ptr<amrex::TableData<Real, 4>> dvz_modes;
	Real dv_rms{};
};

constexpr int kmin = 0;
constexpr int kmax = 16;

auto generateRandomModes(const int alpha_PL, const int seed) -> amrex::TableData<Real, 4>
{
	// generate random amplitudes and phases

	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	amrex::TableData<Real, 4> h_table_data(tlo, thi, amrex::The_Pinned_Arena());
	auto const &h_table = h_table_data.table();

	// use 64-bit Mersenne Twister (do not use 32-bit version for sampling doubles!)
	std::mt19937_64 rng(seed); // NOLINT
	std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);
	std::uniform_real_distribution<double> sample_unit(0., 1.0);

	Real rms = 0;
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				// compute wavenumber |k|
				const Real kx = static_cast<Real>(i);
				const Real ky = static_cast<Real>(j);
				const Real kz = static_cast<Real>(k);
				const Real k_abs = std::sqrt(kx * kx + ky * ky + kz * kz);

				// sample amplitude from Rayleigh distribution
				const Real q1 = sample_unit(rng);
				const Real q2 = sample_unit(rng);
				Real amp = std::sqrt(-2.0 * std::log(q1 + 1.0e-20)) * std::cos(2.0 * M_PI * q2);

				// apply power spectrum (note 4pi k^2 dk implies +2.0 inside pow)
				amp /= pow(k_abs, (alpha_PL + 2.0));

				if (i != 0 || j != 0 || k != 0) {
					rms += amp * amp;
					h_table(i, j, k, 0) = amp;
					h_table(i, j, k, 1) = sample_phase(rng);
				} else { // k == 0, set it to zero
					h_table(i, j, k, 0) = 0;
					h_table(i, j, k, 1) = 0;
				}
			}
		}
	}
	return h_table_data;
}

void projectModes(amrex::TableData<Real, 4> &dvx, amrex::TableData<Real, 4> &dvy, amrex::TableData<Real, 4> &dvz)
{
	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	auto const &dvx_table = dvx.table();
	auto const &dvy_table = dvy.table();
	auto const &dvz_table = dvz.table();

	// delete compressive modes
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				if (i != 0 || j != 0 || k != 0) {
					// compute k_hat = (kx, ky, kz)
					Real kx = std::sin(2.0 * M_PI * i / kmax);
					Real ky = std::sin(2.0 * M_PI * j / kmax);
					Real kz = std::sin(2.0 * M_PI * k / kmax);
					Real kabs = std::sqrt(kx * kx + ky * ky + kz * kz);
					kx /= kabs;
					ky /= kabs;
					kz /= kabs;

					Real vx = dvx_table(i, j, k, 0);
					Real vy = dvy_table(i, j, k, 0);
					Real vz = dvz_table(i, j, k, 0);
					Real v_dot_khat = vx * kx + vy * ky + vz * kz;

					// return v - (v dot k_hat) k_hat
					dvx_table(i, j, k, 0) -= v_dot_khat * kx;
					dvy_table(i, j, k, 0) -= v_dot_khat * ky;
					dvz_table(i, j, k, 0) -= v_dot_khat * kz;
				}
			}
		}
	}
}

auto computeRms(amrex::TableData<Real, 4> &dvx, amrex::TableData<Real, 4> &dvy, amrex::TableData<Real, 4> &dvz) -> Real
{
	amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
	amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
	auto const &dvx_table = dvx.const_table();
	auto const &dvy_table = dvy.const_table();
	auto const &dvz_table = dvz.const_table();

	// compute rms power
	Real rms_sq = 0;
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				Real vx = dvx_table(i, j, k, 0);
				Real vy = dvy_table(i, j, k, 0);
				Real vz = dvz_table(i, j, k, 0);
				rms_sq += vx * vx + vy * vy + vz * vz;
			}
		}
	}
	return std::sqrt(rms_sq);
}

template <> void RadhydroSimulation<StarCluster>::preCalculateInitialConditions()
{
	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		amrex::ParmParse pp("perturb");
		int alpha_PL = 0;
		pp.query("power_law", alpha_PL);

		amrex::Print() << "Generating velocity perturbations...\n";
		amrex::TableData<Real, 4> dvx = generateRandomModes(alpha_PL, 42);
		amrex::TableData<Real, 4> dvy = generateRandomModes(alpha_PL, 107);
		amrex::TableData<Real, 4> dvz = generateRandomModes(alpha_PL, 56);

		// keep soloinoidal modes only
		projectModes(dvx, dvy, dvz);

		// compute normalisation
		userData_.dv_rms = computeRms(dvx, dvy, dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms << "\n";

		// copy data to GPU memory
		amrex::Array<int, 4> tlo{kmin, kmin, kmin, 0};
		amrex::Array<int, 4> thi{kmax, kmax, kmax, 1};
		userData_.dvx_modes = std::make_unique<amrex::TableData<Real, 4>>(tlo, thi);
		userData_.dvy_modes = std::make_unique<amrex::TableData<Real, 4>>(tlo, thi);
		userData_.dvz_modes = std::make_unique<amrex::TableData<Real, 4>>(tlo, thi);
		userData_.dvx_modes->copy(dvx);
		userData_.dvy_modes->copy(dvy);
		userData_.dvz_modes->copy(dvz);
		amrex::Gpu::streamSynchronize();

		isSamplingDone = true;
	}
}

template <> void RadhydroSimulation<StarCluster>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &dvx_modes = userData_.dvx_modes->const_table();
	auto const &dvy_modes = userData_.dvy_modes->const_table();
	auto const &dvz_modes = userData_.dvz_modes->const_table();

	amrex::Real Lx = prob_hi[0] - prob_lo[0];
	amrex::Real Ly = prob_hi[1] - prob_lo[1];
	amrex::Real Lz = prob_hi[2] - prob_lo[2];

	amrex::Real x0 = prob_lo[0] + 0.5 * Lx;
	amrex::Real y0 = prob_lo[1] + 0.5 * Ly;
	amrex::Real z0 = prob_lo[2] + 0.5 * Lz;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		double rho_min = 1.0e-5;
		double rho_max = 10.0;
		double R_sphere = 0.5;
		double R_smooth = 0.025;
		double rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		AMREX_ASSERT(!std::isnan(rho));

		double vx = 0;
		double vy = 0;
		double vz = 0;

		for (int kx = kmin; kx < kmax; ++kx) {
			for (int ky = kmin; ky < kmax; ++ky) {
				for (int kz = kmin; kz < kmax; ++kz) {
					const Real theta = (2.0 * M_PI) * (kx * (x / Lx) + ky * (y / Ly) + kz * (z / Lz));

					// x-velocity
					Real amp = dvx_modes(kx, ky, kz, 0);
					Real phase = dvx_modes(kx, ky, kz, 1);
					vx += amp * std::cos(theta + phase);

					// y-velocity
					amp = dvy_modes(kx, ky, kz, 0);
					phase = dvy_modes(kx, ky, kz, 1);
					vy += amp * std::cos(theta + phase);

					// x-velocity
					amp = dvz_modes(kx, ky, kz, 0);
					phase = dvz_modes(kx, ky, kz, 1);
					vz += amp * std::cos(theta + phase);
				}
			}
		}

		double rms_dv_target = 0.5;
		double rms_dv_actual = userData_.dv_rms;
		double renorm_amp = rms_dv_target / rms_dv_actual;
		vx *= renorm_amp;
		vy *= renorm_amp;
		vz *= renorm_amp;

		state_cc(i, j, k, HydroSystem<StarCluster>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StarCluster>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<StarCluster>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<StarCluster>::x3Momentum_index) = rho * vz;
	});
}

template <> void RadhydroSimulation<StarCluster>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	// TODO(ben): refine on Jeans length

	const Real q_min = 5.0; // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<StarCluster>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);
			if (q > q_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<StarCluster>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<StarCluster>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<StarCluster>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<StarCluster>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
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

	// Problem initialization
	RadhydroSimulation<StarCluster> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int status = 0;
	return status;
}
