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

#include "EOS.hpp"
#include "RadhydroSimulation.hpp"
#include "generate_modes.hpp"
#include "hydro_system.hpp"
#include "star_cluster.hpp"

using amrex::Real;

struct StarCluster {
};

template <> struct quokka::EOS_Traits<StarCluster> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless
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
	// perturbation parameters
	std::unique_ptr<amrex::TableData<Real, 4>> dvx_modes;
	std::unique_ptr<amrex::TableData<Real, 4>> dvy_modes;
	std::unique_ptr<amrex::TableData<Real, 4>> dvz_modes;
	Real dv_rms_generated{};
	Real dv_rms_target{};
	Real rescale_factor{};

	// cloud parameters
	Real R_sphere{};
	Real rho_sphere{};
	Real alpha_vir{};
	int alpha_PL{};
};

// range of modes to perturb
int kmin{2};
int kmax{64};

template <> void RadhydroSimulation<StarCluster>::preCalculateInitialConditions()
{
	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		amrex::Print() << "Generating velocity perturbations...\n";
		int alpha_PL = userData_.alpha_PL;
		amrex::TableData<Real, 4> dvx = generateRandomModes(kmin, kmax, alpha_PL, 42);
		amrex::TableData<Real, 4> dvy = generateRandomModes(kmin, kmax, alpha_PL, 107);
		amrex::TableData<Real, 4> dvz = generateRandomModes(kmin, kmax, alpha_PL, 56);

		// keep solonoidal modes only
		projectModes(kmin, kmax, dvx, dvy, dvz);

		// compute normalisation
		userData_.dv_rms_generated = computeRms(kmin, kmax, dvx, dvy, dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms_generated << "\n";

		// calculate rms velocity for a marginally-bound star cluster
		const Real R_sphere = userData_.R_sphere;
		const Real rho_sph = userData_.rho_sphere;
		const Real alpha_vir = userData_.alpha_vir;
		const Real M_sphere = (4. / 3.) * M_PI * std::pow(R_sphere, 3) * rho_sph;
		const Real rms_dv_target = std::sqrt(alpha_vir * (3. / 5.) * Gconst_ * M_sphere / R_sphere);
		const Real rms_Mach_target = rms_dv_target / quokka::EOS_Traits<StarCluster>::cs_isothermal;
		amrex::Print() << "rms Mach target = " << rms_Mach_target << "\n";

		double rms_dv_actual = userData_.dv_rms_generated;
		userData_.rescale_factor = rms_dv_target / rms_dv_actual;

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

	const int kmin = ::kmin;
	const int kmax = ::kmax;

	amrex::Real Lx = prob_hi[0] - prob_lo[0];
	amrex::Real Ly = prob_hi[1] - prob_lo[1];
	amrex::Real Lz = prob_hi[2] - prob_lo[2];

	amrex::Real x0 = prob_lo[0] + 0.5 * Lx;
	amrex::Real y0 = prob_lo[1] + 0.5 * Ly;
	amrex::Real z0 = prob_lo[2] + 0.5 * Lz;

	// cloud parameters
	const double rho_min = 0.01 * userData_.rho_sphere;
	const double rho_max = userData_.rho_sphere;
	const double R_sphere = userData_.R_sphere;
	const double R_smooth = 0.05 * R_sphere;
	const double renorm_amp = userData_.rescale_factor;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

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

		vx *= renorm_amp * (rho / rho_max);
		vy *= renorm_amp * (rho / rho_max);
		vz *= renorm_amp * (rho / rho_max);

		state_cc(i, j, k, HydroSystem<StarCluster>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StarCluster>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<StarCluster>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<StarCluster>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<StarCluster>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<StarCluster>::internalEnergy_index) = 0;
	});
}

template <> void RadhydroSimulation<StarCluster>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 4; // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real cs = quokka::EOS_Traits<StarCluster>::cs_isothermal;
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<StarCluster>::density_index);
		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

template <> void RadhydroSimulation<StarCluster>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "log_density") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_new_cc_[lev].const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<StarCluster>::density_index);
			output[bx](i, j, k, ncomp) = std::log10(rho);
		});
	}
}

auto problem_main() -> int
{
	// read problem parameters
	amrex::ParmParse pp("perturb");

	// minimum wavenumber
	pp.query("kmin", ::kmin);

	// maximum wavenumber
	pp.query("kmax", ::kmax);

	// negative power-law exponent for power spectrum
	int alpha_PL{};
	pp.query("power_law", alpha_PL);

	// cloud radius
	Real R_sphere{};
	pp.query("cloud_radius", R_sphere);

	// cloud density
	Real rho_sphere{};
	pp.query("cloud_density", rho_sphere);

	// cloud virial parameter
	Real alpha_vir{};
	pp.query("virial_parameter", alpha_vir);

	// boundary conditions
	const int ncomp_cc = Physics_Indices<StarCluster>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<StarCluster> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.Gconst_ = 1.0;	 // units where G = 1
	sim.densityFloor_ = 0.01;

	sim.userData_.R_sphere = R_sphere;
	sim.userData_.rho_sphere = rho_sphere;
	sim.userData_.alpha_vir = alpha_vir;
	sim.userData_.alpha_PL = alpha_PL;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int status = 0;
	return status;
}
