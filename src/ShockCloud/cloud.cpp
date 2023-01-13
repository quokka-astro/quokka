//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///
#include <random>
#include <vector>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "RadhydroSimulation.hpp"
#include "cloud.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

using amrex::Real;

struct ShockCloud {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = quokka::hydrogen_mass_cgs;
constexpr double seconds_in_year = 3.154e7;

template <> struct quokka::EOS_Traits<ShockCloud> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
};

template <> struct Physics_Traits<ShockCloud> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

template <> struct SimulationData<ShockCloud> {
	std::unique_ptr<amrex::TableData<Real, 3>> table_data;
};

constexpr Real Tgas0 = 1.0e7;		 // K
constexpr Real nH0 = 1.0e-4;		 // cm^-3
constexpr Real nH1 = 1.0e-1;		 // cm^-3
constexpr Real R_cloud = 5.0 * 3.086e18; // cm [5 pc]
constexpr Real M0 = 2.0;		 // Mach number of shock

constexpr Real P0 = nH0 * Tgas0 * quokka::boltzmann_constant_cgs; // erg cm^-3
constexpr Real rho0 = nH0 * m_H;				  // g cm^-3
constexpr Real rho1 = nH1 * m_H;

// perturbation parameters
const int kmin = 0;
const int kmax = 16;
Real const A = 0.05 / kmax;

// cloud-tracking variables needed for Dirichlet boundary condition
AMREX_GPU_MANAGED static Real rho_wind = 0;
AMREX_GPU_MANAGED static Real v_wind = 0;
AMREX_GPU_MANAGED static Real P_wind = 0;
AMREX_GPU_MANAGED static Real delta_vx = 0;

template <> void RadhydroSimulation<ShockCloud>::preCalculateInitialConditions()
{
	// generate random phases
	amrex::Array<int, 3> tlo{kmin, kmin, kmin};
	amrex::Array<int, 3> thi{kmax, kmax, kmax};
	userData_.table_data = std::make_unique<amrex::TableData<Real, 3>>(tlo, thi);

	amrex::TableData<Real, 3> h_table_data(tlo, thi, amrex::The_Pinned_Arena());
	auto const &h_table = h_table_data.table();

	// Initialize data on the hostcd
	// 64-bit Mersenne Twister (do not use 32-bit version for sampling doubles!)
	std::mt19937_64 rng(1); // NOLINT
	std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);
	for (int j = tlo[0]; j <= thi[0]; ++j) {
		for (int i = tlo[1]; i <= thi[1]; ++i) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				h_table(i, j, k) = sample_phase(rng);
			}
		}
	}

	// Copy data to GPU memory
	userData_.table_data->copy(h_table_data);
	amrex::Gpu::streamSynchronize();
}

template <> void RadhydroSimulation<ShockCloud>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	auto const &phase_table = userData_.table_data->const_table();

	Real const Lx = (prob_hi[0] - prob_lo[0]);
	Real const Ly = (prob_hi[1] - prob_lo[1]);
	Real const Lz = (prob_hi[2] - prob_lo[2]);

	Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	Real const y0 = prob_lo[1] + 0.8 * (prob_hi[1] - prob_lo[1]);
	Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const x = prob_lo[0] + (i + static_cast<Real>(0.5)) * dx[0];
		Real const y = prob_lo[1] + (j + static_cast<Real>(0.5)) * dx[1];
		Real const z = prob_lo[2] + (k + static_cast<Real>(0.5)) * dx[2];
		Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		// compute perturbations
		Real delta_rho = 0;
		for (int ki = kmin; ki < kmax; ++ki) {
			for (int kj = kmin; kj < kmax; ++kj) {
				for (int kk = kmin; kk < kmax; ++kk) {
					if ((ki == 0) && (kj == 0) && (kk == 0)) {
						continue;
					}
					Real const kx = 2.0 * M_PI * static_cast<Real>(ki) / Lx;
					Real const ky = 2.0 * M_PI * static_cast<Real>(kj) / Lx;
					Real const kz = 2.0 * M_PI * static_cast<Real>(kk) / Lx;
					delta_rho += A * std::sin(x * kx + y * ky + z * kz + phase_table(ki, kj, kk));
				}
			}
		}
		AMREX_ALWAYS_ASSERT(delta_rho > -1.0);

		Real rho = rho0 * (1.0 + delta_rho); // background density
		if (R < R_cloud) {
			rho = rho1 * (1.0 + delta_rho); // cloud density
		}
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = (quokka::EOS_Traits<ShockCloud>::gamma - 1.) * P0;
		Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state_cc(i, j, k, HydroSystem<ShockCloud>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<ShockCloud>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<ShockCloud>::internalEnergy_index) = Eint;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<ShockCloud>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real /*time*/, const amrex::BCRec * /*bcr*/,
												int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();

	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int jhi = domain_hi[1];

	if (j >= jhi) {
		// x2 upper boundary -- constant
		// compute downstream shock conditions from rho0, P0, and M0
		constexpr Real gamma = quokka::EOS_Traits<ShockCloud>::gamma;
		constexpr Real rho2 = rho0 * (gamma + 1.) * M0 * M0 / ((gamma - 1.) * M0 * M0 + 2.);
		constexpr Real P2 = P0 * (2. * gamma * M0 * M0 - (gamma - 1.)) / (gamma + 1.);
		Real const v2 = -M0 * std::sqrt(gamma * P2 / rho2);

		Real const rho = rho2;
		Real const xmom = 0;
		Real const ymom = rho2 * v2;
		Real const zmom = 0;
		Real const Eint = (gamma - 1.) * P2;
		Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		consVar(i, j, k, HydroSystem<ShockCloud>::density_index) = rho;
		consVar(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index) = xmom;
		consVar(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index) = ymom;
		consVar(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index) = zmom;
		consVar(i, j, k, HydroSystem<ShockCloud>::energy_index) = Egas;
		consVar(i, j, k, HydroSystem<ShockCloud>::internalEnergy_index) = Eint;
	}
}

template <> void RadhydroSimulation<ShockCloud>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto tables = cloudyTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<ShockCloud>::energy_index);
				Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = quokka::cooling::ComputeTgasFromEgas(rho, Eint, quokka::EOS_Traits<ShockCloud>::gamma, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

template <> void RadhydroSimulation<ShockCloud>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real eta_threshold = 0.1;		   // gradient refinement threshold
	const Real q_min = std::sqrt(rho0 * rho1); // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<ShockCloud>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);

			Real const q_xplus = state(i + 1, j, k, nidx);
			Real const q_xminus = state(i - 1, j, k, nidx);
			Real const q_yplus = state(i, j + 1, k, nidx);
			Real const q_yminus = state(i, j - 1, k, nidx);
			Real const q_zplus = state(i, j, k + 1, nidx);
			Real const q_zminus = state(i, j, k - 1, nidx);

			Real const del_x = std::max(std::abs(q_xplus - q), std::abs(q - q_xminus));
			Real const del_y = std::max(std::abs(q_yplus - q), std::abs(q - q_yminus));
			Real const del_z = std::max(std::abs(q_zplus - q), std::abs(q - q_zminus));

			Real const gradient_indicator = std::max({del_x, del_y, del_z}) / q;

			if ((gradient_indicator > eta_threshold) && (q > q_min)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.25;
	const double max_time = 2.0e6 * seconds_in_year; // 2 Myr
	const int max_timesteps = 1e5;

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<ShockCloud>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::int_dir); // periodic
		BCs_cc[n].setHi(0, amrex::BCType::int_dir);

		BCs_cc[n].setLo(1, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(1, amrex::BCType::ext_dir);  // Dirichlet

		BCs_cc[n].setLo(2, amrex::BCType::int_dir);
		BCs_cc[n].setHi(2, amrex::BCType::int_dir);
	}

	RadhydroSimulation<ShockCloud> sim(BCs_cc);

	// Standard PPM gives unphysically enormous temperatures when used for
	// this problem (e.g., ~1e14 K or higher), but can be fixed by
	// reconstructing the temperature instead of the pressure
	sim.reconstructionOrder_ = 3;	   // PLM
	sim.densityFloor_ = 1.0e-2 * rho0; // density floor (to prevent vacuum)

	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = 100;
	sim.checkpointInterval_ = 2000;

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
