//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmoSphereDM.cpp
/// \brief Kinematic cosmology test: a gas sphere and one CIC (DM) particle drift together
/// at the same peculiar velocity in an expanding periodic box. Pass if their final
/// positions agree within tolerance.

#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"

#include <cmath>
#include <format>
#include <string>

struct CosmoSphereDM {
	static constexpr amrex::Real rho_min = 1.0e-27;		  // g/cm^3, comoving
	static constexpr amrex::Real rho_max = 1.0e-24;		  // g/cm^3, comoving
	static constexpr amrex::Real P0 = 1.0e-14;		  // erg/cm^3
	static constexpr amrex::Real drift_vel = 1.0e8;		  // cm/s (1000 km/s) both for gas and DM
	static constexpr amrex::Real R_sphere_default = 3.086e23; // 100 kpc
	static constexpr amrex::Real R_smooth_default = 6.172e22; // R_sphere / 5
};

template <> struct quokka::EOS_Traits<CosmoSphereDM> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<CosmoSphereDM> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct Physics_Traits<CosmoSphereDM> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;

	// Cosmology parameters (EdS)
	static constexpr amrex::Real omega_m = 1.0;
	static constexpr amrex::Real omega_r = 0.0;
	static constexpr amrex::Real omega_lambda = 0.0;
	static constexpr amrex::Real omega_b = 0.17;
	static constexpr amrex::Real omega_dm = 0.83;
	static constexpr amrex::Real hubble_constant = 0.7;
	static constexpr amrex::Real a_init = 0.01; // z = 99
	static constexpr amrex::Real cosmology_dt_limit = 0.01;
};

struct SphereParams {
	amrex::Real R_sphere;
	amrex::Real R_smooth;
};

auto read_sphere_params() -> SphereParams
{
	SphereParams sp{CosmoSphereDM::R_sphere_default, CosmoSphereDM::R_smooth_default};
	amrex::ParmParse const pp("sphere");
	pp.query("R_sphere", sp.R_sphere);
	pp.query("R_smooth", sp.R_smooth);
	return sp;
}

template <> void QuokkaSimulation<CosmoSphereDM>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;		 // cell dimensions (dx, dy, dz)
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_; // low left physical coordinates
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_; // top right physical coordinates

	// Center of the domain
	amrex::Real const X0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const Y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const Z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// Gas physical parameters
	amrex::Real const rho_min = 1.0e-27;
	amrex::Real const rho_max = 1.0e-24;
	amrex::Real const P = 1.0e-14;
	amrex::Real const vx = CosmoSphereDM::drift_vel;
	amrex::Real const vy = 0.0;
	amrex::Real const vz = 0.0;
	amrex::Real R_sphere = read_sphere_params().R_sphere;
	amrex::Real R_smooth = read_sphere_params().R_smooth;

	const amrex::Real cm_to_kpc = 1 / (C::parsec * 1.0e3);
	amrex::Print() << "\n=================== Hydro initialization ===================\n"
		       << "  - Background density           : " << rho_min << " g/cm^3\n"
		       << "  - Max density                  : " << rho_max << " g/cm^3\n"
		       << "  - Pressure (constant)          : " << P << " erg/cm^3\n"
		       << "  - Drift velocity (vx)          : " << vx << " cm/s\n"
		       << "  - Sphere radius                : " << R_sphere * cm_to_kpc << " kpc\n"
		       << "  - Tanh smoothing rad.          : " << R_smooth * cm_to_kpc << " kpc\n\n";

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cell distance from the center
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt((x - X0) * (x - X0) + (y - Y0) * (y - Y0) + (z - Z0) * (z - Z0));

		amrex::Real const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		amrex::Real const eint = quokka::EOS<CosmoSphereDM>::ComputeEintFromPres(rho, P);

		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::internalEnergy_index) = eint;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::energy_index) = eint + 0.5 * rho * (vx * vx + vy * vy + vz * vz); // eint + ekin
	});
}

#if AMREX_SPACEDIM == 3
template <> void QuokkaSimulation<CosmoSphereDM>::createInitialCICParticles()
{

	const int lev = 0;
	const amrex::Geometry &geom_lev = this->geom[lev];

	// GpuArray with the borders of the domain
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom_lev.ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom_lev.ProbHiArray();

	// Center of the domain
	amrex::Real const X0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const Y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const Z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
	amrex::RealVect const center_coords{X0, Y0, Z0};
	amrex::IntVect const center_index = geom_lev.CellIndex(center_coords.dataPtr());

	amrex::Real R_sphere = read_sphere_params().R_sphere;
	amrex::Real const rho_max = 1.0e-24;
	amrex::Real const mass_gas = (4.0 / 3.0) * M_PI * std::pow(R_sphere, 3) * rho_max;
	amrex::Real const mass_dm = 5.0 * mass_gas; // DM fivefolds the gas
	amrex::Real const vx_dm = CosmoSphereDM::drift_vel;
	amrex::Real const vy_dm = 0.0;
	amrex::Real const vz_dm = 0.0;

	// Initialize the particle at domain center
	using ParticleType = quokka::CICParticleContainer::ParticleType; // alias

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		amrex::Box const &valid_box = mfi.validbox();

		if (valid_box.contains(center_index)) {
			ParticleType p;
			p.id() = ParticleType::NextID();
			p.cpu() = amrex::ParallelDescriptor::MyProc();
			p.pos(0) = X0;
			p.pos(1) = Y0;
			p.pos(2) = Z0;
			p.rdata(quokka::CICParticleMassIdx) = mass_dm;
			p.rdata(quokka::CICParticleVxIdx) = vx_dm;
			p.rdata(quokka::CICParticleVyIdx) = vy_dm;
			p.rdata(quokka::CICParticleVzIdx) = vz_dm;

			auto const key = std::make_pair(mfi.index(), mfi.LocalTileIndex());
			CICParticles->GetParticles(lev)[key].push_back(p);
		}
	}
	CICParticles->Redistribute(); // Redistribute() moves particles to the box/rank that contains their position
}
#endif

auto problem_main() -> int
{
	int status = 0;

	QuokkaSimulation<CosmoSphereDM> sim;
	sim.setInitialConditions();
	sim.evolve();

	amrex::Print() << "Computing particle-gas distance for the test...\n";

	// Level 0 covers the whole domain (cosmology enforces max_level = 0),
	// so there is no AMR double counting
	const int lev = 0;
	const amrex::Geometry &geom = sim.geom[lev];
	const auto dx = geom.CellSizeArray();
	const auto prob_lo = geom.ProbLoArray();
	const auto prob_hi = geom.ProbHiArray();
	const amrex::MultiFab &mf = sim.state_new_cc_[lev];
	const amrex::Real Lx = prob_hi[0] - prob_lo[0];
	const amrex::Real Ly = prob_hi[1] - prob_lo[1];
	const amrex::Real Lz = prob_hi[2] - prob_lo[2];

	// Extract Domain information (Physical and Index space)
	const amrex::Box &domain_box = geom.Domain();
	const auto domain_lo = domain_box.smallEnd();
	const auto domain_hi = domain_box.bigEnd();

	// Calculate total cells per dimension on level 0
	const int n_cells_x = domain_hi[0] - domain_lo[0] + 1;
	const int n_cells_y = domain_hi[1] - domain_lo[1] + 1;
	const int n_cells_z = domain_hi[2] - domain_lo[2] + 1;

	// Six running sums: rho*cos/sin(theta) for x, y, z
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(
	    reduce_ops); // memory preallocation for the sums

	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox(); // no ghost cells
		const auto &state_arr = mf.array(mfi);	// cell data array

		reduce_ops.eval(
		    box, reduce_data,
		    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
			    const amrex::Real cell_center_x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0]; // x center of the cell
			    const amrex::Real cell_center_y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
			    const amrex::Real cell_center_z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
			    const amrex::Real rho = state_arr(i, j, k, HydroSystem<CosmoSphereDM>::density_index);

			    /// Periodic center of mass from the phase of the first Fourier mode (arithmetic mean breaks across the boundary)
			    const amrex::Real theta_x =
				2.0 * M_PI * (cell_center_x - prob_lo[0]) / Lx; // map coordinates to [0, 2pi] relative to the left edge of the domain
			    const amrex::Real theta_y = 2.0 * M_PI * (cell_center_y - prob_lo[1]) / Ly;
			    const amrex::Real theta_z = 2.0 * M_PI * (cell_center_z - prob_lo[2]) / Lz;

			    // Values to be summed: first Fourier mode components (rho*cos(theta_x), rho*sin(theta_x), ...)
			    return {rho * std::cos(theta_x), rho * std::sin(theta_x), rho * std::cos(theta_y),
				    rho * std::sin(theta_y), rho * std::cos(theta_z), rho * std::sin(theta_z)};
		    }); // eval() adds each cell's tuple to the running sums (block-wise on GPU)
	}

	// Structure binding with per-rank sums (value() does not communicate)
	auto [cos_x, sin_x, cos_y, sin_y, cos_z, sin_z] = reduce_data.value();

	amrex::ParallelDescriptor::ReduceRealSum(cos_x); // MPI_Allreduce(sum)
	amrex::ParallelDescriptor::ReduceRealSum(sin_x);
	amrex::ParallelDescriptor::ReduceRealSum(cos_y);
	amrex::ParallelDescriptor::ReduceRealSum(sin_y);
	amrex::ParallelDescriptor::ReduceRealSum(cos_z);
	amrex::ParallelDescriptor::ReduceRealSum(sin_z);

	// Phase of the mode in (-pi, pi]; atan2 resolves the quadrant
	const amrex::Real theta_cmx = std::atan2(sin_x, cos_x); // arctg2(y,x) account for the relative x and y signs
	const amrex::Real theta_cmy = std::atan2(sin_y, cos_y);
	const amrex::Real theta_cmz = std::atan2(sin_z, cos_z);

	// Phase -> position in [prob_lo, prob_hi); +L makes the fmod argument positive
	const amrex::Real gas_center_x = prob_lo[0] + std::fmod(theta_cmx / (2.0 * M_PI) * Lx + Lx, Lx);
	const amrex::Real gas_center_y = prob_lo[1] + std::fmod(theta_cmy / (2.0 * M_PI) * Ly + Ly, Ly);
	const amrex::Real gas_center_z = prob_lo[2] + std::fmod(theta_cmz / (2.0 * M_PI) * Lz + Lz, Lz);

	amrex::Print() << "\n--- Fourier Diagnostics ---\n"
		       << "  cos_x = " << cos_x << "  sin_x = " << sin_x << "\n"
		       << "  cos_y = " << cos_y << "  sin_y = " << sin_y << "\n"
		       << "  cos_z = " << cos_z << "  sin_z = " << sin_z << "\n"
		       << "  theta_cmx = " << theta_cmx << " (" << theta_cmx * 180.0 / M_PI << " deg)\n"
		       << "  theta_cmy = " << theta_cmy << " (" << theta_cmy * 180.0 / M_PI << " deg)\n"
		       << "  theta_cmz = " << theta_cmz << " (" << theta_cmz * 180.0 / M_PI << " deg)\n";

	// Impossible sentinel values
	amrex::Real px = -1e30, py = -1e30, pz = -1e30;

	// Get particle data using the particle descriptor
	for (int lev_p = 0; lev_p <= sim.finestLevel(); ++lev_p) {
		const auto [real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)
						       ->getParticleDataAtLevel(lev_p); // get particle in the particle descriptor
		if (real_data.size() > 0) {						// If the particle is found locally on this rank
			const auto &p = real_data[0];
			px = p[0];
			py = p[1];
			pz = p[2];
			break; // exit when the particle is found
		}
	}

	// Max-reduction: ranks without the particle contribute the sentinel, so all ranks get its position
	amrex::ParallelDescriptor::ReduceRealMax(px);
	amrex::ParallelDescriptor::ReduceRealMax(py);
	amrex::ParallelDescriptor::ReduceRealMax(pz);

	const amrex::Real cm_to_Mpc = 1 / (C::parsec * 1.0e6);
	const amrex::Real cm_to_kpc = 1 / (C::parsec * 1.0e3);

	amrex::Print() << "\n=================== DOMAIN & RESOLUTION ===================\n"
		       << "  - Cell Resolution (dx, dy, dz) : (" << dx[0] * cm_to_kpc << ", " << dx[1] * cm_to_kpc << ", " << dx[2] * cm_to_kpc << ") kpc\n"
		       << "  - Domain Cells (Nx, Ny, Nz)    : (" << n_cells_x << ", " << n_cells_y << ", " << n_cells_z << ") cells\n"
		       << "  - Domain Size X [Lo / Hi]      : [" << prob_lo[0] * cm_to_Mpc << " / " << prob_hi[0] * cm_to_Mpc << "] Mpc\n"
		       << "  - Domain Size Y [Lo / Hi]      : [" << prob_lo[1] * cm_to_Mpc << " / " << prob_hi[1] * cm_to_Mpc << "] Mpc\n"
		       << "  - Domain Size Z [Lo / Hi]      : [" << prob_lo[2] * cm_to_Mpc << " / " << prob_hi[2] * cm_to_Mpc << "] Mpc\n"
		       << "-----------------------------------------------------------\n";

	if (px > -1e29) { // there is the particle
		// Minimum-image shift: round(d/L) is the nearest integer number of box lengths,
		// subtracting it folds d to [-L/2, L/2]
		const auto wrap = [](amrex::Real d, amrex::Real L) { return d - L * std::round(d / L); };
		const amrex::Real shift_x = wrap(px - gas_center_x, Lx); // wrap the shift to [-Lx/2, Lx/2]
		const amrex::Real shift_y = wrap(py - gas_center_y, Ly);
		const amrex::Real shift_z = wrap(pz - gas_center_z, Lz);
		const amrex::Real shift_mpc = std::sqrt(shift_x * shift_x + shift_y * shift_y + shift_z * shift_z) * cm_to_Mpc;
		const amrex::Real shift_cell = shift_mpc / (dx[0] * cm_to_Mpc);
		const amrex::Real tolerance_cell = 2;
		const amrex::Real tolerance_mpc = tolerance_cell * dx[0] * cm_to_Mpc;

		if (shift_cell > tolerance_cell) {
			amrex::Print() << "\n========================================================\n"
				       << "[TEST FAILED]: CosmoSphereDM misalignment detected!\n"
				       << "  - Particle position    : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
				       << "  - Gas center of mass   : (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", "
				       << gas_center_z * cm_to_Mpc << ") Mpc\n"
				       << "  - Calculated Distance  : " << shift_mpc * 1e3 << " kpc (" << shift_cell << " cells widths)\n"
				       << "  - Allowed Tolerance    : " << tolerance_mpc * 1e3 << " kpc (" << tolerance_cell << " cells widths)\n"
				       << "========================================================\n\n";
			status = 1;
		} else {
			amrex::Print() << "\n========================================================\n"
				       << "[TEST PASSED]: CosmoSphereDM alignment check successful!\n"
				       << "  - Particle position : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
				       << "  - Gas center of mass: (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", "
				       << gas_center_z * cm_to_Mpc << ") Mpc\n"
				       << "  - Calculated Distance  : " << shift_mpc * 1e3 << " kpc (" << shift_cell << " cells widths)\n"
				       << "  - Allowed Tolerance    : " << tolerance_mpc * 1e3 << " kpc (" << tolerance_cell << " cells widths)\n"
				       << "========================================================\n\n";
			status = 0;
		}
	} else {
		amrex::Print() << "[TEST FAILED]: Particle not found.";
		status = 1;
	}
	amrex::Print() << "  Gas sphere and DM drift velocity = " << CosmoSphereDM::drift_vel / 1.0e5 << " km/s";

	// MPI Broadcast, update status from IOprocessor to the others
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return status;
}