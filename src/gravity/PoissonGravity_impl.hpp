#ifndef POISSONGRAVITY_IMPL_HPP_
#define POISSONGRAVITY_IMPL_HPP_
//==============================================================================
// Quokka - a radiation hydrodynamics code for AMR
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file PoissonGravity_impl.hpp
/// \brief Template implementation of the PoissonGravity class

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

template <typename problem_t>
PoissonGravity<problem_t>::PoissonGravity(const amrex::Vector<amrex::Geometry> &geom, const amrex::Vector<amrex::BoxArray> &grids,
					  const amrex::Vector<amrex::DistributionMapping> &dmap, int max_level)
    : geom_(geom), grids_(grids), dmap_(dmap), max_level_(max_level)
{
	BL_PROFILE("PoissonGravity::PoissonGravity()");

	// Read runtime parameters
	read_parameters();

	// Resize vectors for each level
	const int nlevels = max_level_ + 1;
	mlpoisson_.resize(nlevels);
	mlmg_.resize(nlevels);

	// Setup solvers for each level
	for (int lev = 0; lev <= max_level_; ++lev) {
		setup_poisson_solver(lev);
	}
}

template <typename problem_t> void PoissonGravity<problem_t>::read_parameters()
{
	BL_PROFILE("PoissonGravity::read_parameters()");

	amrex::ParmParse pp("gravity");
	pp.query("gravitational_constant", gravitational_constant_);
	pp.query("tolerance", tolerance_);
	pp.query("max_iterations", max_iterations_);
	pp.query("verbose", verbose_);

	if (verbose_ > 0) {
		amrex::Print() << "PoissonGravity parameters:\n";
		amrex::Print() << "  gravitational_constant = " << gravitational_constant_ << "\n";
		amrex::Print() << "  tolerance = " << tolerance_ << "\n";
		amrex::Print() << "  max_iterations = " << max_iterations_ << "\n";
	}
}

template <typename problem_t> void PoissonGravity<problem_t>::setup_poisson_solver(int level)
{
	BL_PROFILE("PoissonGravity::setup_poisson_solver()");

	// Create single-level BoxArray and DistributionMapping for this level
	amrex::Vector<amrex::BoxArray> level_grids(1);
	amrex::Vector<amrex::DistributionMapping> level_dmap(1);
	amrex::Vector<amrex::Geometry> level_geom(1);

	level_grids[0] = grids_[level];
	level_dmap[0] = dmap_[level];
	level_geom[0] = geom_[level];

	// Create MLPoisson solver for this specific level only (Approach 1)
	mlpoisson_[level] = std::make_unique<amrex::MLPoisson>(level_geom, level_grids, level_dmap);

	// Set boundary conditions
	setup_boundary_conditions(level);

	// Create multigrid solver
	mlmg_[level] = std::make_unique<amrex::MLMG>(*mlpoisson_[level]);
	mlmg_[level]->setMaxIter(max_iterations_);
	mlmg_[level]->setMaxFmgIter(0); // No FMG cycles for simplicity
	mlmg_[level]->setVerbose(verbose_);
	mlmg_[level]->setAbsTol(tolerance_);
	mlmg_[level]->setRelTol(tolerance_);
}

template <typename problem_t> void PoissonGravity<problem_t>::setup_boundary_conditions(int level)
{
	BL_PROFILE("PoissonGravity::setup_boundary_conditions()");

	// Set boundary conditions for the Poisson equation
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_lo, bc_hi;

	if (level == 0) {
		// Level 0: Use free-space/vacuum boundary conditions (Dirichlet with phi=0 at infinity)
		// For finite domains, we approximate this with Dirichlet BC at domain boundaries
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (geom_[level].isPeriodic(idim)) {
				bc_lo[idim] = bc_hi[idim] = amrex::LinOpBCType::Periodic;
			} else {
				// Use Dirichlet boundary conditions with phi=0 at the boundary
				// This approximates the free-space boundary condition
				bc_lo[idim] = bc_hi[idim] = amrex::LinOpBCType::Dirichlet;
			}
		}
	} else {
		// Level > 0: Use Dirichlet boundary conditions interpolated from coarser level
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (geom_[level].isPeriodic(idim)) {
				bc_lo[idim] = bc_hi[idim] = amrex::LinOpBCType::Periodic;
			} else {
				bc_lo[idim] = bc_hi[idim] = amrex::LinOpBCType::Dirichlet;
			}
		}
	}

	mlpoisson_[level]->setDomainBC(bc_lo, bc_hi);
}

template <typename problem_t>
void PoissonGravity<problem_t>::solve_for_phi(int level, const amrex::MultiFab &density, amrex::MultiFab &gravitational_potential, amrex::Real /*time*/)
{
	BL_PROFILE("PoissonGravity::solve_for_phi()");

	// Compute RHS = 4*pi*G*rho
	auto rhs = compute_rhs_from_density(level, density);

	// For level > 0, we need boundary conditions from the coarser level
	if (level > 0) {
		// This would normally require the coarse-level solution
		// For now, we assume the boundary conditions are set to zero
		// In a full implementation, this would be interpolated from level-1
		gravitational_potential.setVal(0.0);
	} else {
		// Level 0: initialize to zero (free-space boundary condition)
		gravitational_potential.setVal(0.0);
	}

	// Solve the Poisson equation: ∇²φ = 4πGρ
	const amrex::Real abs_tol = 0.0;
	const amrex::Real rel_tol = tolerance_;
	mlmg_[level]->solve({&gravitational_potential}, {&rhs}, rel_tol, abs_tol);

	if (verbose_ > 1) {
		amrex::Print() << "PoissonGravity: Level " << level << " solve completed in " << mlmg_[level]->getNumIters() << " iterations\n";
	}
}

template <typename problem_t> auto PoissonGravity<problem_t>::compute_rhs_from_density(int level, const amrex::MultiFab &density) -> amrex::MultiFab
{
	BL_PROFILE("PoissonGravity::compute_rhs_from_density()");

	amrex::MultiFab rhs(grids_[level], dmap_[level], 1, 0);

	// RHS = 4*pi*G*rho
	const amrex::Real four_pi_G = 4.0 * M_PI * gravitational_constant_;

	for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &rhs_fab = rhs.array(mfi);
		auto const &rho_fab = density.const_array(mfi);

		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) { rhs_fab(i, j, k) = four_pi_G * rho_fab(i, j, k); });
	}

	return rhs;
}

template <typename problem_t>
void PoissonGravity<problem_t>::compute_gravitational_acceleration(int level, const amrex::MultiFab &gravitational_potential,
								   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &gravitational_acceleration)
{
	BL_PROFILE("PoissonGravity::compute_gravitational_acceleration()");

	// Compute gravitational acceleration: g = -∇φ
	const auto &dx = geom_[level].CellSizeArray();

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		for (amrex::MFIter mfi(gravitational_acceleration[idim]); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const &grav_fab = gravitational_acceleration[idim].array(mfi);
			auto const &phi_fab = gravitational_potential.const_array(mfi);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				// Central difference approximation: g_i = -(φ(i+1) - φ(i-1)) / (2*dx_i)
				if (idim == 0) {
					grav_fab(i, j, k) = -(phi_fab(i + 1, j, k) - phi_fab(i - 1, j, k)) / (2.0 * dx[0]);
				} else if (idim == 1) {
					grav_fab(i, j, k) = -(phi_fab(i, j + 1, k) - phi_fab(i, j - 1, k)) / (2.0 * dx[1]);
				}
#if AMREX_SPACEDIM == 3
				else if (idim == 2) {
					grav_fab(i, j, k) = -(phi_fab(i, j, k + 1) - phi_fab(i, j, k - 1)) / (2.0 * dx[2]);
				}
#endif
			});
		}
	}
}

template <typename problem_t>
void PoissonGravity<problem_t>::apply_operator_split_gravity_update(int level, amrex::MultiFab &state,
								    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &gravitational_acceleration,
								    amrex::Real dt)
{
	BL_PROFILE("PoissonGravity::apply_operator_split_gravity_update()");

	// Apply gravitational acceleration to momentum and energy
	// This assumes standard hydro variable ordering: [rho, rho*u, rho*v, rho*w, E, ...]
	const int irho = 0;
	const int imx = 1;
	const int imy = 2;
#if AMREX_SPACEDIM == 3
	const int imz = 3;
	const int ieng = 4;
#else
	const int ieng = 3;
#endif

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &state_fab = state.array(mfi);
		auto const &gx_fab = gravitational_acceleration[0].const_array(mfi);
		auto const &gy_fab = gravitational_acceleration[1].const_array(mfi);
#if AMREX_SPACEDIM == 3
		auto const &gz_fab = gravitational_acceleration[2].const_array(mfi);
#endif

		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const amrex::Real rho = state_fab(i, j, k, irho);
			const amrex::Real old_mx = state_fab(i, j, k, imx);
			const amrex::Real old_my = state_fab(i, j, k, imy);
#if AMREX_SPACEDIM == 3
			const amrex::Real old_mz = state_fab(i, j, k, imz);
#endif

			// Update momentum: Δ(ρu) = ρ * g * Δt
			const amrex::Real dmx = rho * gx_fab(i, j, k) * dt;
			const amrex::Real dmy = rho * gy_fab(i, j, k) * dt;
#if AMREX_SPACEDIM == 3
			const amrex::Real dmz = rho * gz_fab(i, j, k) * dt;
#endif

			state_fab(i, j, k, imx) += dmx;
			state_fab(i, j, k, imy) += dmy;
#if AMREX_SPACEDIM == 3
			state_fab(i, j, k, imz) += dmz;
#endif

			// Update energy: ΔE = (old_momentum + 0.5*Δmomentum) · g * Δt
			amrex::Real energy_update = (old_mx + 0.5 * dmx) * gx_fab(i, j, k) * dt + (old_my + 0.5 * dmy) * gy_fab(i, j, k) * dt;
#if AMREX_SPACEDIM == 3
			energy_update += (old_mz + 0.5 * dmz) * gz_fab(i, j, k) * dt;
#endif

			state_fab(i, j, k, ieng) += energy_update;
		});
	}
}

template <typename problem_t>
void PoissonGravity<problem_t>::set_dirichlet_boundary_conditions(int level, amrex::MultiFab &gravitational_potential, const amrex::MultiFab &coarse_potential,
								  amrex::Real /*time*/)
{
	BL_PROFILE("PoissonGravity::set_dirichlet_boundary_conditions()");

	if (level == 0) {
		// Level 0: Set boundary to zero (free-space approximation)
		// In a true free-space setup, we would compute the boundary values
		// that give the correct boundary condition at infinity
		gravitational_potential.FillBoundary(geom_[level].periodicity());
	} else {
		// Level > 0: Interpolate from coarser level
		interpolate_boundary_conditions_from_coarse_level(level, gravitational_potential, coarse_potential);
	}
}

template <typename problem_t>
void PoissonGravity<problem_t>::interpolate_boundary_conditions_from_coarse_level(int level, amrex::MultiFab &fine_potential,
										  const amrex::MultiFab &coarse_potential)
{
	BL_PROFILE("PoissonGravity::interpolate_boundary_conditions_from_coarse_level()");

	// This is a simplified placeholder implementation
	// In a full implementation, this would use AMReX's FillPatch functionality
	// to properly interpolate boundary conditions in space and time from the coarser level

	// For now, we fill with a simple boundary fill
	fine_potential.FillBoundary(geom_[level].periodicity());

	// TODO: Implement proper interpolation from coarse_potential to fine_potential boundaries
	// This should use AMReX's interpolation routines to:
	// 1. Identify fine grid cells that are adjacent to coarse-fine boundaries
	// 2. Interpolate coarse_potential values to those fine grid boundary cells
	// 3. Handle time interpolation if coarse_potential is from a different time
}

template <typename problem_t> void PoissonGravity<problem_t>::set_gravitational_constant(amrex::Real G_const) { gravitational_constant_ = G_const; }

template <typename problem_t> auto PoissonGravity<problem_t>::get_gravitational_constant() const -> amrex::Real { return gravitational_constant_; }

#endif // POISSONGRAVITY_IMPL_HPP_