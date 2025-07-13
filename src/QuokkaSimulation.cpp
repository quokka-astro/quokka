#include "QuokkaSimulation.hpp"

template <typename problem_t> void QuokkaSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev)
{
#ifdef QUOKKA_USE_GRAVITY
	if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
		// Fill RHS with density for Poisson equation: RHS = 4*pi*G*rho
		const amrex::Real four_pi_G = 4.0 * M_PI * Gconst_;
		const auto &state = state_new_cc_[lev];
		const int irho = 0; // density component index

		for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const &rhs_fab = rhs.array(mfi);
			auto const &state_fab = state.const_array(mfi);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) { rhs_fab(i, j, k) = four_pi_G * state_fab(i, j, k, irho); });
		}
	} else {
		// Self-gravity disabled, set RHS to zero
		rhs.setVal(0.0);
	}
#else
	// Gravity support not compiled, set RHS to zero
	rhs.setVal(0.0);
#endif
}

template <typename problem_t> void QuokkaSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt)
{
#ifdef QUOKKA_USE_GRAVITY
	if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
		// Initialize PoissonGravity if not already done
		if (poissonGravity_ == nullptr) {
			poissonGravity_ = std::make_unique<PoissonGravity<problem_t>>(geom, grids, dmap, max_level);
		}

		// Compute gravitational acceleration from potential
		amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> gravitational_acceleration;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			gravitational_acceleration[idim].define(grids[lev], dmap[lev], 1, 1);
		}

		poissonGravity_->compute_gravitational_acceleration(lev, phi, gravitational_acceleration);

		// Apply gravitational forces to the state
		auto &state = state_new_cc_[lev];
		poissonGravity_->apply_operator_split_gravity_update(lev, state, gravitational_acceleration, dt);
	}
#endif
}
