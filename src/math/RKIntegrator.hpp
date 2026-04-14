#ifndef RKINTEGRATOR_HPP_ // NOLINT
#define RKINTEGRATOR_HPP_
//==============================================================================
// Quokka -- a radiation-hydrodynamics code built on AMReX
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file RKIntegrator.hpp
/// \brief Generic stage-driven Runge-Kutta integrator skeleton for composite
///        cell-centered / face-centered AMReX state.
///
/// This interface is intentionally higher-level than AMReX_RungeKutta.H. It is
/// designed for Quokka's hydro/MHD update, where an accepted RK stage depends
/// on more than a single cell-centered RHS:
///   - cell-centered conserved state
///   - optional face-centered state (e.g., magnetic fields)
///   - stage-local fluxes, face velocities, and EMFs
///   - optional stage validation/fixup before the stage is accepted
///   - per-stage flux-register / EMF-register increments
///
/// The policy object owns all physics-specific operations. RKIntegrator only
/// schedules stages and manages temporary storage.
///

#include <array>
#include <type_traits>
#include <utility>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FluxRegister.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

namespace quokka
{

using Real = amrex::Real;

/// Non-owning view of a composite state with one cell-centered MultiFab and
/// optional face-centered MultiFabs.
struct CompositeStateView
{
	amrex::MultiFab *cc = nullptr;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> *fc_data = nullptr;
	std::array<amrex::MultiFab *, AMREX_SPACEDIM> fc{};

	[[nodiscard]] auto hasCellState() const -> bool { return cc != nullptr; }

	[[nodiscard]] auto hasFaceState() const -> bool
	{
		for (auto *ptr : fc) {
			if (ptr != nullptr) {
				return true;
			}
		}
		return false;
	}
};

/// Owned scratch state used by the integrator for intermediate RK stages.
struct CompositeStateData
{
	amrex::MultiFab cc;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fc{};
	bool has_face_state = false;

	void defineLike(CompositeStateView const &reference, int nghost_cc, int nghost_fc)
	{
		AMREX_ASSERT(reference.cc != nullptr);

		cc.define(reference.cc->boxArray(), reference.cc->DistributionMap(), reference.cc->nComp(), nghost_cc, amrex::MFInfo(),
			  reference.cc->Factory());

		has_face_state = false;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			if (reference.fc[dir] != nullptr) {
				fc[dir].define(reference.fc[dir]->boxArray(), reference.fc[dir]->DistributionMap(), reference.fc[dir]->nComp(), nghost_fc,
					       amrex::MFInfo(), reference.fc[dir]->Factory());
				has_face_state = true;
			}
		}
	}

	auto view() -> CompositeStateView
	{
		CompositeStateView state{};
		state.cc = &cc;
		state.fc_data = &fc;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			state.fc[dir] = has_face_state ? &fc[dir] : nullptr;
		}
		return state;
	}

	void copyFrom(CompositeStateView const &src, int nghost = 0)
	{
		AMREX_ASSERT(src.cc != nullptr);
		amrex::MultiFab::Copy(cc, *src.cc, 0, 0, cc.nComp(), nghost);

		if (has_face_state) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				if (src.fc[dir] != nullptr) {
					amrex::MultiFab::Copy(fc[dir], *src.fc[dir], 0, 0, fc[dir].nComp(), nghost);
				}
			}
		}
	}
};

/// Short-lived temporaries used while constructing and validating one RK stage.
struct StageScratch
{
	amrex::MultiFab rhs_cc;
	amrex::iMultiFab redo_flag;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fluxes_hi{};
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fluxes_lo{};
	std::array<amrex::MultiFab, AMREX_SPACEDIM> face_vel{};
	std::array<amrex::MultiFab, AMREX_SPACEDIM> emf{};

	bool has_redo_flag = false;
	bool has_fluxes_hi = false;
	bool has_fluxes_lo = false;
	bool has_face_vel = false;
	bool has_emf = false;

	void clearFlags()
	{
		has_redo_flag = false;
		has_fluxes_hi = false;
		has_fluxes_lo = false;
		has_face_vel = false;
		has_emf = false;
	}
};

/// Running quantities that genuinely need to persist across RK stages.
struct StepAccumulators
{
	std::array<amrex::MultiFab, AMREX_SPACEDIM> avg_face_vel{};
	bool has_avg_face_vel = false;

	void defineFaceVelocityLike(CompositeStateView const &reference, int nghost)
	{
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			AMREX_ASSERT(reference.cc != nullptr);
			auto ba_fc = amrex::convert(reference.cc->boxArray(), amrex::IntVect::TheDimensionVector(dir));
			avg_face_vel[dir].define(ba_fc, reference.cc->DistributionMap(), 1, nghost);
			avg_face_vel[dir].setVal(0.0);
		}
		has_avg_face_vel = true;
	}

	void reset()
	{
		if (has_avg_face_vel) {
			for (auto &mf : avg_face_vel) {
				mf.setVal(0.0);
			}
		}
	}
};

/// SSPRK2 stage description for Quokka's hydro update.
struct SSPRK2Scheme
{
	static constexpr int nstages = 2;

	/// Stage times t_n + c_s * dt.
	static constexpr std::array<Real, nstages> c = {Real(0.0), Real(1.0)};

	/// Weights for time-integrated face quantities accumulated per stage, e.g.
	/// flux-register increments or average face velocity.
	static constexpr std::array<Real, nstages> stage_integral_weights = {Real(0.5), Real(0.5)};
};

namespace detail
{
template <typename Policy>
concept HasDefine = requires(Policy &policy, int lev, CompositeStateView const &reference) {
	{ policy.define(lev, reference) };
};

template <typename Policy>
concept HasResetAccumulators = requires(Policy &policy, StepAccumulators &accum) {
	{ policy.reset_accumulators(accum) };
};
} // namespace detail

/// Generic stage-driven RK integrator for Quokka composite state.
///
/// Policy interface expected by this class:
///   void define(int lev, CompositeStateView const& reference);
///   int nghost_cc() const;
///   int nghost_fc() const;
///   void define_stage_scratch(StageScratch&, CompositeStateView const&, int lev) const;
///   void define_step_accumulators(StepAccumulators&, CompositeStateView const&, int lev) const;
///   void fill_boundary(int stage, CompositeStateView, Real stage_time) const;
///   void compute_stage(int stage, StageScratch&, CompositeStateView const&, Real stage_time, Real dt_stage) const;
///   void update_stage(int stage, CompositeStateView, CompositeStateView const&, CompositeStateView const&,
///                     StageScratch const&, Real dt) const;
///   auto validate_stage(int stage, CompositeStateView, CompositeStateView const&, CompositeStateView const&,
///                       StageScratch&, Real dt) const -> bool;
///   void post_stage(int stage, CompositeStateView, StageScratch const&, Real stage_time, Real dt) const;
///   void accumulate_stage(int stage, StageScratch const&, Real dt, StepAccumulators&) const;
///   void finalize_step(CompositeStateView, Real time, Real dt, StepAccumulators const&) const;
template <typename Policy, typename Scheme = SSPRK2Scheme> class RKIntegrator
{
      public:
	explicit RKIntegrator(Policy policy) : policy_(std::move(policy)) {}

	void define(int lev, CompositeStateView const &reference)
	{
		lev_ = lev;
		if constexpr (detail::HasDefine<Policy>) {
			policy_.define(lev, reference);
		}

		stage_state_.defineLike(reference, policy_.nghost_cc(), policy_.nghost_fc());
		policy_.define_stage_scratch(scratch_, reference, lev);
		policy_.define_step_accumulators(accumulators_, reference, lev);
		defined_ = true;
	}

	auto advance(CompositeStateView const &old_state, CompositeStateView new_state, Real time, Real dt) -> bool
	{
		AMREX_ASSERT(defined_);
		AMREX_ASSERT(old_state.cc != nullptr);
		AMREX_ASSERT(new_state.cc != nullptr);

		if constexpr (detail::HasResetAccumulators<Policy>) {
			policy_.reset_accumulators(accumulators_);
		} else {
			accumulators_.reset();
		}

		auto stage_state = stage_state_.view();
		CompositeStateView stage_input = old_state;

		for (int stage = 0; stage < Scheme::nstages; ++stage) {
			const Real stage_time = time + Scheme::c[stage] * dt;
			const Real dt_stage = Scheme::stage_integral_weights[stage] * dt;
			CompositeStateView stage_output = (stage == Scheme::nstages - 1) ? new_state : stage_state;

			policy_.fill_boundary(stage + 1, stage_input, stage_time);
			policy_.compute_stage(stage + 1, scratch_, stage_input, stage_time, dt_stage);
			policy_.update_stage(stage + 1, stage_output, old_state, stage_input, scratch_, dt);

			const bool stage_ok = policy_.validate_stage(stage + 1, stage_output, old_state, stage_input, scratch_, dt);
			if (!stage_ok) {
				return false;
			}

			policy_.post_stage(stage + 1, stage_output, scratch_, stage_time, dt);
			policy_.accumulate_stage(stage + 1, scratch_, dt, accumulators_);
			stage_input = stage_output;
		}

		policy_.finalize_step(new_state, time, dt, accumulators_);
		return true;
	}

	[[nodiscard]] auto getAccumulators() const -> StepAccumulators const & { return accumulators_; }

      private:
	Policy policy_;
	int lev_ = -1;
	bool defined_ = false;
	CompositeStateData stage_state_;
	StageScratch scratch_;
	StepAccumulators accumulators_;
};

/// Hydro/MHD-oriented policy interface sketch for use with RKIntegrator.
///
/// This is an interface contract, not a base class with virtual dispatch.
/// A concrete policy can be templated on `problem_t` and hold any Quokka state
/// needed to compute fluxes, fill boundaries, and update reflux registers.
struct HydroRKPolicyInterface
{
	[[nodiscard]] auto nghost_cc() const -> int;
	[[nodiscard]] auto nghost_fc() const -> int;

	void define(int lev, CompositeStateView const &reference);
	void define_stage_scratch(StageScratch &scratch, CompositeStateView const &reference, int lev) const;
	void define_step_accumulators(StepAccumulators &accum, CompositeStateView const &reference, int lev) const;
	void reset_accumulators(StepAccumulators &accum) const;

	void fill_boundary(int stage, CompositeStateView state, Real stage_time) const;

	void compute_stage(int stage, StageScratch &scratch, CompositeStateView const &input, Real stage_time, Real dt_stage) const;

	void update_stage(int stage, CompositeStateView output, CompositeStateView const &old_state, CompositeStateView const &stage_input,
			  StageScratch const &scratch, Real dt) const;

	/// Returns false if the stage failed and the caller should reject the step.
	auto validate_stage(int stage, CompositeStateView output, CompositeStateView const &old_state, CompositeStateView const &stage_input,
			    StageScratch &scratch, Real dt) const -> bool;

	void post_stage(int stage, CompositeStateView output, StageScratch const &scratch, Real stage_time, Real dt) const;

	/// Perform per-stage flux-register / EMF-register increments and update any
	/// step-spanning accumulators such as time-averaged face velocity.
	void accumulate_stage(int stage, StageScratch const &scratch, Real dt, StepAccumulators &accum) const;

	void finalize_step(CompositeStateView new_state, Real time, Real dt, StepAccumulators const &accum) const;
};

} // namespace quokka

#endif // RKINTEGRATOR_HPP_
