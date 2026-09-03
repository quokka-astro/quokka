//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file DustSources.hpp
/// \brief Defines source-term integrators for dust-gas drag and dust Lorentz forces.
///

#include "AMReX_MultiFab.H"
#include "AMReX_SmallMatrix.H"
#include "dust/DustRuntimeParams.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/ArrayView.hpp"
#include <cmath>
#include <limits>
#include <numbers>

template <typename problem_t> class DustSources
{
      public:
	static constexpr int nscalars_ = Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int nMassScalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int nHydroScalars_ = Physics_NumVars::numHydroVars + nscalars_;
	static constexpr int numDustVars_ = Physics_NumVars::numDustVarsPerGroup; // number of dust variables for each dust group
	static constexpr int nDustGroups_ = Physics_Traits<problem_t>::nDustGroups;

	enum consVarIndex { // NOLINT
		density_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1Momentum_index,
		x2Momentum_index,
		x3Momentum_index,
		energy_index,
		internalEnergy_index, // auxiliary internal energy (rho * e)
		scalar0_index	      // first passive scalar (only present if nscalars > 0!)
	};

	enum primVarIndex { // NOLINT
		primDensity_index = 0,
		x1Velocity_index,
		x2Velocity_index,
		x3Velocity_index,
		pressure_index,
		primEint_index,	   // auxiliary internal energy (rho * e)
		primScalar0_index, // first passive scalar (only present if nscalars > 0!)
	};

	enum dustVarIndex { // NOLINT
		dustDensity_index = Physics_Indices<problem_t>::dustFirstIndex,
		x1DustMomentum_index,
		x2DustMomentum_index,
		x3DustMomentum_index
	};

	static constexpr int primDustFirstIndex = primScalar0_index + nscalars_;
	enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index }; // NOLINT
	using Vec3 = amrex::SmallVector<amrex::Real, 3>;

	struct ReducedOperator {
		amrex::Real coeffIdentity;
		amrex::Real coeffCross;
		amrex::Real coeffParallel;

		AMREX_GPU_HOST_DEVICE auto operator+(ReducedOperator const &rhs) const -> ReducedOperator
		{
			return {coeffIdentity + rhs.coeffIdentity, coeffCross + rhs.coeffCross, coeffParallel + rhs.coeffParallel};
		}

		AMREX_GPU_HOST_DEVICE auto operator-(ReducedOperator const &rhs) const -> ReducedOperator
		{
			return {coeffIdentity - rhs.coeffIdentity, coeffCross - rhs.coeffCross, coeffParallel - rhs.coeffParallel};
		}

		AMREX_GPU_HOST_DEVICE auto operator*(ReducedOperator const &rhs) const -> ReducedOperator
		{
			ReducedOperator result{};
			result.coeffIdentity = coeffIdentity * rhs.coeffIdentity - coeffCross * rhs.coeffCross;
			result.coeffCross = coeffIdentity * rhs.coeffCross + coeffCross * rhs.coeffIdentity;
			result.coeffParallel = coeffIdentity * rhs.coeffParallel + coeffParallel * rhs.coeffIdentity + coeffParallel * rhs.coeffParallel +
					       coeffCross * rhs.coeffCross;
			return result;
		}

		AMREX_GPU_HOST_DEVICE auto operator*(amrex::Real scale) const -> ReducedOperator
		{
			return {scale * coeffIdentity, scale * coeffCross, scale * coeffParallel};
		}

		friend AMREX_GPU_HOST_DEVICE auto operator*(amrex::Real scale, ReducedOperator const &op) -> ReducedOperator { return op * scale; }

		[[nodiscard]] AMREX_GPU_HOST_DEVICE auto inverse() const -> ReducedOperator
		{
			amrex::Real const denom_perp = coeffIdentity * coeffIdentity + coeffCross * coeffCross;
			amrex::Real const parallel_denom = coeffIdentity + coeffParallel;
			AMREX_ASSERT(denom_perp > 0.0);
			AMREX_ASSERT(std::abs(parallel_denom) > 0.0);
			amrex::Real const inv_parallel = 1.0 / parallel_denom;
			return {coeffIdentity / denom_perp, -coeffCross / denom_perp, inv_parallel - coeffIdentity / denom_perp};
		}

		[[nodiscard]] AMREX_GPU_HOST_DEVICE auto apply(Vec3 const &x, Vec3 const &b_hat) const -> Vec3
		{
			amrex::Real const x_parallel = b_hat.dot(x);
			Vec3 result = Vec3::Zero();
			result[0] = coeffIdentity * x[0] + coeffCross * (x[1] * b_hat[2] - x[2] * b_hat[1]) + coeffParallel * x_parallel * b_hat[0];
			result[1] = coeffIdentity * x[1] + coeffCross * (x[2] * b_hat[0] - x[0] * b_hat[2]) + coeffParallel * x_parallel * b_hat[1];
			result[2] = coeffIdentity * x[2] + coeffCross * (x[0] * b_hat[1] - x[1] * b_hat[0]) + coeffParallel * x_parallel * b_hat[2];
			return result;
		}
	};

	struct DustStageAffineOperators {
		ReducedOperator P1;
		ReducedOperator P2;
		ReducedOperator X1;
		ReducedOperator X2;
		ReducedOperator Y1;
		ReducedOperator Y2;
	};

	struct GasStageRates {
		Vec3 k1;
		Vec3 k2;
	};

	struct GirkCoefficients {
		// NOLINTNEXTLINE(misc-confusable-identifiers)
		amrex::Real gamma1 = 0.0;
		amrex::Real gamma2 = 0.0;
		amrex::Real beta1 = 0.0;
		amrex::Real beta2 = 0.0;
		amrex::Real b = 0.0;
	};

	struct DustCoefficientState {
		amrex::Real rhoGas;
		amrex::GpuArray<amrex::Real, nDustGroups_> rhoDust;
		amrex::GpuArray<amrex::Real, nDustGroups_> relativeVelocityMagnitude;
		amrex::Real soundSpeed;
	};
	AMREX_GPU_HOST_DEVICE static auto BuildDustCoefficientState(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> const &rho_d, Vec3 const &p_g,
								    amrex::GpuArray<Vec3, nDustGroups_> const &p_d, amrex::Real sound_speed)
	    -> DustCoefficientState;

	// compute reciprocal of dust stopping time
	AMREX_GPU_HOST_DEVICE static auto ComputeReciprocalStoppingTime(DustCoefficientState const &state) -> amrex::GpuArray<amrex::Real, nDustGroups_>;

	static AMREX_GPU_HOST_DEVICE auto ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
									    amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs,
									    amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius,
									    amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density,
									    bool enable_supersonic_correction) -> amrex::GpuArray<amrex::Real, nDustGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeSoundSpeedFromGasState(amrex::Real rho_g, amrex::Real gas_momentum_sq, amrex::Real E_tot_g,
									amrex::Real magnetic_energy,
									amrex::GpuArray<amrex::Real, nMassScalars_> const &massScalars) -> amrex::Real;
	AMREX_GPU_HOST_DEVICE static auto BuildCellCenteredMagneticField(int i, int j, int k,
									 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> Vec3;
	// compute dimensionless charge-to-mass ratio xi_i = q_i L_0 sqrt(rho_0) / (m_i c), where q_i is the Heaviside--Lorentz charge
	AMREX_GPU_HOST_DEVICE static auto ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const &state)
	    -> amrex::GpuArray<amrex::Real, nDustGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeDustStageAffineOperators(amrex::Real alpha1, amrex::Real omega_L1, amrex::Real alpha2, amrex::Real omega_L2,
									  amrex::Real epsilon, amrex::Real dt,
									  // NOLINTNEXTLINE(misc-confusable-identifiers)
									  amrex::Real gamma1, amrex::Real gamma2, amrex::Real beta1, amrex::Real beta2)
	    -> DustStageAffineOperators;
	AMREX_GPU_HOST_DEVICE static auto SolveGasStageRates(amrex::GpuArray<DustStageAffineOperators, nDustGroups_> const &ops,
							     amrex::GpuArray<Vec3, nDustGroups_> const &q_n, Vec3 const &b_hat) -> GasStageRates;
	AMREX_GPU_HOST_DEVICE static auto SelectGirkCoefficients(bool resolved_branch, quokka::dust::ResolvedRkScheme resolved_rk_scheme) -> GirkCoefficients;
	AMREX_GPU_HOST_DEVICE static auto ComputeMaximumDustSourceTimescale(amrex::GpuArray<amrex::Real, nDustGroups_> const &rho_d,
									    amrex::GpuArray<amrex::Real, nDustGroups_> const &alpha1,
									    amrex::GpuArray<amrex::Real, nDustGroups_> const &omega_L1,
									    amrex::GpuArray<amrex::Real, nDustGroups_> const &alpha2,
									    amrex::GpuArray<amrex::Real, nDustGroups_> const &omega_L2) -> amrex::Real;
	// compute dust source terms and update conserved variables
	static void computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
				    amrex::Real dust_omega_drag_, quokka::dust::CoefficientIterationConfig iteration_config, bool print_dust_counter_);
	static void computeDustDragAndLorentz(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					      amrex::Real dust_omega_drag_, amrex::Real dust_omega_gyro_res_,
					      quokka::dust::ResolvedRkScheme resolved_rk_scheme_, quokka::dust::CoefficientIterationConfig iteration_config,
					      bool print_dust_counter_);
	static void computeDustSource(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
				      amrex::Real dust_omega_drag_, amrex::Real dust_omega_gyro_res_, quokka::dust::ResolvedRkScheme resolved_rk_scheme_,
				      quokka::dust::CoefficientIterationConfig iteration_config, bool print_dust_counter_, bool include_lorentz_force);
};

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeReciprocalStoppingTime(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	alpha.fill(0.0);
	return alpha;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::BuildDustCoefficientState(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> const &rho_d,
									     Vec3 const &p_g, amrex::GpuArray<Vec3, nDustGroups_> const &p_d,
									     amrex::Real sound_speed) -> DustCoefficientState
{
	DustCoefficientState state{rho_g, rho_d, {}, sound_speed};
	Vec3 v_g = Vec3::Zero();
	if (rho_g > 0.0) {
		v_g = (1.0 / rho_g) * p_g;
	}
	for (int g = 0; g < nDustGroups_; ++g) {
		Vec3 v_d = Vec3::Zero();
		if (rho_d[g] > 0.0) {
			v_d = (1.0 / rho_d[g]) * p_d[g];
		}
		Vec3 const relative_velocity = v_d - v_g;
		state.relativeVelocityMagnitude[g] = std::sqrt(relative_velocity.dot(relative_velocity));
	}
	return state;
}

// compute reciprocal of physical dust stopping time following Kwok 1975 with optional supersonic correction
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
										     amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs,
										     amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius,
										     amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density,
										     bool enable_supersonic_correction)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};

	for (int g = 0; g < nDustGroups_; ++g) {
		if (rho_g <= 0.0 || rho_d[g] <= 0.0 || cs <= 0.0) {
			alpha[g] = 0.0;
			continue;
		}
		// compute stopping time t_s with/without supersonic correction
		amrex::Real const t_s_sub = std::sqrt(M_PI * ::quokka::EOS_Traits<problem_t>::gamma) * dust_grain_radius[g] * dust_grain_density[g] /
					    (2.0 * std::numbers::sqrt2 * rho_g * cs);
		amrex::Real const correction = 1.0 + static_cast<int>(enable_supersonic_correction) *
							 (9.0 * M_PI * ::quokka::EOS_Traits<problem_t>::gamma / 128.0) *
							 (rel_vel_mag[g] * rel_vel_mag[g] / (cs * cs));
		amrex::Real const t_s_fin = t_s_sub / std::sqrt(correction);

		alpha[g] = (t_s_fin > 0.0) ? 1.0 / t_s_fin : 0.0;
	}

	return alpha;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeSoundSpeedFromGasState(amrex::Real rho_g, amrex::Real gas_momentum_sq, amrex::Real E_tot_g,
										 amrex::Real magnetic_energy,
										 amrex::GpuArray<amrex::Real, nMassScalars_> const &massScalars) -> amrex::Real
{
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		amrex::ignore_unused(rho_g);
		amrex::ignore_unused(gas_momentum_sq);
		amrex::ignore_unused(E_tot_g);
		amrex::ignore_unused(magnetic_energy);
		amrex::ignore_unused(massScalars);
		return HydroSystem<problem_t>::cs_iso_;
	} else {
		AMREX_ALWAYS_ASSERT(rho_g > 0.0);
		amrex::Real const kinetic_energy = 0.5 * gas_momentum_sq / rho_g;
		amrex::Real const thermal_energy = E_tot_g - kinetic_energy - magnetic_energy;
		amrex::Real const pressure = ::quokka::EOS<problem_t>::ComputePressure(rho_g, thermal_energy, massScalars);
		return ::quokka::EOS<problem_t>::ComputeSoundSpeed(rho_g, pressure, massScalars);
	}
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::BuildCellCenteredMagneticField(int i, int j, int k,
										  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc)
    -> Vec3
{
	Vec3 B = Vec3::Zero();
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cons_fc != nullptr, "BuildCellCenteredMagneticField called without face-centered magnetic fields.");
		B[0] = 0.5 * ((*cons_fc)[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) +
			      (*cons_fc)[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex));
		B[1] = 0.5 * ((*cons_fc)[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) +
			      (*cons_fc)[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex));
		B[2] = 0.5 * ((*cons_fc)[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) +
			      (*cons_fc)[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex));
	}
	return B;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeDustDimensionlessChargeToMassRatio(DustCoefficientState const & /*state*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> dimensionless_charge_to_mass_ratio{};
	dimensionless_charge_to_mass_ratio.fill(0.0);
	return dimensionless_charge_to_mass_ratio;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeDustStageAffineOperators(amrex::Real alpha1, amrex::Real omega_L1, amrex::Real alpha2,
										   amrex::Real omega_L2, amrex::Real epsilon, amrex::Real dt,
										   // NOLINTNEXTLINE(misc-confusable-identifiers)
										   amrex::Real gamma1, amrex::Real gamma2, amrex::Real beta1, amrex::Real beta2)
    -> DustStageAffineOperators
{
	DustStageAffineOperators ops{};
	ReducedOperator const identity{1.0, 0.0, 0.0};
	ReducedOperator const T1 = {-alpha1, omega_L1, 0.0};
	ReducedOperator const T2 = {-alpha2, omega_L2, 0.0};
	ReducedOperator const L1 = dt * T1;
	ReducedOperator const L2 = dt * T2;

	ReducedOperator const block11 = identity - gamma1 * L1;
	ReducedOperator const block22 = identity - gamma2 * L2;
	ReducedOperator const D = block11 * block22 - (beta1 * beta2) * (L1 * L2);
	ReducedOperator const D_inv = D.inverse();

	ReducedOperator const H11 = block22 * D_inv;
	ReducedOperator const H12 = (beta1 * L1) * D_inv;
	ReducedOperator const H21 = (beta2 * L2) * D_inv;
	ReducedOperator const H22 = block11 * D_inv;

	ops.P1 = H11 * T1 + H12 * T2;
	ops.P2 = H21 * T1 + H22 * T2;
	ops.X1 = -epsilon * (gamma1 * (H11 * L1) + beta2 * (H12 * L2));
	ops.Y1 = -epsilon * (beta1 * (H11 * L1) + gamma2 * (H12 * L2));
	ops.X2 = -epsilon * (gamma1 * (H21 * L1) + beta2 * (H22 * L2));
	ops.Y2 = -epsilon * (beta1 * (H21 * L1) + gamma2 * (H22 * L2));

	return ops;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::SolveGasStageRates(amrex::GpuArray<DustStageAffineOperators, nDustGroups_> const &ops,
								      amrex::GpuArray<Vec3, nDustGroups_> const &q_n, Vec3 const &b_hat) -> GasStageRates
{
	GasStageRates rates{Vec3::Zero(), Vec3::Zero()};
	ReducedOperator lambda11{1.0, 0.0, 0.0};
	ReducedOperator lambda12{0.0, 0.0, 0.0};
	ReducedOperator lambda21{0.0, 0.0, 0.0};
	ReducedOperator lambda22{1.0, 0.0, 0.0};
	Vec3 r1 = Vec3::Zero();
	Vec3 r2 = Vec3::Zero();

	for (int g = 0; g < nDustGroups_; ++g) {
		lambda11 = lambda11 + ops[g].X1;
		lambda12 = lambda12 + ops[g].Y1;
		lambda21 = lambda21 + ops[g].X2;
		lambda22 = lambda22 + ops[g].Y2;
		r1 += ops[g].P1.apply(q_n[g], b_hat);
		r2 += ops[g].P2.apply(q_n[g], b_hat);
	}

	ReducedOperator const delta_g = lambda11 * lambda22 - lambda12 * lambda21;
	ReducedOperator const delta_g_inv = delta_g.inverse();
	Vec3 const rhs1 = lambda22.apply(r1, b_hat) - lambda12.apply(r2, b_hat);
	Vec3 const rhs2 = lambda11.apply(r2, b_hat) - lambda21.apply(r1, b_hat);

	rates.k1 = -1.0 * delta_g_inv.apply(rhs1, b_hat);
	rates.k2 = -1.0 * delta_g_inv.apply(rhs2, b_hat);
	return rates;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::SelectGirkCoefficients(bool resolved_branch, quokka::dust::ResolvedRkScheme resolved_rk_scheme)
    -> GirkCoefficients
{
	if (!resolved_branch) {
		return {.gamma1 = 1.0, .gamma2 = 1.0, .beta1 = 1.0, .beta2 = -1.0, .b = 0.0};
	}
	switch (resolved_rk_scheme) {
		case quokka::dust::ResolvedRkScheme::TP2025:
			return {.gamma1 = 1.0, .gamma2 = 0.0, .beta1 = -0.5, .beta2 = 2.0 / 3.0, .b = 1.0};
		case quokka::dust::ResolvedRkScheme::GL4:
			return {.gamma1 = 0.25, .gamma2 = 0.25, .beta1 = 0.25 - std::numbers::sqrt3 / 6.0, .beta2 = 0.25 + std::numbers::sqrt3 / 6.0, .b = 0.5};
		case quokka::dust::ResolvedRkScheme::Midpoint:
			return {.gamma1 = 0.25, .gamma2 = 0.25, .beta1 = 0.25, .beta2 = 0.25, .b = 0.5};
	}
	return {};
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustSources<problem_t>::ComputeMaximumDustSourceTimescale(amrex::GpuArray<amrex::Real, nDustGroups_> const &rho_d,
										     amrex::GpuArray<amrex::Real, nDustGroups_> const &alpha1,
										     amrex::GpuArray<amrex::Real, nDustGroups_> const &omega_L1,
										     amrex::GpuArray<amrex::Real, nDustGroups_> const &alpha2,
										     amrex::GpuArray<amrex::Real, nDustGroups_> const &omega_L2) -> amrex::Real
{
	amrex::Real maximum_timescale = 0.0;
	for (int g = 0; g < nDustGroups_; ++g) {
		if (rho_d[g] <= 0.0) {
			continue;
		}
		amrex::Real const rate_magnitude1 = std::sqrt(alpha1[g] * alpha1[g] + omega_L1[g] * omega_L1[g]);
		amrex::Real const rate_magnitude2 = std::sqrt(alpha2[g] * alpha2[g] + omega_L2[g] * omega_L2[g]);
		amrex::Real const timescale1 = (rate_magnitude1 > 0.0) ? 1.0 / rate_magnitude1 : std::numeric_limits<amrex::Real>::max();
		amrex::Real const timescale2 = (rate_magnitude2 > 0.0) ? 1.0 / rate_magnitude2 : std::numeric_limits<amrex::Real>::max();
		maximum_timescale = amrex::max(maximum_timescale, amrex::max(timescale1, timescale2));
	}
	return maximum_timescale;
}

template <typename problem_t>
void DustSources<problem_t>::computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					     amrex::Real dust_omega_drag_, quokka::dust::CoefficientIterationConfig iteration_config, bool print_dust_counter_)
{
	computeDustSource(consVar_cc_mf, consVar_fc_mf, dt, dust_omega_drag_, 0.0, quokka::dust::ResolvedRkScheme::TP2025, iteration_config,
			  print_dust_counter_, false);
}

template <typename problem_t>
void DustSources<problem_t>::computeDustDragAndLorentz(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf,
						       amrex::Real dt, amrex::Real dust_omega_drag_, amrex::Real dust_omega_gyro_res_,
						       quokka::dust::ResolvedRkScheme resolved_rk_scheme_,
						       quokka::dust::CoefficientIterationConfig iteration_config, bool print_dust_counter_)
{
	computeDustSource(consVar_cc_mf, consVar_fc_mf, dt, dust_omega_drag_, dust_omega_gyro_res_, resolved_rk_scheme_, iteration_config, print_dust_counter_,
			  true);
}

template <typename problem_t>
void DustSources<problem_t>::computeDustSource(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					       amrex::Real dust_omega_drag_, amrex::Real dust_omega_gyro_res_,
					       quokka::dust::ResolvedRkScheme resolved_rk_scheme_, quokka::dust::CoefficientIterationConfig iteration_config,
					       bool print_dust_counter_, bool include_lorentz_force)
{
	amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0}); // [sum of iterations, number of cells, max iterations in any cell, unconverged cells]
	int *p_iteration_counter = iteration_counter.data();
	auto const &consVar_cc = consVar_cc_mf.arrays();
	auto const &cons_fc_x0 = consVar_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = consVar_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = consVar_fc_mf[2].const_arrays();
#endif

	int const numDustVars = Physics_NumVars::numDustVarsPerGroup;
	amrex::Real const omega_drag = dust_omega_drag_;
	amrex::Real const omega_gyro_res = dust_omega_gyro_res_;
	bool const iteration_enabled = iteration_config.enabled;
	amrex::Real const alpha_relative_tolerance = iteration_config.alphaRelativeTolerance;
	amrex::Real const charge_absolute_tolerance = iteration_config.chargeAbsoluteTolerance;
	amrex::Real const charge_relative_tolerance = iteration_config.chargeRelativeTolerance;
	int const configured_max_iterations = iteration_config.maxIterations;

	amrex::ParallelFor(consVar_cc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) {
			cons_fc[0] = cons_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
			cons_fc[1] = cons_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
			cons_fc[2] = cons_fc_x2[bx];
#endif
		}

		amrex::Real const rho_g = consVar_cc[bx](i, j, k, density_index);
		amrex::Real const E_tot = consVar_cc[bx](i, j, k, energy_index);
		amrex::Real const E_int = consVar_cc[bx](i, j, k, internalEnergy_index);

		amrex::GpuArray<amrex::Real, nDustGroups_> rho_d{};
		amrex::GpuArray<amrex::Real, nDustGroups_> epsilon{};
		for (int g = 0; g < nDustGroups_; ++g) {
			rho_d[g] = consVar_cc[bx](i, j, k, dustDensity_index + g * numDustVars);
			epsilon[g] = (rho_g > 0.0) ? rho_d[g] / rho_g : 0.0;
		}

		Vec3 p_g_old = Vec3::Zero();
		amrex::GpuArray<Vec3, nDustGroups_> p_d_old{};
		for (int g = 0; g < nDustGroups_; ++g) {
			p_d_old[g] = Vec3::Zero();
		}
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			p_g_old[dir] = consVar_cc[bx](i, j, k, x1Momentum_index + dir);
			for (int g = 0; g < nDustGroups_; ++g) {
				p_d_old[g][dir] = consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars);
			}
		}

		const int max_iterations = iteration_enabled ? configured_max_iterations : 1;
		int cell_iteration_count = 0;
		bool iteration_converged = !iteration_enabled;
		amrex::GpuArray<amrex::Real, nMassScalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(consVar_cc[bx], i, j, k);

		Vec3 p_g_iter_new = p_g_old;
		amrex::GpuArray<Vec3, nDustGroups_> p_d_iter_new = p_d_old;
		Vec3 p_g_stage1_old = p_g_old;
		Vec3 p_g_stage2_old = p_g_old;
		amrex::GpuArray<Vec3, nDustGroups_> p_d_stage1_old = p_d_old;
		amrex::GpuArray<Vec3, nDustGroups_> p_d_stage2_old = p_d_old;
		amrex::GpuArray<Vec3, nDustGroups_> q_n{};
		amrex::Real E_tot_iter_new = E_tot;
		amrex::Real E_int_iter_new = E_int;
		amrex::Real E_g_stage1_old = E_tot;
		amrex::Real E_g_stage2_old = E_tot;
		Vec3 const B_cc = BuildCellCenteredMagneticField(i, j, k, &cons_fc);
		amrex::Real const magnetic_energy = 0.5 * B_cc.dot(B_cc);
		amrex::Real const B_mag = include_lorentz_force ? std::sqrt(B_cc.dot(B_cc)) : 0.0;
		Vec3 b_hat = Vec3::Zero();
		if (B_mag > 0.0) {
			b_hat = (1.0 / B_mag) * B_cc;
		}
		amrex::Real const dt_lev = 2.0 * dt;
		for (int g = 0; g < nDustGroups_; ++g) {
			// initial relative momentum used for GIRK; do not update inside Picard loop
			q_n[g] = p_d_old[g] - epsilon[g] * p_g_old;
		}

		for (int iteration = 0; iteration < max_iterations; ++iteration) {
			if (print_dust_counter_) {
				cell_iteration_count++;
			}
			amrex::Real const cs1 =
			    ComputeSoundSpeedFromGasState(rho_g, p_g_stage1_old.dot(p_g_stage1_old), E_g_stage1_old, magnetic_energy, massScalars);
			amrex::Real const cs2 =
			    ComputeSoundSpeedFromGasState(rho_g, p_g_stage2_old.dot(p_g_stage2_old), E_g_stage2_old, magnetic_energy, massScalars);

			DustCoefficientState const coefficient_state1 = BuildDustCoefficientState(rho_g, rho_d, p_g_stage1_old, p_d_stage1_old, cs1);
			DustCoefficientState const coefficient_state2 = BuildDustCoefficientState(rho_g, rho_d, p_g_stage2_old, p_d_stage2_old, cs2);
			auto const alpha1 = ComputeReciprocalStoppingTime(coefficient_state1);
			auto const alpha2 = ComputeReciprocalStoppingTime(coefficient_state2);
			amrex::GpuArray<amrex::Real, nDustGroups_> charge1{};
			amrex::GpuArray<amrex::Real, nDustGroups_> charge2{};
			if (include_lorentz_force) {
				charge1 = ComputeDustDimensionlessChargeToMassRatio(coefficient_state1);
				charge2 = ComputeDustDimensionlessChargeToMassRatio(coefficient_state2);
			}
			amrex::GpuArray<amrex::Real, nDustGroups_> omega_L1{};
			amrex::GpuArray<amrex::Real, nDustGroups_> omega_L2{};
			for (int g = 0; g < nDustGroups_; ++g) {
				omega_L1[g] = charge1[g] * B_mag;
				omega_L2[g] = charge2[g] * B_mag;
			}

			amrex::Real const maximum_source_timescale = ComputeMaximumDustSourceTimescale(rho_d, alpha1, omega_L1, alpha2, omega_L2);
			bool const resolved_branch = dt_lev < maximum_source_timescale;
			GirkCoefficients const coefficients = SelectGirkCoefficients(resolved_branch, resolved_rk_scheme_);

			amrex::GpuArray<DustStageAffineOperators, nDustGroups_> ops{};
			for (int g = 0; g < nDustGroups_; ++g) {
				ops[g] = ComputeDustStageAffineOperators(alpha1[g], omega_L1[g], alpha2[g], omega_L2[g], epsilon[g], dt, coefficients.gamma1,
									 coefficients.gamma2, coefficients.beta1, coefficients.beta2);
			}

			GasStageRates const gas_stage = SolveGasStageRates(ops, q_n, b_hat);
			Vec3 const k1_g = gas_stage.k1;
			Vec3 const k2_g = gas_stage.k2;

			amrex::GpuArray<Vec3, nDustGroups_> k1_d{};
			amrex::GpuArray<Vec3, nDustGroups_> k2_d{};
			amrex::GpuArray<Vec3, nDustGroups_> q1{};
			amrex::GpuArray<Vec3, nDustGroups_> q2{};
			for (int g = 0; g < nDustGroups_; ++g) {
				k1_d[g] = ops[g].P1.apply(q_n[g], b_hat) + ops[g].X1.apply(k1_g, b_hat) + ops[g].Y1.apply(k2_g, b_hat);
				k2_d[g] = ops[g].P2.apply(q_n[g], b_hat) + ops[g].X2.apply(k1_g, b_hat) + ops[g].Y2.apply(k2_g, b_hat);

				Vec3 const k_rel1 = k1_d[g] - epsilon[g] * k1_g;
				Vec3 const k_rel2 = k2_d[g] - epsilon[g] * k2_g;
				q1[g] = q_n[g] + dt * (coefficients.gamma1 * k_rel1 + coefficients.beta1 * k_rel2);
				q2[g] = q_n[g] + dt * (coefficients.beta2 * k_rel1 + coefficients.gamma2 * k_rel2);
			}

			Vec3 const p_g_stage1_new = p_g_old + dt * (coefficients.gamma1 * k1_g + coefficients.beta1 * k2_g);
			Vec3 const p_g_stage2_new = p_g_old + dt * (coefficients.beta2 * k1_g + coefficients.gamma2 * k2_g);
			p_g_iter_new = p_g_old + dt * (coefficients.b * k1_g + (1.0 - coefficients.b) * k2_g);
			amrex::GpuArray<Vec3, nDustGroups_> p_d_stage1_new{};
			amrex::GpuArray<Vec3, nDustGroups_> p_d_stage2_new{};
			for (int g = 0; g < nDustGroups_; ++g) {
				p_d_stage1_new[g] = p_d_old[g] + dt * (coefficients.gamma1 * k1_d[g] + coefficients.beta1 * k2_d[g]);
				p_d_stage2_new[g] = p_d_old[g] + dt * (coefficients.beta2 * k1_d[g] + coefficients.gamma2 * k2_d[g]);
				p_d_iter_new[g] = p_d_old[g] + dt * (coefficients.b * k1_d[g] + (1.0 - coefficients.b) * k2_d[g]);
			}

			amrex::Real stage_heat_rate1 = 0.0;
			amrex::Real stage_heat_rate2 = 0.0;
			for (int g = 0; g < nDustGroups_; ++g) {
				if (rho_d[g] > 0.0) {
					stage_heat_rate1 += alpha1[g] * q1[g].dot(q1[g]) / rho_d[g];
					stage_heat_rate2 += alpha2[g] * q2[g].dot(q2[g]) / rho_d[g];
				}
			}
			amrex::Real gas_energy_rate1 = omega_drag * stage_heat_rate1;
			amrex::Real gas_energy_rate2 = omega_drag * stage_heat_rate2;
			if (rho_g > 0.0) {
				gas_energy_rate1 += p_g_stage1_new.dot(k1_g) / rho_g;
				gas_energy_rate2 += p_g_stage2_new.dot(k2_g) / rho_g;
			}
			amrex::Real const E_g_stage1_new = E_tot + dt * (coefficients.gamma1 * gas_energy_rate1 + coefficients.beta1 * gas_energy_rate2);
			amrex::Real const E_g_stage2_new = E_tot + dt * (coefficients.beta2 * gas_energy_rate1 + coefficients.gamma2 * gas_energy_rate2);

			amrex::Real delta_E_g_work = 0.0;
			if (rho_g > 0.0) {
				delta_E_g_work = (p_g_iter_new.dot(p_g_iter_new) - p_g_old.dot(p_g_old)) / (2.0 * rho_g);
			}

			amrex::Real delta_E_d_work_sum = 0.0;
			for (int g = 0; g < nDustGroups_; ++g) {
				if (rho_d[g] > 0.0) {
					delta_E_d_work_sum += (p_d_iter_new[g].dot(p_d_iter_new[g]) - p_d_old[g].dot(p_d_old[g])) / (2.0 * rho_d[g]);
				}
			}

			amrex::Real const delta_E_cons = -(delta_E_g_work + delta_E_d_work_sum);
			amrex::Real delta_E_res_gyro = 0.0;
			if (include_lorentz_force) {
				amrex::Real delta_E_heat_drag = 0.0;
				Vec3 k1_g_drag = Vec3::Zero();
				Vec3 k2_g_drag = Vec3::Zero();
				amrex::Real inner_drag_11 = 0.0;
				amrex::Real inner_drag_12 = 0.0;
				amrex::Real inner_drag_22 = 0.0;
				for (int g = 0; g < nDustGroups_; ++g) {
					if (rho_d[g] > 0.0) {
						delta_E_heat_drag +=
						    dt / rho_d[g] *
						    (coefficients.b * alpha1[g] * q1[g].dot(q1[g]) + (1.0 - coefficients.b) * alpha2[g] * q2[g].dot(q2[g]));

						Vec3 const k1_d_drag = -alpha1[g] * q1[g];
						Vec3 const k2_d_drag = -alpha2[g] * q2[g];
						k1_g_drag -= k1_d_drag;
						k2_g_drag -= k2_d_drag;
						inner_drag_11 += k1_d_drag.dot(k1_d_drag) / rho_d[g];
						inner_drag_12 += k1_d_drag.dot(k2_d_drag) / rho_d[g];
						inner_drag_22 += k2_d_drag.dot(k2_d_drag) / rho_d[g];
					}
				}
				if (rho_g > 0.0) {
					inner_drag_11 += k1_g_drag.dot(k1_g_drag) / rho_g;
					inner_drag_12 += k1_g_drag.dot(k2_g_drag) / rho_g;
					inner_drag_22 += k2_g_drag.dot(k2_g_drag) / rho_g;
				}

				amrex::Real const delta_E_res = delta_E_cons - delta_E_heat_drag;
				amrex::Real const b1 = coefficients.b;
				amrex::Real const b2 = 1.0 - coefficients.b;
				amrex::Real const m11 = 2.0 * b1 * coefficients.gamma1 - b1 * b1;
				amrex::Real const m22 = 2.0 * b2 * coefficients.gamma2 - b2 * b2;
				amrex::Real const m12 = b1 * coefficients.beta1 + b2 * coefficients.beta2 - b1 * b2;
				amrex::Real const delta_E_res_drag_only =
				    0.5 * dt * dt * (m11 * inner_drag_11 + 2.0 * m12 * inner_drag_12 + m22 * inner_drag_22);
				delta_E_res_gyro = delta_E_res - delta_E_res_drag_only;
			}
			E_tot_iter_new = E_tot + delta_E_g_work + omega_drag * (delta_E_cons - delta_E_res_gyro) + omega_gyro_res * delta_E_res_gyro;
			E_int_iter_new = E_int + omega_drag * (delta_E_cons - delta_E_res_gyro) + omega_gyro_res * delta_E_res_gyro;

			if (!iteration_enabled) {
				break;
			}

			amrex::Real const cs1_new =
			    ComputeSoundSpeedFromGasState(rho_g, p_g_stage1_new.dot(p_g_stage1_new), E_g_stage1_new, magnetic_energy, massScalars);
			amrex::Real const cs2_new =
			    ComputeSoundSpeedFromGasState(rho_g, p_g_stage2_new.dot(p_g_stage2_new), E_g_stage2_new, magnetic_energy, massScalars);
			DustCoefficientState const coefficient_state1_new = BuildDustCoefficientState(rho_g, rho_d, p_g_stage1_new, p_d_stage1_new, cs1_new);
			DustCoefficientState const coefficient_state2_new = BuildDustCoefficientState(rho_g, rho_d, p_g_stage2_new, p_d_stage2_new, cs2_new);
			auto const alpha1_new = ComputeReciprocalStoppingTime(coefficient_state1_new);
			auto const alpha2_new = ComputeReciprocalStoppingTime(coefficient_state2_new);
			amrex::GpuArray<amrex::Real, nDustGroups_> charge1_new{};
			amrex::GpuArray<amrex::Real, nDustGroups_> charge2_new{};
			if (include_lorentz_force) {
				charge1_new = ComputeDustDimensionlessChargeToMassRatio(coefficient_state1_new);
				charge2_new = ComputeDustDimensionlessChargeToMassRatio(coefficient_state2_new);
			}
			bool alpha_converged = true;
			bool charge_converged = true;
			amrex::GpuArray<amrex::Real, nDustGroups_> omega_L1_new{};
			amrex::GpuArray<amrex::Real, nDustGroups_> omega_L2_new{};
			for (int g = 0; g < nDustGroups_; ++g) {
				omega_L1_new[g] = charge1_new[g] * B_mag;
				omega_L2_new[g] = charge2_new[g] * B_mag;
				if (rho_d[g] <= 0.0) {
					continue;
				}
				alpha_converged = alpha_converged && (std::abs(alpha1_new[g] - alpha1[g]) <= alpha_relative_tolerance * alpha1[g]) &&
						  (std::abs(alpha2_new[g] - alpha2[g]) <= alpha_relative_tolerance * alpha2[g]);
				if (include_lorentz_force && B_mag > 0.0) {
					charge_converged = charge_converged &&
							   (std::abs(charge1_new[g] - charge1[g]) <=
							    charge_absolute_tolerance + charge_relative_tolerance * std::abs(charge1[g])) &&
							   (std::abs(charge2_new[g] - charge2[g]) <=
							    charge_absolute_tolerance + charge_relative_tolerance * std::abs(charge2[g]));
				}
			}
			amrex::Real const maximum_source_timescale_new =
			    ComputeMaximumDustSourceTimescale(rho_d, alpha1_new, omega_L1_new, alpha2_new, omega_L2_new);
			bool const branch_converged = (dt_lev < maximum_source_timescale_new) == resolved_branch;
			iteration_converged = alpha_converged && charge_converged && branch_converged;
			if (iteration_converged) {
				break;
			}

			p_g_stage1_old = p_g_stage1_new;
			p_g_stage2_old = p_g_stage2_new;
			p_d_stage1_old = p_d_stage1_new;
			p_d_stage2_old = p_d_stage2_new;
			E_g_stage1_old = E_g_stage1_new;
			E_g_stage2_old = E_g_stage2_new;
		}

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			consVar_cc[bx](i, j, k, x1Momentum_index + dir) = p_g_iter_new[dir];
			for (int g = 0; g < nDustGroups_; ++g) {
				consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars) = p_d_iter_new[g][dir];
			}
		}
		consVar_cc[bx](i, j, k, energy_index) = E_tot_iter_new;
		consVar_cc[bx](i, j, k, internalEnergy_index) = E_int_iter_new;

		if (print_dust_counter_) {
			amrex::Gpu::Atomic::Add(&p_iteration_counter[0], cell_iteration_count); // sum of iterations
			amrex::Gpu::Atomic::Add(&p_iteration_counter[1], 1);			// number of cells
			amrex::Gpu::Atomic::Max(&p_iteration_counter[2], cell_iteration_count); // max iterations in any cell
		}
		if (!iteration_converged) {
			amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1);
		}
	});
	if (print_dust_counter_ || iteration_enabled) {
		auto *h_iteration_counter = iteration_counter.copyToHost();
		int unconverged_cells = h_iteration_counter[3];
		if (iteration_enabled) {
			amrex::ParallelDescriptor::ReduceIntSum(unconverged_cells);
			if (amrex::ParallelDescriptor::IOProcessor() && unconverged_cells > 0) {
				amrex::Print() << "WARNING: Dust " << (include_lorentz_force ? "drag and Lorentz" : "drag")
					       << " Picard iteration did not converge in " << unconverged_cells << " cell(s); using the final iterate.\n";
			}
		}
		if (print_dust_counter_) {
			long global_iteration_sum = h_iteration_counter[0]; // NOLINT(google-runtime-int)
			long global_cell_count = h_iteration_counter[1];    // NOLINT(google-runtime-int)
			int global_max_iterations = h_iteration_counter[2];

			amrex::ParallelDescriptor::ReduceLongSum(global_iteration_sum);
			amrex::ParallelDescriptor::ReduceLongSum(global_cell_count);
			amrex::ParallelDescriptor::ReduceIntMax(global_max_iterations);

			if (amrex::ParallelDescriptor::IOProcessor() && global_cell_count > 0) {
				const double avg_iterations = static_cast<double>(global_iteration_sum) / static_cast<double>(global_cell_count);
				amrex::Print() << "Dust " << (include_lorentz_force ? "drag and Lorentz" : "drag") << " Picard iteration statistics:\n";
				amrex::Print() << "  total cells updated: " << global_cell_count << "\n";
				amrex::Print() << "  average iterations per cell: " << avg_iterations << "\n";
				amrex::Print() << "  maximum iterations in any cell: " << global_max_iterations << "\n";
			}
		}
	}
}
