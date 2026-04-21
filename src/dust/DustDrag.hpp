//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file DustDrag.hpp
/// \brief Defines a class for integrating dust-gas drag force and dust Lorentz force.
///

#include "AMReX_LUSolver.H"
#include "AMReX_MultiFab.H"
#include "AMReX_SmallMatrix.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/ArrayView_3d.hpp"
#include <numbers>

template <typename problem_t> class DustDrag
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
	using Mat3 = amrex::SmallMatrix<amrex::Real, 3, 3>;
	using Vec6 = amrex::Array1D<amrex::Real, 0, 5>;
	using Mat6 = amrex::Array2D<amrex::Real, 0, 5, 0, 5, amrex::Order::C>;

	struct DustStageAffineOperators {
		Mat3 W1;
		Mat3 W2;
		Mat3 X1;
		Mat3 X2;
		Mat3 Y1;
		Mat3 Y2;
	};

	// compute reciprocal of dust stopping time
	AMREX_GPU_HOST_DEVICE static auto ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
									amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
	    -> amrex::GpuArray<amrex::Real, nDustGroups_>;

	static AMREX_GPU_HOST_DEVICE auto ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
									    amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs,
									    amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius,
									    amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density,
									    bool enable_supersonic_correction) -> amrex::GpuArray<amrex::Real, nDustGroups_>;
	AMREX_GPU_HOST_DEVICE static auto BuildCrossMatrix(Vec3 const &b_hat) -> Mat3;
	AMREX_GPU_HOST_DEVICE static auto ComputeSoundSpeedFromGasState(amrex::Real rho_g, amrex::Real gas_momentum_sq, amrex::Real E_tot_g,
									amrex::Real magnetic_energy,
									amrex::GpuArray<amrex::Real, nMassScalars_> const &massScalars) -> amrex::Real;
	AMREX_GPU_HOST_DEVICE static auto BuildCellCenteredMagneticField(int i, int j, int k,
									 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> Vec3;
	AMREX_GPU_HOST_DEVICE static auto ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeDustStageAffineOperators(Mat3 const &T, amrex::Real epsilon, amrex::Real dt, amrex::Real gamma1,
									  amrex::Real gamma2, amrex::Real beta1, amrex::Real beta2) -> DustStageAffineOperators;
	// compute dust-gas drag source terms and update conserved variables
	static void computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
				    amrex::Real dust_omega_, int enableIterDustStoptime_, bool print_dust_counter_);
	static void computeDustDragLorentz(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					   amrex::Real dust_omega1_, amrex::Real dust_omega2_, int enableIterDustStoptime_, bool print_dust_counter_);
};

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
									      amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha;
	alpha.fill(0.0);
	return alpha;
}

// compute reciprocal of physical dust stopping time following Kwok 1975 with optional supersonic correction
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
										  amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs,
										  amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius,
										  amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density,
										  bool enable_supersonic_correction)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha;

	for (int g = 0; g < nDustGroups_; ++g) {
		if (rho_g <= 0.0 || rho_d[g] <= 0.0 || cs <= 0.0) {
			alpha[g] = 0.0;
			continue;
		}
		// compute stopping time t_s with/without supersonic correction
		amrex::Real t_s_sub = std::sqrt(M_PI * quokka::EOS_Traits<problem_t>::gamma) * dust_grain_radius[g] * dust_grain_density[g] /
				      (2.0 * std::numbers::sqrt2 * rho_g * cs);
		amrex::Real const correction = 1.0 + static_cast<int>(enable_supersonic_correction) *
							 (9.0 * M_PI * quokka::EOS_Traits<problem_t>::gamma / 128.0) *
							 (rel_vel_mag[g] * rel_vel_mag[g] / (cs * cs));
		amrex::Real const t_s_fin = t_s_sub / std::sqrt(correction);

		alpha[g] = (t_s_fin > 0.0) ? 1.0 / t_s_fin : 0.0;
	}

	return alpha;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::BuildCrossMatrix(Vec3 const &b_hat) -> Mat3
{
	Mat3 result = Mat3::Zero();
	result(0, 1) = b_hat[2];
	result(0, 2) = -b_hat[1];
	result(1, 0) = -b_hat[2];
	result(1, 2) = b_hat[0];
	result(2, 0) = b_hat[1];
	result(2, 1) = -b_hat[0];
	return result;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeSoundSpeedFromGasState(amrex::Real rho_g, amrex::Real gas_momentum_sq, amrex::Real E_tot_g,
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
		amrex::Real const pressure = quokka::EOS<problem_t>::ComputePressure(rho_g, thermal_energy, massScalars);
		return quokka::EOS<problem_t>::ComputeSoundSpeed(rho_g, pressure, massScalars);
	}
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::BuildCellCenteredMagneticField(int i, int j, int k,
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

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeDustChargeToMassRatio() -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> charge_to_mass_ratio;
	for (int g = 0; g < nDustGroups_; ++g) {
		charge_to_mass_ratio[g] = 1.0;
	}
	return charge_to_mass_ratio;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeDustStageAffineOperators(Mat3 const &T, amrex::Real epsilon, amrex::Real dt, amrex::Real gamma1,
										amrex::Real gamma2, amrex::Real beta1, amrex::Real beta2)
    -> DustStageAffineOperators
{
	DustStageAffineOperators ops;
	Mat6 S;
	Mat3 const I3 = Mat3::Identity();
	Mat3 const block11 = I3 + (gamma1 * dt) * T;
	Mat3 const block12 = (beta1 * dt) * T;
	Mat3 const block21 = (beta2 * dt) * T;
	Mat3 const block22 = I3 + (gamma2 * dt) * T;

	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) {
			S(row, col) = block11(row, col);
			S(row, col + 3) = block12(row, col);
			S(row + 3, col) = block21(row, col);
			S(row + 3, col + 3) = block22(row, col);
		}
	}
	amrex::LUSolver<6, amrex::Real> const solver(S);

	for (int basis = 0; basis < 3; ++basis) {
		Vec3 e = Vec3::Zero();
		e[basis] = 1.0;

		Vec3 const T_e = T * e;

		Vec6 rhs_q;
		Vec6 rhs_k1;
		Vec6 rhs_k2;
		for (int dir = 0; dir < 3; ++dir) {
			rhs_q(dir) = T_e[dir];
			rhs_q(dir + 3) = T_e[dir];
			rhs_k1(dir) = gamma1 * dt * T_e[dir];
			rhs_k1(dir + 3) = beta2 * dt * T_e[dir];
			rhs_k2(dir) = beta1 * dt * T_e[dir];
			rhs_k2(dir + 3) = gamma2 * dt * T_e[dir];
		}

		Vec6 sol_q;
		Vec6 sol_k1;
		Vec6 sol_k2;
		solver(sol_q.begin(), rhs_q.begin());
		solver(sol_k1.begin(), rhs_k1.begin());
		solver(sol_k2.begin(), rhs_k2.begin());

		for (int row = 0; row < 3; ++row) {
			ops.W1(row, basis) = sol_q(row);
			ops.W2(row, basis) = sol_q(row + 3);
			ops.X1(row, basis) = epsilon * sol_k1(row);
			ops.X2(row, basis) = epsilon * sol_k1(row + 3);
			ops.Y1(row, basis) = epsilon * sol_k2(row);
			ops.Y2(row, basis) = epsilon * sol_k2(row + 3);
		}
	}

	return ops;
}

template <typename problem_t>
void DustDrag<problem_t>::computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					  amrex::Real dust_omega_, int enableIterDustStoptime_, bool print_dust_counter_)
{
	amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0}); // [sum of iterations, number of cells, max iterations in any cell]
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
	amrex::Real const omega = dust_omega_;

	// NOLINTNEXTLINE(modernize-use-trailing-return-type)
	amrex::ParallelFor(consVar_cc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if (Physics_Traits<problem_t>::is_mhd_enabled) { // if instead of if constexpr to avoid nvcc issues
			cons_fc[0] = cons_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
			cons_fc[1] = cons_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
			cons_fc[2] = cons_fc_x2[bx];
#endif
		}
		amrex::Real rho_g = consVar_cc[bx](i, j, k, density_index);
		amrex::Real E_tot = consVar_cc[bx](i, j, k, energy_index);
		amrex::Real E_int = consVar_cc[bx](i, j, k, internalEnergy_index);

		amrex::GpuArray<amrex::Real, nDustGroups_> rho_d;
		for (int g = 0; g < nDustGroups_; ++g) {
			rho_d[g] = consVar_cc[bx](i, j, k, dustDensity_index + g * numDustVars);
		}

		amrex::GpuArray<amrex::Real, nDustGroups_> epsilon;
		for (int g = 0; g < nDustGroups_; ++g) {
			epsilon[g] = (rho_g > 0.0) ? rho_d[g] / rho_g : 0.0;
		}

		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_g_old{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, nDustGroups_> vel_d_old;

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			int mom_g_idx = x1Momentum_index + dir;
			vel_g_old[dir] = (rho_g > 0.0) ? consVar_cc[bx](i, j, k, mom_g_idx) / rho_g : 0.0;

			for (int g = 0; g < nDustGroups_; ++g) {
				int mom_d_idx = x1DustMomentum_index + dir + g * numDustVars;
				vel_d_old[g][dir] = (rho_d[g] > 0.0) ? consVar_cc[bx](i, j, k, mom_d_idx) / rho_d[g] : 0.0;
			}
		}

		// set iteration parameters
		const int max_iterations = (enableIterDustStoptime_ != 0) ? 20 : 1;
		const amrex::Real tolerance = 1.0e-6;
		int cell_iteration_count = 0;
		amrex::Real const dt_lev = 2.0 * dt;
		amrex::GpuArray<amrex::Real, nMassScalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(consVar_cc[bx], i, j, k);
		amrex::Real const magnetic_energy = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &cons_fc);
		amrex::Real E_tot_iter_old = E_tot;
		amrex::Real E_tot_iter_new = E_tot;
		amrex::Real E_int_iter_new = E_int;

		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, nDustGroups_ + 1> vel_iter_old;
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, nDustGroups_ + 1> vel_iter_new;

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			vel_iter_old[0][dir] = vel_g_old[dir];
			for (int g = 0; g < nDustGroups_; ++g) {
				vel_iter_old[1 + g][dir] = vel_d_old[g][dir];
			}
		}

		// Picard iteration loop
		for (int iteration = 0; iteration < max_iterations; ++iteration) {
			cell_iteration_count++;
			// compute sound speed for stopping time calculation
			amrex::Real gas_momentum_sq = 0.0;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				gas_momentum_sq += rho_g * rho_g * vel_iter_old[0][dir] * vel_iter_old[0][dir];
			}
			amrex::Real const cs = ComputeSoundSpeedFromGasState(rho_g, gas_momentum_sq, E_tot_iter_old, magnetic_energy, massScalars);

			amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag;
			for (int g = 0; g < nDustGroups_; ++g) {
				amrex::Real rel_speed_sq = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					rel_speed_sq += (vel_iter_old[1 + g][dir] - vel_iter_old[0][dir]) * (vel_iter_old[1 + g][dir] - vel_iter_old[0][dir]);
				}
				rel_vel_mag[g] = std::sqrt(rel_speed_sq);
			}

			amrex::GpuArray<amrex::Real, nDustGroups_> alpha = ComputeReciprocalStoppingTime(rho_g, rho_d, rel_vel_mag, cs);
			amrex::Real t_s_max = 0.0;
			for (int g = 0; g < nDustGroups_; ++g) {
				if (alpha[g] == 0.0) {
					t_s_max = std::numeric_limits<amrex::Real>::max();
					break;
				}
				amrex::Real t_s = 1.0 / alpha[g];
				t_s_max = amrex::max(t_s_max, t_s);
			}

			amrex::Real gamma1 = 0; // NOLINT
			amrex::Real gamma2 = 0;
			amrex::Real beta1 = 0; // NOLINT
			amrex::Real beta2 = 0;
			amrex::Real b = 0;
			if (dt_lev < t_s_max) {
				gamma1 = 1.0;
				gamma2 = 0.0;
				beta1 = -0.5;
				beta2 = 2.0 / 3.0;
				b = 1.0;
			} else {
				gamma1 = 1.0;
				gamma2 = 1.0;
				beta1 = 1.0;
				beta2 = -1.0;
				b = 0.0;
			}

			amrex::GpuArray<amrex::Real, nDustGroups_> Lambda;
			amrex::GpuArray<amrex::Real, nDustGroups_> delta1;
			amrex::GpuArray<amrex::Real, nDustGroups_> delta2;
			for (int g = 0; g < nDustGroups_; ++g) {
				Lambda[g] = 1.0 / (1.0 + alpha[g] * dt * (gamma1 + gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)));
				delta1[g] = 1.0 / (1.0 + gamma1 * dt * alpha[g]);
				delta2[g] = 1.0 / (1.0 + gamma2 * dt * alpha[g]);
			}

			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::Real const v_g = vel_g_old[dir];

				amrex::GpuArray<amrex::Real, nDustGroups_> v_d;
				for (int g = 0; g < nDustGroups_; ++g) {
					v_d[g] = vel_d_old[g][dir];
				}

				amrex::GpuArray<amrex::Real, nDustGroups_ + 1> u;
				u[0] = rho_g * v_g;
				for (int g = 0; g < nDustGroups_; ++g) {
					u[1 + g] = rho_d[g] * v_d[g];
				}

				amrex::GpuArray<amrex::Real, nDustGroups_ + 1> k1;
				amrex::GpuArray<amrex::Real, nDustGroups_ + 1> k2;
				amrex::Real A1 = 0.0;
				amrex::Real A2 = 0.0;
				amrex::Real B1 = 0.0;
				amrex::Real B2 = 0.0;
				amrex::Real C1 = 0.0;
				amrex::Real C2 = 0.0;
				amrex::Real D1 = 1.0;
				amrex::Real D2 = 1.0;
				for (int g = 0; g < nDustGroups_; ++g) {
					A1 += alpha[g] * u[1 + g] * delta1[g] -
					      beta1 * dt * alpha[g] * alpha[g] * u[1 + g] * (1.0 + alpha[g] * dt * (gamma1 - beta2)) * delta1[g] * Lambda[g];

					A2 += alpha[g] * u[1 + g] * delta2[g] -
					      beta2 * dt * alpha[g] * alpha[g] * u[1 + g] * (1.0 + alpha[g] * dt * (gamma2 - beta1)) * delta2[g] * Lambda[g];

					B1 += alpha[g] * epsilon[g] * delta1[g] -
					      beta1 * dt * alpha[g] * alpha[g] * epsilon[g] * (1.0 + alpha[g] * dt * (gamma1 - beta2)) * delta1[g] * Lambda[g];

					B2 += alpha[g] * epsilon[g] * delta2[g] -
					      beta2 * dt * alpha[g] * alpha[g] * epsilon[g] * (1.0 + alpha[g] * dt * (gamma2 - beta1)) * delta2[g] * Lambda[g];

					C1 += alpha[g] * epsilon[g] * delta1[g] - dt * alpha[g] * alpha[g] * epsilon[g] *
										      (gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) * delta1[g] *
										      Lambda[g];

					C2 += alpha[g] * epsilon[g] * delta2[g] - dt * alpha[g] * alpha[g] * epsilon[g] *
										      (gamma1 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) * delta2[g] *
										      Lambda[g];

					D1 += gamma1 * dt * alpha[g] * epsilon[g] * delta1[g] -
					      beta1 * beta2 * dt * dt * alpha[g] * alpha[g] * epsilon[g] * delta1[g] * Lambda[g];

					D2 += gamma2 * dt * alpha[g] * epsilon[g] * delta2[g] -
					      beta1 * beta2 * dt * dt * alpha[g] * alpha[g] * epsilon[g] * delta2[g] * Lambda[g];
				}

				amrex::Real denominator = beta1 * beta2 * dt * dt * C1 * C2 - D1 * D2;

				k1[0] = (beta1 * dt * C1 * (A2 - B2 * u[0]) - D2 * (A1 - B1 * u[0])) / denominator;
				k2[0] = (beta2 * dt * C2 * (A1 - B1 * u[0]) - D1 * (A2 - B2 * u[0])) / denominator;

				for (int g = 0; g < nDustGroups_; ++g) {
					k1[1 + g] = alpha[g] * Lambda[g] *
						    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma2 - beta1)) +
						     k1[0] * epsilon[g] * dt * (gamma1 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) +
						     k2[0] * beta1 * epsilon[g] * dt);

					k2[1 + g] = alpha[g] * Lambda[g] *
						    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma1 - beta2)) +
						     k2[0] * epsilon[g] * dt * (gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) +
						     k1[0] * beta2 * epsilon[g] * dt);
				}

				vel_iter_new[0][dir] = vel_g_old[dir] + (rho_g > 0.0 ? dt * (b * k1[0] + (1.0 - b) * k2[0]) / rho_g : 0.0);
				for (int g = 0; g < nDustGroups_; ++g) {
					vel_iter_new[1 + g][dir] =
					    vel_d_old[g][dir] + (rho_d[g] > 0.0 ? dt * (b * k1[1 + g] + (1.0 - b) * k2[1 + g]) / rho_d[g] : 0.0);
				}
			}

			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> delta_mom_g{};
			amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, nDustGroups_> delta_mom_d;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				delta_mom_g[dir] = rho_g * (vel_iter_new[0][dir] - vel_g_old[dir]);
				for (int g = 0; g < nDustGroups_; ++g) {
					delta_mom_d[g][dir] = rho_d[g] * (vel_iter_new[1 + g][dir] - vel_d_old[g][dir]);
				}
			}

			amrex::Real delta_E_g1 = 0.0;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::Real const avg_v_g = 0.5 * (vel_g_old[dir] + vel_iter_new[0][dir]);
				delta_E_g1 += delta_mom_g[dir] * avg_v_g;
			}

			amrex::Real delta_E_g2 = delta_E_g1;
			for (int g = 0; g < nDustGroups_; ++g) {
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					amrex::Real const avg_v_d = 0.5 * (vel_d_old[g][dir] + vel_iter_new[1 + g][dir]);
					delta_E_g2 += delta_mom_d[g][dir] * avg_v_d;
				}
			}

			amrex::Real const delta_E = delta_E_g1 - omega * delta_E_g2;
			E_tot_iter_new = E_tot + delta_E;
			E_int_iter_new = E_int - omega * delta_E_g2;

			// check convergence conditions
			// calculate the reference speed
			amrex::Real max_speed_old = 0.0;
			{
				amrex::Real speed_sq = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq += vel_iter_old[0][dir] * vel_iter_old[0][dir];
				}
				max_speed_old = std::sqrt(speed_sq);
			}
			for (int g = 0; g < nDustGroups_; ++g) {
				amrex::Real speed_sq = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq += vel_iter_old[1 + g][dir] * vel_iter_old[1 + g][dir];
				}
				max_speed_old = amrex::max(max_speed_old, std::sqrt(speed_sq));
			}
			const amrex::Real abs_tolerance = tolerance * amrex::max(max_speed_old, 1.0e-12);
			// check convergence based on maximum speed change
			amrex::Real max_speed_change = 0.0;
			{
				amrex::Real speed_sq_old = 0.0;
				amrex::Real speed_sq_new = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq_old += vel_iter_old[0][dir] * vel_iter_old[0][dir];
					speed_sq_new += vel_iter_new[0][dir] * vel_iter_new[0][dir];
				}
				max_speed_change = std::abs(std::sqrt(speed_sq_new) - std::sqrt(speed_sq_old));
			}
			for (int g = 0; g < nDustGroups_; ++g) {
				amrex::Real speed_sq_old = 0.0;
				amrex::Real speed_sq_new = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq_old += vel_iter_old[1 + g][dir] * vel_iter_old[1 + g][dir];
					speed_sq_new += vel_iter_new[1 + g][dir] * vel_iter_new[1 + g][dir];
				}
				max_speed_change = amrex::max(max_speed_change, std::abs(std::sqrt(speed_sq_new) - std::sqrt(speed_sq_old)));
			}

			// if the maximum speed change is less than the absolute tolerance, exit the loop early
			if (max_speed_change <= abs_tolerance) {
				break;
			}

			vel_iter_old = vel_iter_new;
			E_tot_iter_old = E_tot_iter_new;
		}

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			consVar_cc[bx](i, j, k, x1Momentum_index + dir) = rho_g * vel_iter_new[0][dir];
			for (int g = 0; g < nDustGroups_; ++g) {
				consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars) = rho_d[g] * vel_iter_new[1 + g][dir];
			}
		}
		consVar_cc[bx](i, j, k, energy_index) = E_tot_iter_new;
		consVar_cc[bx](i, j, k, internalEnergy_index) = E_int_iter_new;
		amrex::Gpu::Atomic::Add(&p_iteration_counter[0], cell_iteration_count); // sum of iterations
		amrex::Gpu::Atomic::Add(&p_iteration_counter[1], 1);			// number of cells
		amrex::Gpu::Atomic::Max(&p_iteration_counter[2], cell_iteration_count); // max iterations in any cell
	});
	if (print_dust_counter_) {
		auto *h_iteration_counter = iteration_counter.copyToHost();
		long global_iteration_sum = h_iteration_counter[0]; // NOLINT(google-runtime-int)
		long global_cell_count = h_iteration_counter[1];    // NOLINT(google-runtime-int)
		int global_max_iterations = h_iteration_counter[2];

		amrex::ParallelDescriptor::ReduceLongSum(global_iteration_sum);
		amrex::ParallelDescriptor::ReduceLongSum(global_cell_count);
		amrex::ParallelDescriptor::ReduceIntMax(global_max_iterations);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			if (global_cell_count > 0) {
				const double avg_iterations = static_cast<double>(global_iteration_sum) / static_cast<double>(global_cell_count);
				amrex::Print() << "Dust drag Picard iteration statistics:\n";
				amrex::Print() << "  total cells updated: " << global_cell_count << "\n";
				amrex::Print() << "  average iterations per cell: " << avg_iterations << "\n";
				amrex::Print() << "  maximum iterations in any cell: " << global_max_iterations << "\n";
			}
		}
	}
}

template <typename problem_t>
void DustDrag<problem_t>::computeDustDragLorentz(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf,
						 amrex::Real dt, amrex::Real dust_omega1_, amrex::Real dust_omega2_, int enableIterDustStoptime_,
						 bool print_dust_counter_)
{
	amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0}); // [sum of iterations, number of cells, max iterations in any cell]
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
	amrex::Real const omega1 = dust_omega1_;
	amrex::Real const omega2 = dust_omega2_;
	auto const charge_to_mass_ratio = ComputeDustChargeToMassRatio();

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

		amrex::GpuArray<amrex::Real, nDustGroups_> rho_d;
		amrex::GpuArray<amrex::Real, nDustGroups_> epsilon;
		for (int g = 0; g < nDustGroups_; ++g) {
			rho_d[g] = consVar_cc[bx](i, j, k, dustDensity_index + g * numDustVars);
			epsilon[g] = (rho_g > 0.0) ? rho_d[g] / rho_g : 0.0;
		}

		Vec3 p_g_old = Vec3::Zero();
		amrex::GpuArray<Vec3, nDustGroups_> p_d_old;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			p_g_old[dir] = consVar_cc[bx](i, j, k, x1Momentum_index + dir);
			for (int g = 0; g < nDustGroups_; ++g) {
				p_d_old[g][dir] = consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars);
			}
		}

		const int max_iterations = (enableIterDustStoptime_ != 0) ? 20 : 1;
		const amrex::Real tolerance = 1.0e-6;
		int cell_iteration_count = 0;
		amrex::GpuArray<amrex::Real, nMassScalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(consVar_cc[bx], i, j, k);

		Vec3 p_g_iter_old = p_g_old;
		Vec3 p_g_iter_new = p_g_old;
		amrex::GpuArray<Vec3, nDustGroups_> p_d_iter_old = p_d_old;
		amrex::GpuArray<Vec3, nDustGroups_> p_d_iter_new = p_d_old;
		amrex::GpuArray<amrex::Real, nDustGroups_> alpha;
		amrex::GpuArray<amrex::Real, nDustGroups_> omega_L;
		amrex::GpuArray<Vec3, nDustGroups_> q_n;
		amrex::GpuArray<Vec3, nDustGroups_> q1;
		amrex::GpuArray<Vec3, nDustGroups_> q2;
		Vec3 k1_g = Vec3::Zero();
		Vec3 k2_g = Vec3::Zero();
		amrex::GpuArray<Vec3, nDustGroups_> k1_d;
		amrex::GpuArray<Vec3, nDustGroups_> k2_d;
		amrex::Real b = 1.0;
		amrex::Real gamma1 = 1.0;
		amrex::Real gamma2 = 0.0;
		amrex::Real beta1 = -0.5;
		amrex::Real beta2 = 2.0 / 3.0;
		amrex::Real E_tot_iter_old = E_tot;
		amrex::Real E_tot_iter_new = E_tot;
		amrex::Real E_int_iter_new = E_int;
		Vec3 const B_cc = BuildCellCenteredMagneticField(i, j, k, &cons_fc);
		amrex::Real const B_mag = std::sqrt(B_cc.dot(B_cc));
		amrex::Real const magnetic_energy = 0.5 * B_mag * B_mag;
		Vec3 b_hat = Vec3::Zero();
		if (B_mag > 0.0) {
			b_hat = (1.0 / B_mag) * B_cc;
		}
		Mat3 const B_cross = BuildCrossMatrix(b_hat);
		amrex::Real const dt_lev = 2.0 * dt;
		for (int g = 0; g < nDustGroups_; ++g) {
			omega_L[g] = charge_to_mass_ratio[g] * B_mag;
			q_n[g] = epsilon[g] * p_g_old - p_d_old[g];
		}

		for (int iteration = 0; iteration < max_iterations; ++iteration) {
			cell_iteration_count++;
			amrex::Real const cs =
			    ComputeSoundSpeedFromGasState(rho_g, p_g_iter_old.dot(p_g_iter_old), E_tot_iter_old, magnetic_energy, massScalars);

			amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag;
			for (int g = 0; g < nDustGroups_; ++g) {
				Vec3 v_g_iter_old = Vec3::Zero();
				Vec3 v_d_iter_old = Vec3::Zero();
				if (rho_g > 0.0) {
					v_g_iter_old = (1.0 / rho_g) * p_g_iter_old;
				}
				if (rho_d[g] > 0.0) {
					v_d_iter_old = (1.0 / rho_d[g]) * p_d_iter_old[g];
				}
				Vec3 const rel_vel = v_d_iter_old - v_g_iter_old;
				rel_vel_mag[g] = std::sqrt(rel_vel.dot(rel_vel));
			}
			alpha = ComputeReciprocalStoppingTime(rho_g, rho_d, rel_vel_mag, cs);

			amrex::Real lambda_max = 0.0;
			for (int g = 0; g < nDustGroups_; ++g) {
				amrex::Real const rate_mag = std::sqrt(alpha[g] * alpha[g] + omega_L[g] * omega_L[g]);
				amrex::Real lambda = std::numeric_limits<amrex::Real>::max();
				if (rate_mag > 0.0) {
					lambda = 1.0 / rate_mag;
				}
				lambda_max = amrex::max(lambda_max, lambda);
			}

			if (dt_lev < lambda_max) {
				gamma1 = 1.0;
				gamma2 = 0.0;
				beta1 = -0.5;
				beta2 = 2.0 / 3.0;
				b = 1.0;
			} else {
				gamma1 = 1.0;
				gamma2 = 1.0;
				beta1 = 1.0;
				beta2 = -1.0;
				b = 0.0;
			}

			amrex::GpuArray<DustStageAffineOperators, nDustGroups_> ops;
			for (int g = 0; g < nDustGroups_; ++g) {
				Mat3 const T = alpha[g] * Mat3::Identity() + omega_L[g] * B_cross;
				ops[g] = ComputeDustStageAffineOperators(T, epsilon[g], dt, gamma1, gamma2, beta1, beta2);
			}

			Mat3 G1X = Mat3::Zero();
			Mat3 G1Y = Mat3::Zero();
			Mat3 G2X = Mat3::Zero();
			Mat3 G2Y = Mat3::Zero();
			Vec3 C1 = Vec3::Zero();
			Vec3 C2 = Vec3::Zero();

			for (int g = 0; g < nDustGroups_; ++g) {
				C1 -= ops[g].W1 * q_n[g];
				C2 -= ops[g].W2 * q_n[g];
				G1X += ops[g].X1;
				G1Y += ops[g].Y1;
				G2X += ops[g].X2;
				G2Y += ops[g].Y2;
			}

			Mat6 S_g;
			Mat3 const I3 = Mat3::Identity();
			Mat3 const block11 = I3 + G1X;
			Mat3 const block12 = G1Y;
			Mat3 const block21 = G2X;
			Mat3 const block22 = I3 + G2Y;
			for (int row = 0; row < 3; ++row) {
				for (int col = 0; col < 3; ++col) {
					S_g(row, col) = block11(row, col);
					S_g(row, col + 3) = block12(row, col);
					S_g(row + 3, col) = block21(row, col);
					S_g(row + 3, col + 3) = block22(row, col);
				}
			}

			Vec6 rhs_g;
			for (int dir = 0; dir < 3; ++dir) {
				rhs_g(dir) = C1[dir];
				rhs_g(dir + 3) = C2[dir];
			}
			Vec6 sol_g;
			amrex::LUSolver<6, amrex::Real> const solver(S_g);
			solver(sol_g.begin(), rhs_g.begin());
			for (int dir = 0; dir < 3; ++dir) {
				k1_g[dir] = sol_g(dir);
				k2_g[dir] = sol_g(dir + 3);
			}

			for (int g = 0; g < nDustGroups_; ++g) {
				k1_d[g] = ops[g].W1 * q_n[g] + ops[g].X1 * k1_g + ops[g].Y1 * k2_g;
				k2_d[g] = ops[g].W2 * q_n[g] + ops[g].X2 * k1_g + ops[g].Y2 * k2_g;

				Vec3 const k_rel1 = epsilon[g] * k1_g - k1_d[g];
				Vec3 const k_rel2 = epsilon[g] * k2_g - k2_d[g];
				q1[g] = q_n[g] + dt * (gamma1 * k_rel1 + beta1 * k_rel2);
				q2[g] = q_n[g] + dt * (beta2 * k_rel1 + gamma2 * k_rel2);
			}

			p_g_iter_new = p_g_old + dt * (b * k1_g + (1.0 - b) * k2_g);
			for (int g = 0; g < nDustGroups_; ++g) {
				p_d_iter_new[g] = p_d_old[g] + dt * (b * k1_d[g] + (1.0 - b) * k2_d[g]);
			}

			Vec3 v_g_iter_old = Vec3::Zero();
			Vec3 v_g_iter_new = Vec3::Zero();
			if (rho_g > 0.0) {
				v_g_iter_old = (1.0 / rho_g) * p_g_iter_old;
				v_g_iter_new = (1.0 / rho_g) * p_g_iter_new;
			}

			amrex::Real max_speed_old = std::sqrt(v_g_iter_old.dot(v_g_iter_old));
			for (int g = 0; g < nDustGroups_; ++g) {
				Vec3 v_d_iter_old = Vec3::Zero();
				if (rho_d[g] > 0.0) {
					v_d_iter_old = (1.0 / rho_d[g]) * p_d_iter_old[g];
				}
				max_speed_old = amrex::max(max_speed_old, std::sqrt(v_d_iter_old.dot(v_d_iter_old)));
			}
			amrex::Real max_speed_change = std::abs(std::sqrt(v_g_iter_new.dot(v_g_iter_new)) - std::sqrt(v_g_iter_old.dot(v_g_iter_old)));
			for (int g = 0; g < nDustGroups_; ++g) {
				Vec3 v_d_iter_old = Vec3::Zero();
				Vec3 v_d_iter_new = Vec3::Zero();
				if (rho_d[g] > 0.0) {
					v_d_iter_old = (1.0 / rho_d[g]) * p_d_iter_old[g];
					v_d_iter_new = (1.0 / rho_d[g]) * p_d_iter_new[g];
				}
				max_speed_change = amrex::max(max_speed_change,
							      std::abs(std::sqrt(v_d_iter_new.dot(v_d_iter_new)) - std::sqrt(v_d_iter_old.dot(v_d_iter_old))));
			}
			amrex::Real const abs_tolerance = tolerance * amrex::max(max_speed_old, 1.0e-12);
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

			amrex::Real delta_E_heat_phy = 0.0;
			for (int g = 0; g < nDustGroups_; ++g) {
				if (rho_d[g] > 0.0) {
					delta_E_heat_phy += dt * alpha[g] / rho_d[g] * (b * q1[g].dot(q1[g]) + (1.0 - b) * q2[g].dot(q2[g]));
				}
			}

			amrex::Real const delta_E_heat_tot = -(delta_E_g_work + delta_E_d_work_sum);
			amrex::Real const delta_E_heat_num = delta_E_heat_tot - delta_E_heat_phy;
			E_tot_iter_new = E_tot + delta_E_g_work + omega1 * delta_E_heat_phy + omega2 * delta_E_heat_num;
			E_int_iter_new = E_int + omega1 * delta_E_heat_phy + omega2 * delta_E_heat_num;

			if (max_speed_change <= abs_tolerance) {
				break;
			}

			p_g_iter_old = p_g_iter_new;
			p_d_iter_old = p_d_iter_new;
			E_tot_iter_old = E_tot_iter_new;
		}

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			consVar_cc[bx](i, j, k, x1Momentum_index + dir) = p_g_iter_new[dir];
			for (int g = 0; g < nDustGroups_; ++g) {
				consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars) = p_d_iter_new[g][dir];
			}
		}
		consVar_cc[bx](i, j, k, energy_index) = E_tot_iter_new;
		consVar_cc[bx](i, j, k, internalEnergy_index) = E_int_iter_new;

		amrex::Gpu::Atomic::Add(&p_iteration_counter[0], cell_iteration_count); // sum of iterations
		amrex::Gpu::Atomic::Add(&p_iteration_counter[1], 1);			// number of cells
		amrex::Gpu::Atomic::Max(&p_iteration_counter[2], cell_iteration_count); // max iterations in any cell
	});
	if (print_dust_counter_) {
		auto *h_iteration_counter = iteration_counter.copyToHost();
		long global_iteration_sum = h_iteration_counter[0]; // NOLINT(google-runtime-int)
		long global_cell_count = h_iteration_counter[1];    // NOLINT(google-runtime-int)
		int global_max_iterations = h_iteration_counter[2];

		amrex::ParallelDescriptor::ReduceLongSum(global_iteration_sum);
		amrex::ParallelDescriptor::ReduceLongSum(global_cell_count);
		amrex::ParallelDescriptor::ReduceIntMax(global_max_iterations);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			if (global_cell_count > 0) {
				const double avg_iterations = static_cast<double>(global_iteration_sum) / static_cast<double>(global_cell_count);
				amrex::Print() << "Dust drag Lorentz Picard iteration statistics:\n";
				amrex::Print() << "  total cells updated: " << global_cell_count << "\n";
				amrex::Print() << "  average iterations per cell: " << avg_iterations << "\n";
				amrex::Print() << "  maximum iterations in any cell: " << global_max_iterations << "\n";
			}
		}
	}
}
