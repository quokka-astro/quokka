//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file DustDrag.hpp
/// \brief Defines a class for integrating dust-gas drag force.
///

#include "AMReX_MultiFab.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/ArrayView_3d.hpp"

template <typename problem_t> class DustDrag
{
      public:
	static constexpr int nscalars_ = Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int nHydroScalars_ = Physics_NumVars::numHydroVars + nscalars_;
	static constexpr int numDustVars_ = Physics_NumVars::numDustVarsPerGroup; // number of dust variables for each dust group

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

	// compute reciprocal of dust stopping time
	AMREX_GPU_HOST_DEVICE static auto
	ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups> /*rho_d*/,
				      amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, Physics_Traits<problem_t>::nDustGroups + 1> /*vel*/,
				      double /*cs*/) -> amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups>;

	// compute dust-gas drag source terms and update conserved variables
	static void computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
				    amrex::Real dust_omega_, int enableInterDustStoptime_);
};

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeReciprocalStoppingTime(
    amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups> /*rho_d*/,
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, Physics_Traits<problem_t>::nDustGroups + 1> /*vel*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups>
{
	constexpr int N = Physics_Traits<problem_t>::nDustGroups;
	amrex::GpuArray<amrex::Real, N> alpha;
	alpha.fill(0.0);
	return alpha;
}

template <typename problem_t>
void DustDrag<problem_t>::computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt,
					  amrex::Real dust_omega_, int enableInterDustStoptime_)
{
	auto const &consVar_cc = consVar_cc_mf.arrays();
	auto const &cons_fc_x0 = consVar_fc_mf[0].const_arrays();
#if AMREX_SPACEDIM >= 2
	auto const &cons_fc_x1 = consVar_fc_mf[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
	auto const &cons_fc_x2 = consVar_fc_mf[2].const_arrays();
#endif

	constexpr int N = Physics_Traits<problem_t>::nDustGroups;
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

		amrex::GpuArray<amrex::Real, N> rho_d;
		for (int g = 0; g < N; ++g) {
			rho_d[g] = consVar_cc[bx](i, j, k, dustDensity_index + g * numDustVars);
		}

		amrex::GpuArray<amrex::Real, N> epsilon;
		for (int g = 0; g < N; ++g) {
			epsilon[g] = (rho_g > 0.0) ? rho_d[g] / rho_g : 0.0;
		}

		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_g_old{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N> vel_d_old;

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			int mom_g_idx = x1Momentum_index + dir;
			vel_g_old[dir] = (rho_g > 0.0) ? consVar_cc[bx](i, j, k, mom_g_idx) / rho_g : 0.0;

			for (int g = 0; g < N; ++g) {
				int mom_d_idx = x1DustMomentum_index + dir + g * numDustVars;
				vel_d_old[g][dir] = (rho_d[g] > 0.0) ? consVar_cc[bx](i, j, k, mom_d_idx) / rho_d[g] : 0.0;
			}
		}

		// set iteration parameters
		const int max_iterations = (enableInterDustStoptime_ != 0) ? 20 : 1;
		const amrex::Real tolerance = 1.0e-6;

		// initialize iteration intermediate variables
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N + 1> vel_inter_old;
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N + 1> vel_inter_new;

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			vel_inter_old[0][dir] = vel_g_old[dir];
			for (int g = 0; g < N; ++g) {
				vel_inter_old[1 + g][dir] = vel_d_old[g][dir];
			}
		}

		// Picard iteration loop
		for (int iteration = 0; iteration < max_iterations; ++iteration) {
			double cs = 0.0;
			if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
				cs = HydroSystem<problem_t>::cs_iso_;
			} else {
				cs = HydroSystem<problem_t>::ComputeSoundSpeed(consVar_cc[bx], i, j, k, &cons_fc);
			}
			amrex::GpuArray<amrex::Real, N> alpha = ComputeReciprocalStoppingTime(rho_g, rho_d, vel_inter_old, cs);

			amrex::Real t_s_max = 0.0;
			for (int g = 0; g < N; ++g) {
				if (alpha[g] == 0.0) {
					t_s_max = std::numeric_limits<amrex::Real>::max();
					break;
				}
				amrex::Real t_s = 1.0 / alpha[g];
				t_s_max = amrex::max(t_s_max, t_s);
			}

			amrex::Real const dt_lev = 2.0 * dt;
			amrex::Real gamma1 = 0; // NOLINT
			amrex::Real gamma2 = 0;
			amrex::Real beta1 = 0; // NOLINT
			amrex::Real beta2 = 0;
			amrex::Real b = 0;
			if (dt_lev < t_s_max) {
				// Δt < t_s^max
				gamma1 = 1.0;
				gamma2 = 0.0;
				beta1 = -0.5;
				beta2 = 2.0 / 3.0;
				b = 1.0;
			} else {
				// Δt > t_s^max
				gamma1 = 1.0;
				gamma2 = 1.0;
				beta1 = 1.0;
				beta2 = -1.0;
				b = 0.0;
			}

			amrex::GpuArray<amrex::Real, N> Lambda;
			amrex::GpuArray<amrex::Real, N> delta1;
			amrex::GpuArray<amrex::Real, N> delta2;
			for (int g = 0; g < N; ++g) {
				Lambda[g] = 1.0 / (1.0 + alpha[g] * dt * (gamma1 + gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)));
				delta1[g] = 1.0 / (1.0 + gamma1 * dt * alpha[g]);
				delta2[g] = 1.0 / (1.0 + gamma2 * dt * alpha[g]);
			}

			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::Real const v_g = vel_g_old[dir];

				amrex::GpuArray<amrex::Real, N> v_d;
				for (int g = 0; g < N; ++g) {
					v_d[g] = vel_d_old[g][dir];
				}

				amrex::GpuArray<amrex::Real, N + 1> u;
				u[0] = rho_g * v_g;
				for (int g = 0; g < N; ++g) {
					u[1 + g] = rho_d[g] * v_d[g];
				}

				amrex::GpuArray<amrex::Real, N + 1> k1 = {};
				amrex::GpuArray<amrex::Real, N + 1> k2 = {};
				amrex::Real A1 = 0.0;
				amrex::Real A2 = 0.0;
				amrex::Real B1 = 0.0;
				amrex::Real B2 = 0.0;
				amrex::Real C1 = 0.0;
				amrex::Real C2 = 0.0;
				amrex::Real D1 = 1.0;
				amrex::Real D2 = 1.0;
				for (int g = 0; g < N; ++g) {
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

				for (int g = 0; g < N; ++g) {
					k1[1 + g] = alpha[g] * Lambda[g] *
						    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma2 - beta1)) +
						     k1[0] * epsilon[g] * dt * (gamma1 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) +
						     k2[0] * beta1 * epsilon[g] * dt);

					k2[1 + g] = alpha[g] * Lambda[g] *
						    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma1 - beta2)) +
						     k2[0] * epsilon[g] * dt * (gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) +
						     k1[0] * beta2 * epsilon[g] * dt);
				}

				vel_inter_new[0][dir] = vel_g_old[dir] + dt * (b * k1[0] + (1.0 - b) * k2[0]) / rho_g;

				for (int g = 0; g < N; ++g) {
					vel_inter_new[1 + g][dir] = vel_d_old[g][dir] + dt * (b * k1[1 + g] + (1.0 - b) * k2[1 + g]) / rho_d[g];
				}
			}

			// update momenta and energy
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> delta_mom_g{};
			amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N> delta_mom_d;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				delta_mom_g[dir] = rho_g * (vel_inter_new[0][dir] - vel_g_old[dir]);
				consVar_cc[bx](i, j, k, x1Momentum_index + dir) = rho_g * vel_g_old[dir] + delta_mom_g[dir];
				for (int g = 0; g < N; ++g) {
					delta_mom_d[g][dir] = rho_d[g] * (vel_inter_new[1 + g][dir] - vel_d_old[g][dir]);
					consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars) =
					    rho_d[g] * vel_d_old[g][dir] + delta_mom_d[g][dir];
				}
			}
			amrex::Real delta_E_g1 = 0.0;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::Real const avg_v_g = 0.5 * (vel_g_old[dir] + vel_inter_new[0][dir]);
				delta_E_g1 += delta_mom_g[dir] * avg_v_g;
			}
			amrex::Real delta_E_g2 = delta_E_g1;
			for (int g = 0; g < N; ++g) {
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					amrex::Real avg_v_d = 0.5 * (vel_d_old[g][dir] + vel_inter_new[1 + g][dir]);
					delta_E_g2 += delta_mom_d[g][dir] * avg_v_d;
				}
			}
			amrex::Real const delta_E = delta_E_g1 - omega * delta_E_g2;
			consVar_cc[bx](i, j, k, energy_index) = E_tot + delta_E;
			consVar_cc[bx](i, j, k, internalEnergy_index) = E_int - omega * delta_E_g2;

			// check convergence conditions
			// calculate the reference speed
			amrex::Real max_speed_old = 0.0;
			{
				amrex::Real speed_sq = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq += vel_inter_old[0][dir] * vel_inter_old[0][dir];
				}
				max_speed_old = std::sqrt(speed_sq);
			}
			for (int g = 0; g < N; ++g) {
				amrex::Real speed_sq = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq += vel_inter_old[1 + g][dir] * vel_inter_old[1 + g][dir];
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
					speed_sq_old += vel_inter_old[0][dir] * vel_inter_old[0][dir];
					speed_sq_new += vel_inter_new[0][dir] * vel_inter_new[0][dir];
				}
				amrex::Real const speed_change = std::abs(std::sqrt(speed_sq_new) - std::sqrt(speed_sq_old));
				max_speed_change = amrex::max(max_speed_change, speed_change);
			}
			for (int g = 0; g < N; ++g) {
				amrex::Real speed_sq_old = 0.0;
				amrex::Real speed_sq_new = 0.0;
				for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
					speed_sq_old += vel_inter_old[1 + g][dir] * vel_inter_old[1 + g][dir];
					speed_sq_new += vel_inter_new[1 + g][dir] * vel_inter_new[1 + g][dir];
				}
				amrex::Real const speed_change = std::abs(std::sqrt(speed_sq_new) - std::sqrt(speed_sq_old));
				max_speed_change = amrex::max(max_speed_change, speed_change);
			}

			// if the maximum speed change is less than the absolute tolerance, exit the loop early
			if (max_speed_change <= abs_tolerance) {
				break;
			}

			vel_inter_old = vel_inter_new;
		}
	});
}