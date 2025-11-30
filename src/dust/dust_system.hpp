#ifndef DUST_SYSTEM_HPP_ // NOLINT
#define DUST_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file dust_system.hpp
/// \brief Defines a class for solving the dust Euler equations and integrating dust-gas drag force.
///

#include "AMReX_MultiFab.H"
#include "dust/DustState.hpp"
#include "dust/dustRiemannSolver.hpp"
#include "physics_info.hpp"
#include "util/ArrayView_3d.hpp"

template <typename problem_t> class DustSystem
{
      public:
	static constexpr int nscalars_ = Physics_Traits<problem_t>::numPassiveScalars;
	static constexpr int nHydroScalars_ = Physics_NumVars::numHydroVars + nscalars_;
	static constexpr int numDustVars_ = Physics_NumVars::numDustVarsPerGroup; // number of dust variables for each dust group

	enum consVarIndex {
		density_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1Momentum_index,
		x2Momentum_index,
		x3Momentum_index,
		energy_index,
		internalEnergy_index, // auxiliary internal energy (rho * e)
		scalar0_index	      // first passive scalar (only present if nscalars > 0!)
	};

	enum primVarIndex {
		primDensity_index = 0,
		x1Velocity_index,
		x2Velocity_index,
		x3Velocity_index,
		pressure_index,
		primEint_index,	   // auxiliary internal energy (rho * e)
		primScalar0_index, // first passive scalar (only present if nscalars > 0!)
	};

	enum dustVarIndex { dustDensity_index = Physics_Indices<problem_t>::dustFirstIndex, x1DustMomentum_index, x2DustMomentum_index, x3DustMomentum_index };

	static constexpr int primDustFirstIndex = primScalar0_index + nscalars_;
	enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index };

	// compute dust fluxes for all dust groups
	template <FluxDir DIR>
	AMREX_GPU_DEVICE static void ComputeDustFluxes(quokka::Array4View<amrex::Real, DIR> &x1Flux, quokka::Array4View<const amrex::Real, DIR> &x1LeftState,
						       quokka::Array4View<const amrex::Real, DIR> &x1RightState, int i, int j, int k);

	// compute dust-gas drag source terms and update conserved variables
	static void computeDustDrag(amrex::MultiFab &consVar_cc_mf, amrex::Real dt, amrex::Real dust_omega_,
				    amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups> dust_alpha_);
};

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE void DustSystem<problem_t>::ComputeDustFluxes(quokka::Array4View<amrex::Real, DIR> &x1Flux,
							       quokka::Array4View<const amrex::Real, DIR> &x1LeftState,
							       quokka::Array4View<const amrex::Real, DIR> &x1RightState, int i, int j, int k)
{
	for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
		// gather left- and right- density for dust
		const double dust_rho_L = x1LeftState(i, j, k, primDustDensity_index + numDustVars_ * g);
		const double dust_rho_R = x1RightState(i, j, k, primDustDensity_index + numDustVars_ * g);

		// assign normal component of velocity according to DIR
		int dust_velN_index = x1DustVelocity_index + numDustVars_ * g;
		int dust_velV_index = x2DustVelocity_index + numDustVars_ * g;
		int dust_velW_index = x3DustVelocity_index + numDustVars_ * g;

		if constexpr (DIR == FluxDir::X1) {
			dust_velN_index = x1DustVelocity_index + numDustVars_ * g;
			dust_velV_index = x2DustVelocity_index + numDustVars_ * g;
			dust_velW_index = x3DustVelocity_index + numDustVars_ * g;
		} else if constexpr (DIR == FluxDir::X2) {
#if (AMREX_SPACEDIM == 2)
			dust_velN_index = x2DustVelocity_index + numDustVars_ * g;
			dust_velV_index = x1DustVelocity_index + numDustVars_ * g;
			dust_velW_index = x3DustVelocity_index + numDustVars_ * g; // unchanged in 2D
#endif
#if (AMREX_SPACEDIM == 3)
			dust_velN_index = x2DustVelocity_index + numDustVars_ * g;
			dust_velV_index = x3DustVelocity_index + numDustVars_ * g;
			dust_velW_index = x1DustVelocity_index + numDustVars_ * g;
#endif
		} else if constexpr (DIR == FluxDir::X3) {
			dust_velN_index = x3DustVelocity_index + numDustVars_ * g;
			dust_velV_index = x1DustVelocity_index + numDustVars_ * g;
			dust_velW_index = x2DustVelocity_index + numDustVars_ * g;
		}

		quokka::DustState dust_sL{};
		dust_sL.rho = dust_rho_L;
		dust_sL.u = x1LeftState(i, j, k, dust_velN_index);
		dust_sL.v = x1LeftState(i, j, k, dust_velV_index);
		dust_sL.w = x1LeftState(i, j, k, dust_velW_index);

		quokka::DustState dust_sR{};
		dust_sR.rho = dust_rho_R;
		dust_sR.u = x1RightState(i, j, k, dust_velN_index);
		dust_sR.v = x1RightState(i, j, k, dust_velV_index);
		dust_sR.w = x1RightState(i, j, k, dust_velW_index);

		// solve the dust Riemann problem in canonical form (i.e., where the x-dir is the normal direction)
		auto dust_F_canonical = quokka::Riemann::dustRiemannSolver<problem_t, numDustVars_>(dust_sL, dust_sR);

		quokka::valarray<double, numDustVars_> dust_F = dust_F_canonical;

		// permute dust momentum components according to flux direction DIR
		dust_F[dust_velN_index - numDustVars_ * g - nHydroScalars_] = dust_F_canonical[x1DustMomentum_index - nHydroScalars_];
		dust_F[dust_velV_index - numDustVars_ * g - nHydroScalars_] = dust_F_canonical[x2DustMomentum_index - nHydroScalars_];
		dust_F[dust_velW_index - numDustVars_ * g - nHydroScalars_] = dust_F_canonical[x3DustMomentum_index - nHydroScalars_];

		// copy all dust flux components to the flux array
		for (int nc = 0; nc < numDustVars_; ++nc) {
			AMREX_ASSERT(!std::isnan(dust_F[nc])); // check dust flux is valid
			x1Flux(i, j, k, Physics_Indices<problem_t>::dustFirstIndex + nc + numDustVars_ * g) = dust_F[nc];
		}
	}
}

template <typename problem_t>
void DustSystem<problem_t>::computeDustDrag(amrex::MultiFab &consVar_cc_mf, amrex::Real dt, amrex::Real dust_omega_,
					    amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups> dust_alpha_)
{
	auto const &consVar_cc = consVar_cc_mf.arrays();
	constexpr int N = Physics_Traits<problem_t>::nDustGroups;
	int const numDustVars = Physics_NumVars::numDustVarsPerGroup;
	amrex::Real const omega = dust_omega_;

	amrex::GpuArray<amrex::Real, N> alpha = dust_alpha_;

	// NOLINTNEXTLINE(modernize-use-trailing-return-type)
	amrex::ParallelFor(consVar_cc_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		amrex::Real rho_g = consVar_cc[bx](i, j, k, density_index);

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

			amrex::GpuArray<amrex::Real, N + 1> k1 = {};
			amrex::GpuArray<amrex::Real, N + 1> k2 = {};
			amrex::GpuArray<amrex::Real, N> Lambda;
			amrex::GpuArray<amrex::Real, N> delta1;
			amrex::GpuArray<amrex::Real, N> delta2;
			for (int g = 0; g < N; ++g) {
				Lambda[g] = 1.0 / (1.0 + alpha[g] * dt * (gamma1 + gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)));
				delta1[g] = 1.0 / (1.0 + gamma1 * dt * alpha[g]);
				delta2[g] = 1.0 / (1.0 + gamma2 * dt * alpha[g]);
			}

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
				k1[1 + g] =
				    alpha[g] * Lambda[g] *
				    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma2 - beta1)) +
				     k1[0] * epsilon[g] * dt * (gamma1 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) + k2[0] * beta1 * epsilon[g] * dt);

				k2[1 + g] =
				    alpha[g] * Lambda[g] *
				    ((u[0] * epsilon[g] - u[1 + g]) * (1.0 + alpha[g] * dt * (gamma1 - beta2)) +
				     k2[0] * epsilon[g] * dt * (gamma2 + alpha[g] * dt * (gamma1 * gamma2 - beta1 * beta2)) + k1[0] * beta2 * epsilon[g] * dt);
			}

			consVar_cc[bx](i, j, k, x1Momentum_index + dir) += dt * (b * k1[0] + (1.0 - b) * k2[0]);

			for (int g = 0; g < N; ++g) {
				consVar_cc[bx](i, j, k, x1DustMomentum_index + dir + g * numDustVars) += dt * (b * k1[1 + g] + (1.0 - b) * k2[1 + g]);
			}
		}

		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_g_new{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N> vel_d_new;
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> delta_mom_g{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, N> delta_mom_d;

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			int mom_g_idx = x1Momentum_index + dir;
			vel_g_new[dir] = (rho_g > 0.0) ? consVar_cc[bx](i, j, k, mom_g_idx) / rho_g : 0.0;
			delta_mom_g[dir] = consVar_cc[bx](i, j, k, mom_g_idx) - rho_g * vel_g_old[dir];

			for (int g = 0; g < N; ++g) {
				int mom_d_idx = x1DustMomentum_index + dir + g * numDustVars;
				vel_d_new[g][dir] = (rho_d[g] > 0.0) ? consVar_cc[bx](i, j, k, mom_d_idx) / rho_d[g] : 0.0;
				delta_mom_d[g][dir] = consVar_cc[bx](i, j, k, mom_d_idx) - rho_d[g] * vel_d_old[g][dir];
			}
		}

		amrex::Real delta_E_g1 = 0.0;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			amrex::Real const avg_v_g = 0.5 * (vel_g_old[dir] + vel_g_new[dir]);
			delta_E_g1 += delta_mom_g[dir] * avg_v_g;
		}

		amrex::Real delta_E_g2 = delta_E_g1;
		for (int g = 0; g < N; ++g) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::Real avg_v_d = 0.5 * (vel_d_old[g][dir] + vel_d_new[g][dir]);
				delta_E_g2 += delta_mom_d[g][dir] * avg_v_d;
			}
		}

		amrex::Real const delta_E = delta_E_g1 - omega * delta_E_g2;

		consVar_cc[bx](i, j, k, energy_index) += delta_E;
		consVar_cc[bx](i, j, k, internalEnergy_index) += -omega * delta_E_g2;
	});
}

#endif // DUST_SYSTEM_HPP_