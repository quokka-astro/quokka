#ifndef DUST_SYSTEM_HPP_ // NOLINT
#define DUST_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file dust_system.hpp
/// \brief Defines a class for solving the dust Euler equations.
///

#include "AMReX_MultiFab.H"
#include "dust/DustState.hpp"
#include "dust/dustRiemannSolver.hpp"
#include "physics_info.hpp"
#include "util/ArrayView.hpp"

template <typename problem_t> class DustSystem
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

	// compute dust fluxes for all dust groups
	template <FluxDir DIR>
	AMREX_GPU_DEVICE static void ComputeDustFluxes(quokka::Array4View<amrex::Real, DIR> &x1Flux, quokka::Array4View<const amrex::Real, DIR> &x1LeftState,
						       quokka::Array4View<const amrex::Real, DIR> &x1RightState, int i, int j, int k);
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

#endif // DUST_SYSTEM_HPP_
