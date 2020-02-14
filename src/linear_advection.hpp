#ifndef LINEAR_ADVECTION_HPP_ // NOLINT
#define LINEAR_ADVECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.hpp
/// \brief Defines a class for solving a scalar linear advection equation.
///

// c++ headers
#include <cassert>
#include <cmath>

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
class LinearAdvectionSystem : public HyperbolicSystem
{
      public:
	AthenaArray<double> density_;

	LinearAdvectionSystem(int nx, double vx, double Lx);

	void AddSourceTerms(AthenaArray<double> &source_terms) override;
	void AdvanceTimestep() override; //< Advances system by one timestep
	void SetCFLNumber(double CFL_number);

	auto NumGhostZones() -> int;
	auto Nx() -> int;
	auto ComputeMass() -> double;

      protected:
	AthenaArray<double> density_xleft_;
	AthenaArray<double> density_xright_;
	AthenaArray<double> density_flux_;

	double advection_vx_;

	void FillGhostZones() override;
	void ConservedToPrimitive() override;
	void ComputeTimestep() override;
	void ReconstructStates() override;
	void ComputeFluxes() override;
	void AddFluxes() override;
};

#endif // LINEAR_ADVECTION_HPP_
