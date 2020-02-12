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

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
class LinearAdvectionSystem : public HyperbolicSystem
{
      public:
	AthenaArray<double> density;

	/// Computes timestep and advances system
	explicit LinearAdvectionSystem(int nx, double vx, double Lx);
	void AddSourceTerms(AthenaArray<double> *source_terms) override;
	void AdvanceTimestep() override;
	auto NumGhostZones() -> int;
	auto Nx() -> int;
	void FillGhostZones() override;

      protected:
	AthenaArray<double> interface_density;
	AthenaArray<double> flux_density;

	double advection_vx;
	double Lx;
	double dx;
	double dt;
	double CFL_number = 1.0;
	int nx;
	int nghost = 2;

	void ConservedToPrimitive() override;
	void ComputeTimestep() override;
	void ReconstructStates() override;
	void ComputeFluxes() override;
	void AddFluxes() override;
};

#endif // LINEAR_ADVECTION_HPP_
