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
	AthenaArray<double> density;

	explicit LinearAdvectionSystem(int nx, double vx, double Lx);

	void AddSourceTerms(AthenaArray<double> &source_terms) override;
	void
	AdvanceTimestep() override; //< Computes timestep and advances system
	void FillGhostZones() override;

	auto NumGhostZones() -> int;
	auto Nx() -> int;
	auto ComputeMass() -> double;

      protected:
	AthenaArray<double> density_xleft;
	AthenaArray<double> density_xright;
	AthenaArray<double> density_flux_fromleft;
	AthenaArray<double> density_flux_fromright;

	double advection_vx;
	double Lx;
	double dx;
	int nx;
	int nghost = 2;

	void ConservedToPrimitive() override;
	void ComputeTimestep() override;
	void ReconstructStates() override;
	void ComputeFluxes() override;
	void AddFluxes() override;
};

#endif // LINEAR_ADVECTION_HPP_
