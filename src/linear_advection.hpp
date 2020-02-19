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

// library headers
#include "NamedType/named_type.hpp" // provides fluent::NamedType

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
class LinearAdvectionSystem : public HyperbolicSystem
{
      public:
	AthenaArray<double> density_;

	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;
	using VxType = fluent::NamedType<double, struct VxParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;
	static const VxType::argument Vx;

	LinearAdvectionSystem(NxType const &nx, LxType const &lx,
			      VxType const &vx, CFLType const &cflNumber);

	void AddSourceTerms(AthenaArray<double> &source_terms) override;
	void AdvanceTimestep() override; //< Advances system by one timestep

	// setter functions:

	void set_cflNumber(double cflNumber);

	// accessor functions:

	auto nghost() -> int;
	auto nx() -> int;
	auto time() -> double;

	auto ComputeMass() -> double;

      protected:
	AthenaArray<double> densityXLeft_;
	AthenaArray<double> densityXRight_;
	AthenaArray<double> densityXFlux_;
	AthenaArray<double> densityPrediction_;

	double advectionVx_;

	void FillGhostZones() override;
	void ComputeTimestep() override;
	void ConservedToPrimitive() override;

	void ReconstructStatesConstant(int lo, int hi);
	void ComputeFluxes(int lo, int hi) override;
	void PredictHalfStep(int lo, int hi);

	template <typename F>
	void ReconstructStatesPLM(F &&limiter, int lo, int hi);
	void ReconstructStatesPPM(AthenaArray<double> &q, int lo, int hi);

	void AddFluxes() override;
};

#endif // LINEAR_ADVECTION_HPP_
