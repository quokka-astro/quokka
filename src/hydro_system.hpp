#ifndef HYDRO_SYSTEM_HPP_ // NOLINT
#define HYDRO_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.hpp
/// \brief Defines a class for solving the (1d) Euler equations.
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
class HydroSystem : public HyperbolicSystem
{
      public:
	AthenaArray<double> density_;

	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;

	HydroSystem(NxType const &nx, LxType const &lx,
		    CFLType const &cflNumber);

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

	void FillGhostZones() override;
	void ComputeTimestep() override;
	void ConservedToPrimitive() override;

	void ReconstructStatesConstant(std::pair<int, int> range);
	void ComputeFluxes(std::pair<int, int> range) override;
	void PredictHalfStep(std::pair<int, int> range);

	template <typename F>
	void ReconstructStatesPLM(F &&limiter, std::pair<int, int> range);
	void ReconstructStatesPPM(AthenaArray<double> &q,
				  std::pair<int, int> range);
	void FlattenShocks(AthenaArray<double> &q, std::pair<int, int> range);

	void AddFluxes() override;
};

#endif // HYDRO_SYSTEM_HPP_
