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
	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;
	using NvarsType = fluent::NamedType<int, struct NvarsParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;
	static const NvarsType::argument Nvars;

	HydroSystem(NxType const &nx, LxType const &lx,
		    CFLType const &cflNumber, NvarsType const &nvars);

	void AddSourceTerms(AthenaArray<double> &source_terms) override;

	// setter functions:

	void set_cflNumber(double cflNumber);

	// accessor functions:

	auto nghost() -> int;
	auto nx() -> int;
	auto time() -> double;

	auto ComputeMass() -> double;

      protected:
	AthenaArray<double> density_;
	AthenaArray<double> x1Momentum_;
	AthenaArray<double> energy_;

	void ConservedToPrimitive(AthenaArray<double> &cons) override;
	void ComputeFluxes(std::pair<int, int> range) override;
	void ComputeTimestep() override;

	void FlattenShocks(AthenaArray<double> &q, std::pair<int, int> range);
};

#endif // HYDRO_SYSTEM_HPP_
