#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers
#include <cassert>
#include <cmath>
#include <valarray>

// library headers
#include "NamedType/named_type.hpp" // provides fluent::NamedType
#include "fmt/include/fmt/format.h"

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
class RadSystem : public HyperbolicSystem
{
      public:
	enum consVarIndex {
		radEnergy_index = 0,
		x1RadFlux_index = 1,
	};

	const double c_light = 1.0;   // use c=1 units
	const double c_hat = c_light; // for now

	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;

	RadSystem(NxType const &nx, LxType const &lx, CFLType const &cflNumber);

	void ConservedToPrimitive(AthenaArray<double> &cons,
				  std::pair<int, int> range) override;

	// setter functions:

	void set_cflNumber(double cflNumber);
	auto set_radEnergy(int i) -> double &;
	auto set_x1RadFlux(int i) -> double &;

	// accessor functions:

	auto radEnergy(int i) -> double;
	auto x1RadFlux(int i) -> double;
	auto ComputeRadEnergy() -> double;

      protected:
	AthenaArray<double> radEnergy_;
	AthenaArray<double> x1RadFlux_;

	void ComputeFluxes(std::pair<int, int> range) override;
	void ComputeTimestep() override;
};

#endif // RADIATION_SYSTEM_HPP_
