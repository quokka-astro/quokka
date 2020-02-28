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
#include <valarray>

// library headers
#include "NamedType/named_type.hpp" // provides fluent::NamedType
#include "fmt/include/fmt/format.h"

// internal headers
#include "athena_arrays.hpp"
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
class HydroSystem : public HyperbolicSystem
{
      public:
	enum consVarIndex {
		density_index = 0,
		x1Momentum_index = 1,
		energy_index = 2
	};

	enum primVarIndex {
		primDensity_index = 0,
		x1Velocity_index = 1,
		pressure_index = 2
	};

	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;
	using GammaType = fluent::NamedType<double, struct GammaParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;
	static const GammaType::argument Gamma;

	HydroSystem(NxType const &nx, LxType const &lx,
		    CFLType const &cflNumber, GammaType const &gamma);

	void AddSourceTerms(AthenaArray<double> &source_terms);
	void ConservedToPrimitive(AthenaArray<double> &cons,
				  std::pair<int, int> range) override;

	// setter functions:

	void set_cflNumber(double cflNumber);
	auto set_density(int i) -> double &;
	auto set_x1Momentum(int i) -> double &;
	auto set_energy(int i) -> double &;

	// accessor functions:

	auto density(int i) -> double;
	auto x1Momentum(int i) -> double;
	auto energy(int i) -> double;

	auto primDensity(int i) -> double;
	auto x1Velocity(int i) -> double;
	auto pressure(int i) -> double;

	auto ComputeMass() -> double;

      protected:
	AthenaArray<double> density_;
	AthenaArray<double> x1Momentum_;
	AthenaArray<double> energy_;

	AthenaArray<double> primDensity_;
	AthenaArray<double> x1Velocity_;
	AthenaArray<double> pressure_;

	double gamma_;

	void ComputeFluxes(std::pair<int, int> range) override;
	void ComputeTimestep() override;

	void FlattenShocks(AthenaArray<double> &q, std::pair<int, int> range);
};

#endif // HYDRO_SYSTEM_HPP_
