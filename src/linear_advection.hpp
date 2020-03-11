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
	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;
	using VxType = fluent::NamedType<double, struct VxParameter>;
	using NvarsType = fluent::NamedType<int, struct NvarsParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;
	static const VxType::argument Vx;
	static const NvarsType::argument Nvars;

	LinearAdvectionSystem(NxType const &nx, LxType const &lx,
			      VxType const &vx, CFLType const &cflNumber,
			      NvarsType const &nvars);

	void AddSourceTerms(AthenaArray<double> &U,
			    std::pair<int, int> range) override;
	auto ComputeMass() -> double;

	// accessor functions

	auto density(int i) -> double;	     // returns rvalue
	auto set_density(int i) -> double &; // returns lvalue

      protected:
	AthenaArray<double> density_; // shallow copy of consVars_(i,:)
	double advectionVx_;

	void ConservedToPrimitive(AthenaArray<double> &cons,
				  std::pair<int, int> range) override;
	void ComputeTimestep(double dt_max) override;
	void ComputeFluxes(std::pair<int, int> range) override;
	void AddFluxesSDC(AthenaArray<double> &U_new,
			  AthenaArray<double> &U0) override;
};

#endif // LINEAR_ADVECTION_HPP_
