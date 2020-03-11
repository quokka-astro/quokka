#ifndef HYPERBOLIC_SYSTEM_HPP_ // NOLINT
#define HYPERBOLIC_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.hpp
/// \brief Defines classes and functions for use with hyperbolic systems of
/// conservation laws.
///
/// This file provides classes, data structures and functions for hyperbolic
/// systems of conservation laws.
///

// c++ headers
#include <cassert>
#include <cmath>

// library headers

// internal headers
#include "athena_arrays.hpp"

/// Provide type-safe global sign ('sgn') function.
template <typename T> auto sgn(T val) -> int
{
	return (T(0) < val) - (val < T(0));
}

/// Class for a hyperbolic system of conservation laws (Cannot be instantiated,
/// must be subclassed.)
class HyperbolicSystem
{
      public:
	AthenaArray<double> consVar_;

	/// Computes timestep and advances system
	void AdvanceTimestep();
	void AdvanceTimestep(double dt_max);
	void AdvanceTimestepRK2(double dt_max);
	void AdvanceTimestepSDC(double dt_max);

	// setter functions:

	void set_cflNumber(double cflNumber);

	// accessor functions:

	auto nvars() -> int;
	auto nghost() -> int;
	auto nx() -> int;
	auto time() -> double;
	auto dt() -> double;

	// inline functions:

	__attribute__((always_inline)) inline static auto minmod(double a,
								 double b)
	    -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) *
		       std::min(std::abs(a), std::abs(b));
	}

	__attribute__((always_inline)) inline static auto MC(double a, double b)
	    -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) *
		       std::min(0.5 * std::abs(a + b),
				std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	virtual void FillGhostZones(AthenaArray<double> &cons);
	virtual void ConservedToPrimitive(AthenaArray<double> &cons,
					  std::pair<int, int> range) = 0;
	virtual void AddSourceTerms(AthenaArray<double> &U,
				    std::pair<int, int> range) = 0;
	virtual bool CheckStatesValid(AthenaArray<double> &cons,
				      const std::pair<int, int> range);

      protected:
	AthenaArray<double> primVar_;
	AthenaArray<double> consVarPredictStep_;
	AthenaArray<double> consVarPredictStepPrev_;
	AthenaArray<double> x1LeftState_;
	AthenaArray<double> x1RightState_;
	AthenaArray<double> x1Flux_;
	AthenaArray<double> x1FluxDiffusive_;

	double cflNumber_ = 1.0;
	double dt_ = 0;
	const double dtExpandFactor_ = 1.2;
	double dtPrev_ = std::numeric_limits<double>::max();
	double time_ = 0.;
	double lx_;
	double dx_;
	int nx_;
	int dim1_;
	int nvars_;
	const int nghost_ = 4; // 4 ghost cells required for PPM

	HyperbolicSystem(int nx, double lx, double cflNumber, int nvars)
	    : nx_(nx), lx_(lx), dx_(lx / static_cast<double>(nx)),
	      cflNumber_(cflNumber), nvars_(nvars)
	{
		assert(lx_ > 0.0);				   // NOLINT
		assert(nx_ > 2);				   // NOLINT
		assert(nghost_ > 1);				   // NOLINT
		assert((cflNumber_ > 0.0) && (cflNumber_ <= 1.0)); // NOLINT

		dim1_ = nx_ + 2 * nghost_;

		consVar_.NewAthenaArray(nvars_, dim1_);
		primVar_.NewAthenaArray(nvars_, dim1_);
		consVarPredictStep_.NewAthenaArray(nvars_, dim1_);
		consVarPredictStepPrev_.NewAthenaArray(nvars_, dim1_);
		x1LeftState_.NewAthenaArray(nvars_, dim1_);
		x1RightState_.NewAthenaArray(nvars_, dim1_);
		x1Flux_.NewAthenaArray(nvars_, dim1_);
		x1FluxDiffusive_.NewAthenaArray(nvars_, dim1_);
	}

	virtual void AddFluxesRK2(AthenaArray<double> &U0,
				  AthenaArray<double> &U1);
	virtual void AddFluxesSDC(AthenaArray<double> &U_new,
				  AthenaArray<double> &U_0) = 0;

	void ReconstructStatesConstant(AthenaArray<double> &q,
				       std::pair<int, int> range);
	template <typename F>
	void ReconstructStatesPLM(AthenaArray<double> &q,
				  std::pair<int, int> range, F &&limiter);
	void ReconstructStatesPPM(AthenaArray<double> &q,
				  std::pair<int, int> range);

	virtual void PredictStep(std::pair<int, int> range);
	void ComputeTimestep();

	virtual void ComputeTimestep(double dt_max) = 0;
	virtual void ComputeFluxes(std::pair<int, int> range) = 0;
};

#endif // HYPERBOLIC_SYSTEM_HPP_
