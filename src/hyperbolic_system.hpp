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
	/// Computes timestep and advances system
	void AdvanceTimestep();

	/// Add source terms to the conserved variables.
	///
	/// \param[in] source_terms A 4-dimensional array of (nvars, nx, ny, nz)
	///
	virtual void AddSourceTerms(AthenaArray<double> &source_terms) = 0;

	// setter functions:

	void set_cflNumber(double cflNumber);

	// accessor functions:

	auto nvars() -> int;
	auto nghost() -> int;
	auto nx() -> int;
	auto time() -> double;

	// inline functions:

	__attribute__((always_inline)) inline static auto minmod(double a,
								 double b)
	    -> double
	{
		auto result = 0.0;

		if ((sgn(a) == sgn(b)) && (a != b) && (a != 0.0) &&
		    (b != 0.0)) {
			if (std::abs(a) < std::abs(b)) {
				result = a;
			} else {
				result = b;
			}
		}

		return result;
	}

      protected:
	AthenaArray<double> consVar_;
	AthenaArray<double> primVar_;
	AthenaArray<double> consVarPredictStep_;
	AthenaArray<double> x1LeftState_;
	AthenaArray<double> x1RightState_;
	AthenaArray<double> x1Flux_;

	double cflNumber_ = 1.0;
	double dt_ = 0;
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
		x1LeftState_.NewAthenaArray(nvars_, dim1_);
		x1RightState_.NewAthenaArray(nvars_, dim1_);
		x1Flux_.NewAthenaArray(nvars_, dim1_);
	}

	void FillGhostZones();
	void AddFluxes();
	void ReconstructStatesConstant(std::pair<int, int> range);
	template <typename F>
	void ReconstructStatesPLM(F &&limiter, std::pair<int, int> range);
	void ReconstructStatesPPM(AthenaArray<double> &q,
				  std::pair<int, int> range);
	void PredictHalfStep(std::pair<int, int> range);

	virtual void ConservedToPrimitive(AthenaArray<double> &cons) = 0;
	virtual void ComputeTimestep() = 0;
	virtual void ComputeFluxes(std::pair<int, int> range) = 0;
};

#endif // HYPERBOLIC_SYSTEM_HPP_
