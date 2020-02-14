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
#include <cmath>

// library headers

// internal headers
#include "athena_arrays.hpp"

/// Class for a hyperbolic system of conservation laws (Cannot be instantiated,
/// must be subclassed.)
class HyperbolicSystem
{
      public:
	/// Computes timestep and advances system
	virtual void AdvanceTimestep() = 0;

	/// Add source terms to the conserved variables.
	///
	/// \param[in] source_terms A 4-dimensional array of (nvars, nx, ny, nz)
	///
	virtual void AddSourceTerms(AthenaArray<double> &source_terms) = 0;

	static auto minmod(double a, double b) -> double;

      protected:
	double CFL_number_ = 1.0;
	double dt_ = 0;
	double Lx_;
	double dx_;
	int nx_;
	int nghost_ = 2;

	HyperbolicSystem(int nx, double Lx, double CFL)
	    : nx_(nx), Lx_(Lx), dx_(Lx / static_cast<double>(nx)),
	      CFL_number_(CFL)
	{
	}

	virtual void FillGhostZones() = 0;
	virtual void ConservedToPrimitive() = 0;
	virtual void ComputeTimestep() = 0;
	virtual void ReconstructStates() = 0;
	virtual void ComputeFluxes() = 0;
	virtual void AddFluxes() = 0;
};

#endif // HYPERBOLIC_SYSTEM_HPP_
