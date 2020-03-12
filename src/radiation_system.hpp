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
#include <iostream>
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
		gasEnergy_index = 2,
		gasDensity_index = 3,
	};

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index = 1,
	};

	double c_light_ = 2.99792458e10;	 // cgs
	double c_hat_ = c_light_;		 // for now
	double radiation_constant_ = 7.5646e-15; // cgs

	const double mean_molecular_mass_ = (0.5) * 1.6726231e-24; // cgs
	const double boltzmann_constant_ = 1.380658e-16;	   // cgs
	const double gamma_ = (5. / 3.);

	double Erad_floor_ = 0.0;

	using NxType = fluent::NamedType<int, struct NxParameter>;
	using LxType = fluent::NamedType<double, struct LxParameter>;
	using CFLType = fluent::NamedType<double, struct CFLParameter>;

	static const NxType::argument Nx;
	static const LxType::argument Lx;
	static const CFLType::argument CFL;

	RadSystem(NxType const &nx, LxType const &lx, CFLType const &cflNumber);

	void FillGhostZones(AthenaArray<double> &cons) override;
	void ConservedToPrimitive(AthenaArray<double> &cons,
				  std::pair<int, int> range) override;
	void AddSourceTerms(AthenaArray<double> &cons,
			    std::pair<int, int> range) override;
	bool CheckStatesValid(AthenaArray<double> &cons,
			      std::pair<int, int> range) override;

	auto ComputeOpacity(double rho, double Temp) -> double;
	auto ComputeOpacityTempDerivative(double rho, double Temp) -> double;

	// setter functions:

	void set_cflNumber(double cflNumber);
	void set_lx(double lx);
	auto set_radEnergy(int i) -> double &;
	auto set_x1RadFlux(int i) -> double &;
	auto set_gasEnergy(int i) -> double &;
	auto set_staticGasDensity(int i) -> double &;
	auto set_radEnergySource(int i) -> double &;
	void set_c_light(double c_light);
	void set_radiation_constant(double arad);

	// accessor functions:

	auto radEnergy(int i) -> double;
	auto x1RadFlux(int i) -> double;
	auto gasEnergy(int i) -> double;
	auto staticGasDensity(int i) -> double;
	auto ComputeRadEnergy() -> double;
	auto ComputeGasEnergy() -> double;
	auto c_light() -> double;
	auto radiation_constant() -> double;

      protected:
	AthenaArray<double> radEnergy_;
	AthenaArray<double> x1RadFlux_;
	AthenaArray<double> gasEnergy_;
	AthenaArray<double> staticGasDensity_;
	AthenaArray<double> radEnergySource_;

	// virtual function overrides

	void PredictStep(const std::pair<int, int> range) override;
	void AddFluxesSDC(AthenaArray<double> &U_new,
			  AthenaArray<double> &U_0) override;
	void AddFluxesRK2(AthenaArray<double> &U0,
			  AthenaArray<double> &U1) override;
	void ComputeFluxes(std::pair<int, int> range) override;
	void ComputeTimestep(double dt_max) override;
};

#endif // RADIATION_SYSTEM_HPP_
