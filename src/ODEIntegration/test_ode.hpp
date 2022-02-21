#ifndef TEST_ODE_HPP_ // NOLINT
#define TEST_ODE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.hpp
/// \brief Defines a test problem for ODE integration.
///

// external headers
#include <memory>
#include <string>
#include <vector>

#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_TableData.H"

#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif
#include <fmt/format.h>

// internal headers
#include "GrackleDataReader.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "rk4.hpp"
#include "valarray.hpp"

// types

struct ODETest {};

constexpr double m_H = hydrogen_mass_cgs_;
constexpr double seconds_in_year = 3.154e7;

constexpr double Tgas0 = 6000.;     // K
constexpr double rho0 = 0.01 * m_H; // g cm^-3

struct cloudy_tables {
  std::unique_ptr<std::vector<double>> log_nH;
  std::unique_ptr<std::vector<double>> log_Tgas;

  std::unique_ptr<amrex::TableData<double, 2>> primCooling;
  std::unique_ptr<amrex::TableData<double, 2>> primHeating;
  std::unique_ptr<amrex::TableData<double, 2>> metalCooling;
  std::unique_ptr<amrex::TableData<double, 2>> metalHeating;
  std::unique_ptr<amrex::TableData<double, 2>> mean_mol_weight;
};

struct ODEUserData {
  amrex::Real rho = NAN;
  cloudy_tables *tables = nullptr;
};

// function definitions
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto cooling_function(Real rho, Real T)
    -> Real;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto
user_rhs(Real t, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int;

void readCloudyData();

#endif // TEST_ODE_HPP_
