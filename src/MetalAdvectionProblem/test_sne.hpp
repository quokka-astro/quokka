#ifndef TEST_HYDRO3D_BLAST_HPP_ // NOLINT
#define TEST_HYDRO3D_BLAST_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.hpp
/// \brief Defines a test problem for a 3D explosion.
///

// external headers
#include <fstream>
#include <fmt/format.h>

constexpr double  Msun     = 2.e33;
constexpr double  Const_G  = 6.67e-8;
constexpr double  yr_to_s  = 3.154e7;
constexpr double  Myr      = 1.e6*yr_to_s;
constexpr double  pc       = 3.018e18;
constexpr double  kpc      = 1.e3 * pc;
constexpr double  ks_sigma_sfr = 6.e-5/yr_to_s/kpc/kpc;
constexpr double  hscale   = 150. * pc;
constexpr double  sqrtpi       = 1.772453;
constexpr double  probSN_prefac   = ks_sigma_sfr/(hscale*sqrtpi);
constexpr double  Mu       = 0.6;
constexpr double  kmps     = 1.e5; 
constexpr double  Const_mH = 1.67e-24;
constexpr double  kb       = 1.3807e-16;
constexpr amrex::Real T_floor = 10.0; // K

// internal headers
#include "hydro_system.hpp"
#include "interpolate.hpp"

#endif // TEST_HYDRO3D_BLAST_HPP_
