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

#define  CONST_pc                (3.086e+18)
#define  CONST_mH                (1.6735575e-24)
#define  CONST_G                 (6.67259e-8) 
#define  CONST_Msun              (2.e33)
#define  CONST_amu               (1.6605e-24)
#define  CONST_kB                (1.3807e-15)
#define  UNIT_DENSITY            (1.0*CONST_mH)
#define  UNIT_LENGTH             (1.0e3*CONST_pc)
#define  UNIT_VELOCITY           1.0e7
#define  GLOB_TIME               (UNIT_LENGTH/UNIT_VELOCITY)
#define  UNIT_VOL                std::pow(UNIT_LENGTH,3.0)
#define  UNIT_MASS               (UNIT_DENSITY*UNIT_VOL)
#define  UNIT_ENERGY             (UNIT_MASS*std::pow(UNIT_VELOCITY,2.0))
#define  UNIT_PRESS              (UNIT_MASS/(std::pow(GLOB_TIME,2.0)*UNIT_LENGTH))
#define  KELVIN                  (UNIT_VELOCITY*UNIT_VELOCITY*CONST_amu/CONST_kB) 
#define  M_VIR                   (1.e12*CONST_Msun/UNIT_MASS)
#define  M_D                     (5.e10*CONST_Msun/UNIT_MASS)
#define  R_VIR                   258.0
#define  A1                      4.0
#define  B                       0.4
#define  C                       12.0
#define  D                       6.0
#define  R_S                     21.5
#define  RHO_D0                  3.0
#define  RHO_H0                  1.1e-3
#define  f                       0.95
#define  CSC                     0.2
#define  CSH                     2.0322
#define  TH                      (3.e6/KELVIN)
#define  TD                      (4.e4/KELVIN)


// internal headers
#include "hydro_system.hpp"
extern "C" {
    #include "interpolate.h"
}

// function definitions
auto testproblem_hydro_sedov() -> int;

#endif // TEST_HYDRO3D_BLAST_HPP_
