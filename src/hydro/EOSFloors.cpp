// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file EOSFloors.cpp
/// \brief Definitions of the EOS bounds declared in EOSFloors.hpp.

#include "hydro/EOSFloors.hpp"

#include "AMReX_Algorithm.H"
#include "AMReX_ParmParse.H"

namespace quokka::eos_floors
{

AMREX_GPU_MANAGED amrex::Real min_temp; // NOLINT
AMREX_GPU_MANAGED amrex::Real max_temp; // NOLINT
AMREX_GPU_MANAGED amrex::Real min_dens; // NOLINT
AMREX_GPU_MANAGED amrex::Real max_dens; // NOLINT
AMREX_GPU_MANAGED amrex::Real min_e;	// NOLINT
AMREX_GPU_MANAGED amrex::Real max_e;	// NOLINT
AMREX_GPU_MANAGED amrex::Real min_p;	// NOLINT
AMREX_GPU_MANAGED amrex::Real max_p;	// NOLINT

void ReadParameters()
{
	max_temp = 1.e200;
	max_dens = 1.e200;
	min_e = 1.e-200;
	max_e = 1.e200;
	min_p = 1.e-200;
	max_p = 1.e200;

	amrex::Real temp = 1.e-10;
	amrex::Real dens = 1.e-100;
	const amrex::ParmParse pp("eos");
	pp.query("min_temp", temp);
	pp.query("min_dens", dens);

	// as Microphysics' eos_init: a floor is never below 1e-200
	min_temp = amrex::max(1.e-200, temp);
	min_dens = amrex::max(1.e-200, dens);
}

} // namespace quokka::eos_floors
