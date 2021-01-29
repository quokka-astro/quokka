#ifndef SIMULATION_HPP_ // NOLINT
#define SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file simulation.cpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation.

// c++ headers
#include "AMReX_DistributionMapping.H"
#include <cassert>
#include <cmath>

// library headers
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

// internal headers

// Simulation class should be initialized only once per program (i.e., is a singleton)
class SingleLevelSimulation
{
      public:
	int n_cell{128};
	int max_grid_size{32};
	int nsteps{10};

	amrex::BoxArray simBoxArray;
	amrex::Geometry simGeometry;
	amrex::IntVect dom_lo{AMREX_D_DECL(0, 0, 0)};
	amrex::IntVect dom_hi{AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1)};
	amrex::Box domain{dom_lo, dom_hi};

	// This defines the physical box, [-1,1] in each direction.
	amrex::RealBox real_box{
	    {AMREX_D_DECL(-amrex::Real(1.0), -amrex::Real(1.0), -amrex::Real(1.0))},
	    {AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0))}};

	// periodic in all directions
	amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};

	// How Boxes are distributed among MPI processes
	amrex::DistributionMapping simDistributionMapping;

	// we allocate two phi multifabs; one will store the old state, the other the new.
	amrex::MultiFab phi_old;
	amrex::MultiFab phi_new;

	// Nghost = number of ghost cells for each array
	int Nghost = 1;
	// Ncomp = number of components for each array
	int Ncomp = 1;
    // dx = cell size
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};

	SingleLevelSimulation()
	{
		// ParmParse is way of reading inputs from the inputs file
		amrex::ParmParse pp;

		// We need to get n_cell from the inputs file - this is the number of cells on each
		// side of a square (or cubic) domain.
		pp.get("n_cell", n_cell);

		// The domain is broken into boxes of size max_grid_size
		pp.get("max_grid_size", max_grid_size);

		// Default nsteps to 10, allow us to set it to something else in the inputs file
		pp.query("nsteps", nsteps);

		// Initialize the boxarray "ba" from the single box "bx"
		simBoxArray.define(domain);
		// Break up boxarray "ba" into chunks no larger than "max_grid_size" along a
		// direction
		simBoxArray.maxSize(max_grid_size);

		// This defines a Geometry object
		simGeometry.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
		dx = simGeometry.CellSizeArray();

		// initial DistributionMapping with boxarray
		simDistributionMapping = amrex::DistributionMapping(simBoxArray);
	}
};

#endif // SIMULATION_HPP_