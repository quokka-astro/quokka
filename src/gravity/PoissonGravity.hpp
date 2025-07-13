#ifndef POISSONGRAVITY_HPP_
#define POISSONGRAVITY_HPP_
//==============================================================================
// Quokka - a radiation hydrodynamics code for AMR
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file PoissonGravity.hpp
/// \brief Implements the PoissonGravity class for self-gravity with AMR subcycling
/// using Approach 1 (Enzo approach) - one Poisson solve per level advance.

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include "physics_info.hpp"

template <typename problem_t> class PoissonGravity
{
      public:
	// Constructor
	explicit PoissonGravity(const amrex::Vector<amrex::Geometry> &geom, const amrex::Vector<amrex::BoxArray> &grids,
				const amrex::Vector<amrex::DistributionMapping> &dmap, int max_level);

	// Destructor
	~PoissonGravity() = default;

	// Delete copy constructor and assignment operator
	PoissonGravity(const PoissonGravity &) = delete;
	auto operator=(const PoissonGravity &) -> PoissonGravity & = delete;

	// Move constructor and assignment operator
	PoissonGravity(PoissonGravity &&) = default;
	auto operator=(PoissonGravity &&) -> PoissonGravity & = default;

	// Main interface functions
	void solve_for_phi(int level, const amrex::MultiFab &density, amrex::MultiFab &gravitational_potential, amrex::Real time = 0.0);

	void compute_gravitational_acceleration(int level, const amrex::MultiFab &gravitational_potential,
						amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &gravitational_acceleration);

	void apply_operator_split_gravity_update(int level, amrex::MultiFab &state,
						 const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &gravitational_acceleration, amrex::Real dt);

	// Boundary condition handling
	void set_dirichlet_boundary_conditions(int level, amrex::MultiFab &gravitational_potential, const amrex::MultiFab &coarse_potential,
					       amrex::Real time = 0.0);

	// Configuration and parameters
	void read_parameters();
	void set_gravitational_constant(amrex::Real G_const);
	auto get_gravitational_constant() const -> amrex::Real;

      private:
	// AMR grid information
	const amrex::Vector<amrex::Geometry> &geom_;
	const amrex::Vector<amrex::BoxArray> &grids_;
	const amrex::Vector<amrex::DistributionMapping> &dmap_;
	int max_level_;

	// Physics parameters
	amrex::Real gravitational_constant_ = Physics_Traits<problem_t>::gravitational_constant;
	amrex::Real tolerance_ = 1.0e-12;
	int max_iterations_ = 200;
	int verbose_ = 0;

	// Multigrid solver objects (one per level for level-by-level solves)
	amrex::Vector<std::unique_ptr<amrex::MLPoisson>> mlpoisson_;
	amrex::Vector<std::unique_ptr<amrex::MLMG>> mlmg_;

	// Internal helper functions
	void setup_poisson_solver(int level);
	void setup_boundary_conditions(int level);
	auto compute_rhs_from_density(int level, const amrex::MultiFab &density) -> amrex::MultiFab;
	void interpolate_boundary_conditions_from_coarse_level(int level, amrex::MultiFab &fine_potential, const amrex::MultiFab &coarse_potential);
};

// Include template implementation
#include "PoissonGravity_impl.hpp"

#endif // POISSONGRAVITY_HPP_