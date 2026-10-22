#ifndef MAC_PROJECTOR_HPP_ // NOLINT
#define MAC_PROJECTOR_HPP_

//==============================================================================
// Adapted from Hydro::MacProjector in AMReX-Hydro
// (https://github.com/AMReX-Fluids/AMReX-Hydro, commit a153304acd78).
// Copyright (c) 2017, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory and the Alliance for Sustainable Energy,
// LLC., through National Renewable Energy Laboratory. All rights reserved.
// Redistributed under the BSD-3-Clause-LBNL license; see LICENSE.AMReX-Hydro.
//==============================================================================
/// \file MacProjector.hpp
/// \brief Projects a face-centered vector field onto its divergence-free part.
///
/// Only the non-EB, variable-coefficient path of the upstream class is kept.

#include <memory>

#include "AMReX_Array.H"
#include "AMReX_Geometry.H"
#include "AMReX_LO_BCTYPES.H"
#include "AMReX_MLABecLaplacian.H"
#include "AMReX_MLMG.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Vector.H"

namespace quokka
{

/// Projects a face-centered field u onto its divergence-free part by solving
/// -div(beta grad phi) = -div(u) and setting u <- u - beta grad phi.
class MacProjector
{
      public:
	/// \param a_umac face-centered field on each level; modified in place by Project()
	/// \param a_beta face-centered coefficients on each level
	/// \param a_geom geometry of each level
	/// \param a_lpinfo linear operator options
	MacProjector(const amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>> &a_umac,
		     const amrex::Vector<amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM>> &a_beta, const amrex::Vector<amrex::Geometry> &a_geom,
		     const amrex::LPInfo &a_lpinfo);

	/// Set the domain boundary conditions. Must be called before SetLevelBC() and Project().
	void SetDomainBC(const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &lobc, const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &hibc);

	/// Set the boundary values on level \p amrlev (nullptr means homogeneous).
	void SetLevelBC(int amrlev, const amrex::MultiFab *levelbcdata);

	/// Project umac, starting the solve from phi = 0.
	void Project(amrex::Real reltol, amrex::Real atol);

	void SetVerbose(int v) noexcept
	{
		verbose_ = v;
		mlmg_->setVerbose(verbose_);
	}

	/// Access the MLMG solver to change its options.
	auto GetMLMG() noexcept -> amrex::MLMG & { return *mlmg_; }

      private:
	void SetOptions();
	void AverageDownVelocity();

	std::unique_ptr<amrex::MLABecLaplacian> abeclap_;
	std::unique_ptr<amrex::MLMG> mlmg_;

	amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>> umac_;
	amrex::Vector<amrex::MultiFab> rhs_;
	amrex::Vector<amrex::MultiFab> phi_;
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes_;

	amrex::Vector<amrex::Geometry> geom_;

	int verbose_ = 0;
	int maxiter_ = 200;

	bool needsDomainBCs_ = true;
	amrex::Vector<int> needsLevelBCs_;
};

} // namespace quokka

#endif // MAC_PROJECTOR_HPP_
