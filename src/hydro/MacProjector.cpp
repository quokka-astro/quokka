//==============================================================================
// Adapted from Hydro::MacProjector in AMReX-Hydro
// (https://github.com/AMReX-Fluids/AMReX-Hydro, commit a153304acd78).
// Copyright (c) 2017, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory and the Alliance for Sustainable Energy,
// LLC., through National Renewable Energy Laboratory. All rights reserved.
// Redistributed under the BSD-3-Clause-LBNL license; see LICENSE.AMReX-Hydro.
//==============================================================================
/// \file MacProjector.cpp
/// \brief Implements the MAC projection of a face-centered vector field.

#include <string>

#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"

#include "hydro/MacProjector.hpp"

namespace quokka
{

MacProjector::MacProjector(const amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>> &a_umac,
			   const amrex::Vector<amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM>> &a_beta, const amrex::Vector<amrex::Geometry> &a_geom,
			   const amrex::LPInfo &a_lpinfo)
    : umac_(a_umac), geom_(a_geom), needsLevelBCs_(a_geom.size(), 1)
{
	const auto nlevs = static_cast<int>(a_beta.size());
	amrex::Vector<amrex::BoxArray> ba(nlevs);
	amrex::Vector<amrex::DistributionMapping> dm(nlevs);
	for (int ilev = 0; ilev < nlevs; ++ilev) {
		ba[ilev] = amrex::convert(a_beta[ilev][0]->boxArray(), amrex::IntVect::TheZeroVector());
		dm[ilev] = a_beta[ilev][0]->DistributionMap();
	}

	rhs_.resize(nlevs);
	phi_.resize(nlevs);
	fluxes_.resize(nlevs);
	for (int ilev = 0; ilev < nlevs; ++ilev) {
		rhs_[ilev].define(ba[ilev], dm[ilev], 1, 0);
		phi_[ilev].define(ba[ilev], dm[ilev], 1, 1);
		rhs_[ilev].setVal(0.0);
		phi_[ilev].setVal(0.0);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fluxes_[ilev][idim].define(amrex::convert(ba[ilev], amrex::IntVect::TheDimensionVector(idim)), dm[ilev], 1, 0);
		}
	}

	abeclap_ = std::make_unique<amrex::MLABecLaplacian>(geom_, ba, dm, a_lpinfo);

	bool use_gauss_seidel = true;
	{
		const amrex::ParmParse pp("mac_proj");
		pp.query("use_gauss_seidel", use_gauss_seidel);
	}
	abeclap_->setGaussSeidel(use_gauss_seidel);

	abeclap_->setScalars(0.0, 1.0);
	for (int ilev = 0; ilev < nlevs; ++ilev) {
		abeclap_->setBCoeffs(ilev, a_beta[ilev]);
	}

	mlmg_ = std::make_unique<amrex::MLMG>(*abeclap_);
	SetOptions();
}

void MacProjector::SetDomainBC(const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &lobc, const amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> &hibc)
{
	abeclap_->setDomainBC(lobc, hibc);
	needsDomainBCs_ = false;
}

void MacProjector::SetLevelBC(int amrlev, const amrex::MultiFab *levelbcdata)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!needsDomainBCs_, "MacProjector: SetDomainBC must be called before SetLevelBC");
	abeclap_->setLevelBC(amrlev, levelbcdata);
	needsLevelBCs_[amrlev] = 0;
}

void MacProjector::Project(amrex::Real reltol, amrex::Real atol)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!needsDomainBCs_, "MacProjector: SetDomainBC must be called before Project");
	const auto nlevs = static_cast<int>(rhs_.size());
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(umac_.size()) >= nlevs, "MacProjector: umac must be set on every level");

	for (int ilev = 0; ilev < nlevs; ++ilev) {
		// reset phi so that repeated calls do not depend on the previous solution
		phi_[ilev].setVal(0.0);
		if (needsLevelBCs_[ilev] != 0) {
			abeclap_->setLevelBC(ilev, nullptr);
			needsLevelBCs_[ilev] = 0;
		}
	}

	AverageDownVelocity();

	// MLABecLaplacian solves -div(beta grad phi) = rhs, so rhs = -div(umac)
	for (int ilev = 0; ilev < nlevs; ++ilev) {
		amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> u;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			u[idim] = umac_[ilev][idim];
		}
		amrex::computeDivergence(rhs_[ilev], u, geom_[ilev]);
		rhs_[ilev].mult(-1.0);
	}

	mlmg_->solve(amrex::GetVecOfPtrs(phi_), amrex::GetVecOfConstPtrs(rhs_), reltol, atol);

	// the MLMG flux is -beta grad phi
	mlmg_->getFluxes(amrex::GetVecOfArrOfPtrs(fluxes_), amrex::MLMG::Location::FaceCenter);
	for (int ilev = 0; ilev < nlevs; ++ilev) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::Add(*umac_[ilev][idim], fluxes_[ilev][idim], 0, 0, 1, 0);
		}
	}

	AverageDownVelocity();
}

void MacProjector::SetOptions()
{
	int maxorder = 3;
	int bottom_verbose = 0;
	int bottom_maxiter = 200;
	amrex::Real bottom_rtol = 1.0e-4;
	amrex::Real bottom_atol = -1.0;
	std::string bottom_solver("bicg");
	int num_pre_smooth = 2;
	int num_post_smooth = 2;
	int num_final_smooth = 8;

	const amrex::ParmParse pp("mac_proj");
	pp.query("verbose", verbose_);
	pp.query("maxorder", maxorder);
	pp.query("bottom_verbose", bottom_verbose);
	pp.query("maxiter", maxiter_);
	pp.query("bottom_maxiter", bottom_maxiter);
	pp.query("bottom_rtol", bottom_rtol);
	pp.query("bottom_atol", bottom_atol);
	pp.query("bottom_solver", bottom_solver);
	pp.query("num_pre_smooth", num_pre_smooth);
	pp.query("num_post_smooth", num_post_smooth);
	pp.query("num_final_smooth", num_final_smooth);

	abeclap_->setMaxOrder(maxorder);
	mlmg_->setVerbose(verbose_);
	mlmg_->setBottomVerbose(bottom_verbose);
	mlmg_->setMaxIter(maxiter_);
	mlmg_->setBottomMaxIter(bottom_maxiter);
	mlmg_->setBottomTolerance(bottom_rtol);
	mlmg_->setBottomToleranceAbs(bottom_atol);
	mlmg_->setPreSmooth(num_pre_smooth);
	mlmg_->setPostSmooth(num_post_smooth);
	mlmg_->setFinalSmooth(num_final_smooth);

	if (bottom_solver == "smoother") {
		mlmg_->setBottomSolver(amrex::MLMG::BottomSolver::smoother);
	} else if (bottom_solver == "bicg") {
		mlmg_->setBottomSolver(amrex::MLMG::BottomSolver::bicgstab);
	} else if (bottom_solver == "cg") {
		mlmg_->setBottomSolver(amrex::MLMG::BottomSolver::cg);
	} else if (bottom_solver == "bicgcg") {
		mlmg_->setBottomSolver(amrex::MLMG::BottomSolver::bicgcg);
	} else if (bottom_solver == "cgbicg") {
		mlmg_->setBottomSolver(amrex::MLMG::BottomSolver::cgbicg);
	} else {
		amrex::Abort("MacProjector: unknown mac_proj.bottom_solver '" + bottom_solver + "'");
	}
}

void MacProjector::AverageDownVelocity()
{
	const auto finest_level = static_cast<int>(umac_.size()) - 1;
	for (int lev = finest_level; lev > 0; --lev) {
		const amrex::IntVect rr = geom_[lev].Domain().size() / geom_[lev - 1].Domain().size();
		amrex::average_down_faces(amrex::GetArrOfConstPtrs(umac_[lev]), umac_[lev - 1], rr, geom_[lev - 1]);
	}
}

} // namespace quokka
