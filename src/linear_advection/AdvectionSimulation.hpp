#ifndef ADVECTION_SIMULATION_HPP_ // NOLINT
#define ADVECTION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AdvectionSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for linear advection.

#include <algorithm>
#include <array>
#include <fstream>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TagBox.H"
#include <AMReX_FluxRegister.H>

#include "linear_advection/linear_advection.hpp"
#include "simulation.hpp"
#include "util/ArrayView.hpp"
#include <format>

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_cc_;
	using AMRSimulation<problem_t>::state_new_cc_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dt_;
	using AMRSimulation<problem_t>::BCs_cc_;
	using AMRSimulation<problem_t>::nghost_cc_;
	using AMRSimulation<problem_t>::nghost_fc_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::istep;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::componentNames_cc_;

	using AMRSimulation<problem_t>::CustomPlotFileName;
	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::refRatio;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::tOld_;
	using AMRSimulation<problem_t>::tNew_;
	using AMRSimulation<problem_t>::boxArray;
	using AMRSimulation<problem_t>::DistributionMap;

	using AMRSimulation<problem_t>::max_level;
	using AMRSimulation<problem_t>::n_error_buf;

#if AMREX_SPACEDIM == 3
	using AMRSimulation<problem_t>::luminosityTables_;
#endif // AMREX_SPACEDIM == 3

	explicit AdvectionSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) { initialize(); }
	explicit AdvectionSimulation() : AMRSimulation<problem_t>() { initialize(); }

	void initialize()
	{
		AMRSimulation<problem_t>::initialize();
		componentNames_cc_.push_back({"density"});
	}

	void setCustomGhostCells() override
	{
		// PPM_EP reconstructs a 3-cell ghost range with a 5-point stencil.
		constexpr int reconstruct_ghost = 3;
		constexpr int required_cell_ghost = reconstruct_ghost + 2;
		nghost_cc_ = std::max(nghost_cc_, required_cell_ghost);
		nghost_fc_ = std::max(nghost_fc_, nghost_cc_);
	}

	void computeMaxSignalLocal(int level) override;
	void printCellProperties(int lev, amrex::IntVect const &index) override;
	void preCalculateInitialConditions() override;
	void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;
	void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;
#if AMREX_SPACEDIM == 3
	void createInitialRadParticles() override;
	void createInitialCICParticles() override;
	void createInitialCICRadParticles() override;
	void createInitialStochasticStellarPopParticles() override;
	void createInitialSinkParticles() override;
	void createInitialTestParticles() override;
#endif // AMREX_SPACEDIM == 3
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/) override;
	void computeBeforeTimestep() override;
	void computeAfterTimestep() override;
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);
	void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames,
						int lev, int interval) override;
	void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;
	void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;
	// compute projected vars

	// compute statistics
	auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;

	void FixupState(int lev) override;

	// tag cells for refinement
	void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	auto computeFluxes(amrex::MultiFab const &consVar, int nvars, int lev)
	    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
			  std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	template <FluxDir DIR>
	void fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel,
			  amrex::MultiFab &x1LeftState, amrex::MultiFab &x1RightState, int ng_reconstruct, int nvars);

	double advectionVx_ = 1.0; // default
	double advectionVy_ = 0.0; // default
	double advectionVz_ = 0.0; // default

	amrex::Real errorNorm_ = NAN;

	static constexpr int integratorOrder_ = 2; // RK2-SSP = 2, forward Euler = 1
};

template <typename problem_t> void AdvectionSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_cc_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal, advectionVx_, advectionVy_, advectionVz_, indexRange);
	}
}

template <typename problem_t> void AdvectionSimulation<problem_t>::printCellProperties(int lev, amrex::IntVect const &index)
{
	// deliberately empty
}

template <typename problem_t> void AdvectionSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev)
{
	// deliberately empty
}

template <typename problem_t> void AdvectionSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt)
{
	// deliberately empty
}

template <typename problem_t> void AdvectionSimulation<problem_t>::preCalculateInitialConditions()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if face-centered vars are used
}

#if AMREX_SPACEDIM == 3

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialRadParticles()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if Rad particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialCICParticles()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if CIC particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialCICRadParticles()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if CICRad particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialStochasticStellarPopParticles()
{
	// Optional implementation
	// note: an implementation is only effective if StochasticStellarPop particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialSinkParticles()
{
	// Optional implementation
	// note: an implementation is only effective if Sink particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialTestParticles()
{
	// Optional implementation
	// note: an implementation is only effective if Test particles are used
}
#endif // AMREX_SPACEDIM == 3

template <typename problem_t> void AdvectionSimulation<problem_t>::computeBeforeTimestep()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const
{
	// user should implement
}

template <typename problem_t> auto AdvectionSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	// user should implement
	return std::map<std::string, amrex::Real>{};
}

template <typename problem_t> void AdvectionSimulation<problem_t>::refineGrid(int /*lev*/, amrex::TagBoxArray & /*tags*/, amrex::Real /*time*/, int /*ngrow*/)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow)
{
	// call user-defined RefineGrid to set tags
	refineGrid(lev, tags, time, ngrow);
}

template <typename problem_t> void AdvectionSimulation<problem_t>::FixupState(int lev)
{
	// fix negative states
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{
	// user implemented
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	// compute reference solution
	const int ncomp = state_new_cc_[0].nComp();
	amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, 0);
	computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray(), geom[0].ProbHiArray());

	// compute error norm
	amrex::MultiFab residual(boxArray(0), DistributionMap(0), ncomp, 0);
	amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, 0);
	amrex::MultiFab::Saxpy(residual, -1., state_new_cc_[0], 0, 0, ncomp, 0);

	amrex::Real sol_norm = 0.;
	amrex::Real err_norm = 0.;
	// compute rms of each component
	for (int n = 0; n < ncomp; ++n) {
		sol_norm += std::pow(state_ref_level0.norm1(n), 2);
		err_norm += std::pow(residual.norm1(n), 2);
	}
	sol_norm = std::sqrt(sol_norm);
	err_norm = std::sqrt(err_norm);
	const double rel_error = err_norm / sol_norm;
	errorNorm_ = rel_error;

	amrex::Print() << "\nRelative rms L1 error norm = " << rel_error << "\n\n";
}

template <typename problem_t> void AdvectionSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/)
{
	// based on amrex/Tests/EB/CNS/Source/CNS_advance.cpp

	// since we are starting a new timestep, need to swap old and new states on this
	// level
	std::swap(state_old_cc_[lev], state_new_cc_[lev]);

	// check state validity
	AMREX_ASSERT(!state_old_cc_[lev].contains_nan(0, state_old_cc_[lev].nComp()));
	AMREX_ASSERT(!state_old_cc_[lev].contains_nan()); // check ghost cells

	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];

	// get flux registers
	amrex::FluxRegister *fr_as_crse = nullptr;
	amrex::FluxRegister *fr_as_fine = nullptr;
	if (do_reflux && lev < finest_level) {
		fr_as_crse = flux_reg_[lev + 1].get();
		fr_as_crse->setVal(0.0);
	}

	if (do_reflux && lev > 0) {
		fr_as_fine = flux_reg_[lev].get();
	}

	// create temporary MultiFab to store the fluxes from each grid on this level
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

	if (do_reflux) {
		for (int j = 0; j < AMREX_SPACEDIM; j++) {
			amrex::BoxArray ba = state_new_cc_[lev].boxArray();
			ba.surroundingNodes(j);
			fluxes[j].define(ba, dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, 0);
			fluxes[j].setVal(0.0);
		}
	}

	// We use the RK2-SSP integrator in a method-of-lines framework. It needs 2
	// registers: one to store the old timestep, and one to store the intermediate stage
	// and final stage. The intermediate stage and final stage reuse the same register.

	// update ghost zones [w/ old timestep]
	// (N.B. the input and output multifabs are allowed to be the same, as done here)
	fillBoundaryConditions(state_old_cc_[lev], state_old_cc_[lev], lev, time, quokka::centering::cc, quokka::direction::na,
			       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone);

	amrex::Real fluxScaleFactor = NAN;
	if constexpr (integratorOrder_ == 2) {
		fluxScaleFactor = 0.5;
	} else if constexpr (integratorOrder_ == 1) {
		fluxScaleFactor = 1.0;
	}

	// advance all grids on local processor (Stage 1 of integrator)
	{
		auto const &stateOld = state_old_cc_[lev];
		auto &stateNew = state_new_cc_[lev];
		auto [fluxArrays, faceVelArrays, leftStateArrays, rightStateArrays] = computeFluxes(stateOld, Physics_Indices<problem_t>::nvarTotal_cc, lev);

		// Write face velocities to disk
		// this->writeFaceVelocitiesToDisk(faceVelArrays, lev, cycleCount_);

		// Write reconstructed states to disk
		// this->writeReconstructedStatesToDisk(leftStateArrays, rightStateArrays, lev, cycleCount_);

		// Stage 1 of RK2-SSP
		LinearAdvectionSystem<problem_t>::PredictStep(stateOld, stateNew, fluxArrays, dt_lev, geomLevel.CellSizeArray(),
							      Physics_Indices<problem_t>::nvarTotal_cc);

		if (do_reflux) {
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				fluxes[i].plus(fluxArrays[i], 0, fluxArrays[i].nComp(), 0);
			}
		}
	}

	if constexpr (integratorOrder_ == 2) {
		// update ghost zones [w/ intermediate stage stored in state_new_cc_]
		fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, (time + dt_lev), quokka::centering::cc, quokka::direction::na,
				       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone);

		// advance all grids on local processor (Stage 2 of integrator)
		{
			auto const &stateInOld = state_old_cc_[lev];
			auto const &stateInStar = state_new_cc_[lev];
			auto &stateOut = state_new_cc_[lev];
			auto [fluxArrays, faceVelArrays, leftStateArrays, rightStateArrays] =
			    computeFluxes(stateInStar, Physics_Indices<problem_t>::nvarTotal_cc, lev);

			// Stage 2 of RK2-SSP
			LinearAdvectionSystem<problem_t>::AddFluxesRK2(stateOut, stateInOld, stateInStar, fluxArrays, dt_lev, geomLevel.CellSizeArray(),
								       Physics_Indices<problem_t>::nvarTotal_cc);

			if (do_reflux) {
				for (int i = 0; i < AMREX_SPACEDIM; ++i) {
					fluxes[i].plus(fluxArrays[i], 0, fluxArrays[i].nComp(), 0);
				}
			}
		}
	}

	if (do_reflux) {
		incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxes, lev, fluxScaleFactor * dt_lev);
	}
}

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
		  std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int reconstructRange = 3; // fully reconstruct a parabola within *three* cells outside the valid region
	// NOTE: one cell is needed to get L/R states at the FAB boundaries.
	//   The extra cells are needed to get L/R states (and therefore the face velocity) for *two* ghost faces.
	//   (For hydro, we need *two* ghost face velocities in order to do particle MAC advection.)

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange - 1);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructRange - 1);
	}

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, primVar, flux[0], facevel[0], leftState[0], rightState[0], reconstructRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVar, primVar, flux[1], facevel[1], leftState[1], rightState[1], reconstructRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVar, primVar, flux[2], facevel[2], leftState[2], rightState[2], reconstructRange, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();
	return std::make_tuple(std::move(flux), std::move(facevel), std::move(leftState), std::move(rightState));
}

template <typename problem_t>
template <FluxDir DIR>
void AdvectionSimulation<problem_t>::fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux,
						  amrex::MultiFab &x1FaceVel, amrex::MultiFab &x1LeftState, amrex::MultiFab &x1RightState,
						  const int ng_reconstruct, const int nvars)
{
	amrex::Real advectionVel = NAN;
	if constexpr (DIR == FluxDir::X1) {
		advectionVel = advectionVx_;
	} else if constexpr (DIR == FluxDir::X2) {
		advectionVel = advectionVy_;
	} else if constexpr (DIR == FluxDir::X3) {
		advectionVel = advectionVz_;
	}

	LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consState, primVar, nghost_cc_, nvars);

	LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar, x1LeftState, x1RightState, ng_reconstruct, nvars);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux, x1LeftState, x1RightState, x1FaceVel, advectionVel, nvars);
}

// Save single-level plotfile
// This is a wrapper around the WriteSingleLevelPlotfile function in the AMReX library.
// The step number of the plotfile is set to istep[lev] and the time is set to the current time tNew_[lev].
// Example usage: write debug_rhs0000000 debug_rhs0000001 etc with interval plotfileInterval_
//   const int lev_debug = 0;
//   amrex::Vector<std::string> flatCompNames{"rhs"};
//   WriteSingleLevelPlotfileSimplified("debug_rhs", rhs[lev_debug], flatCompNames, lev_debug, plotfileInterval_);
template <typename problem_t>
void AdvectionSimulation<problem_t>::WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf,
									const amrex::Vector<std::string> &compNames, int lev, int interval)
{
	if ((istep[lev] % interval) != 0) {
		return;
	}
	const auto plotfile_name = CustomPlotFileName(plotfile_prefix.c_str(), istep[lev]);
	WriteSingleLevelPlotfile(plotfile_name, mf, compNames, geom[lev], tNew_[lev], istep[lev]);
}

#endif // ADVECTION_SIMULATION_HPP_
