#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file RadhydroSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include <array>
#include <climits>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "AMReX.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuControl.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"
#include "AMReX_YAFluxRegister.H"

#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class RadhydroSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::boundaryConditions_;
	using AMRSimulation<problem_t>::componentNames_;
	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::Verbose;
	using AMRSimulation<problem_t>::disable_radiation_transport_terms;
	using AMRSimulation<problem_t>::constantDt_;
	using AMRSimulation<problem_t>::boxArray;
	using AMRSimulation<problem_t>::DistributionMap;

	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;

	static constexpr int nvarTotal_ = RadSystem<problem_t>::nvar_;
	static constexpr int ncompHydro_ = HydroSystem<problem_t>::nvar_; // hydro
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;

	amrex::Real radiationCflNumber_ = 0.3;
	bool is_hydro_enabled_ = false;
	bool is_radiation_enabled_ = true;
	bool computeReferenceSolution_ = false;
	amrex::Real errorNorm_ = NAN;

	int integratorOrder_ = 2; // 1 == forward Euler; 2 == RK2-SSP (default)
	int reconstructionOrder_ = 3; // 1 == donor cell; 2 == PLM; 3 == PPM (default)

	// member functions

	explicit RadhydroSimulation(amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : AMRSimulation<problem_t>(boundaryConditions,
				       RadSystem<problem_t>::nvar_, ncompHyperbolic_)
	{
		componentNames_ = {"gasDensity",    "x-GasMomentum", "y-GasMomentum",
				   "z-GasMomentum", "gasEnergy",     "radEnergy",
				   "x-RadFlux",	    "y-RadFlux",     "z-RadFlux"};
	}

	RadhydroSimulation(amrex::IntVect & /*gridDims*/, amrex::RealBox & /*boxSize*/,
			   amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : AMRSimulation<problem_t>(boundaryConditions,
				       RadSystem<problem_t>::nvar_, ncompHyperbolic_)
	{
		componentNames_ = {"gasDensity",    "x-GasMomentum", "y-GasMomentum",
				   "z-GasMomentum", "gasEnergy",     "radEnergy",
				   "x-RadFlux",	    "y-RadFlux",     "z-RadFlux"};
	}

	void checkHydroStates(int level);
	void computeMaxSignalLocal(int level) override;
	void setInitialConditionsAtLevel(int level) override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
					  int iteration, int ncycle) override;
	void computeAfterTimestep() override;
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref,
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	auto expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, int nstartNew,
			      int ncompNew) -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	void advanceHydroAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
				 amrex::YAFluxRegister *fr_as_crse,
				 amrex::YAFluxRegister *fr_as_fine);

	// radiation subcycle
	void swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew);
	void advanceRadiationSubstepAtLevel(int lev, amrex::Real time,
						   amrex::Real dt_radiation, int iteration, int nsubsteps,
						   amrex::YAFluxRegister *fr_as_crse,
						   amrex::YAFluxRegister *fr_as_fine);
	void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro,
				      amrex::YAFluxRegister *fr_as_crse,
				      amrex::YAFluxRegister *fr_as_fine);

	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew,
			const amrex::Box &indexRange, amrex::Real time, double dt,
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);

	auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar,
				    const amrex::Box &indexRange, int nvars,
				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
	    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>,
			  std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;

	auto computeHydroFluxes(amrex::Array4<const amrex::Real> const &consVar,
				const amrex::Box &indexRange, int nvars)
	    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::Array4<const amrex::Real> const &consState,
			       amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);
};

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	BL_PROFILE("RadhydroSimulation::computeMaxSignalLocal()");

	// hydro: loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew = state_new_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);

		if (is_hydro_enabled_) {
			// hydro enabled
			HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal,
								      indexRange);
		} else if (is_radiation_enabled_) {
			// hydro disabled, radiation enabled
			RadSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal,
								    indexRange);
		} else {
			// hydro disabled, radiation disabled
			amrex::Abort("At least one of hydro or radiation must be enabled! Cannot "
				     "compute a time step.");
		}
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::checkHydroStates(int const level)
{
	BL_PROFILE("RadhydroSimulation::checkHydroStates()");

	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew = state_new_[level].const_array(iter);
		AMREX_ALWAYS_ASSERT(HydroSystem<problem_t>::CheckStatesValid(indexRange, stateNew));
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::setInitialConditionsAtLevel(int level)
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void RadhydroSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags,
					     amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement -- user should implement
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	// user should implement
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	// check conservation of total energy
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	amrex::Real const Egas0 = initSumCons[RadSystem<problem_t>::gasEnergy_index];
	amrex::Real const Erad0 = initSumCons[RadSystem<problem_t>::radEnergy_index];
	amrex::Real const Etot0 = Egas0 + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad0;

	amrex::Real const Egas = state_new_[0].sum(RadSystem<problem_t>::gasEnergy_index) * vol;
	amrex::Real const Erad = state_new_[0].sum(RadSystem<problem_t>::radEnergy_index) * vol;
	amrex::Real const Etot = Egas + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad;

	amrex::Real const abs_err = (Etot - Etot0);
	amrex::Real const rel_err = abs_err / Etot0;

	amrex::Print() << "\nInitial gas+radiation energy = " << Etot0 << std::endl;
	amrex::Print() << "Final gas+radiation energy = " << Etot << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err << std::endl;
	amrex::Print() << std::endl;

	if (computeReferenceSolution_) {
		// compute reference solution
		const int ncomp = state_new_[0].nComp();
		const int nghost = state_new_[0].nGrow();
		amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, nghost);
		computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray());

		// compute error norm
		amrex::MultiFab residual(boxArray(0), DistributionMap(0), ncomp, nghost);
		amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, nghost);
		amrex::MultiFab::Saxpy(residual, -1., state_new_[0], 0, 0, ncomp, nghost);

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
		amrex::Print() << "Relative rms L1 error norm = " << rel_error << std::endl;
	}
	amrex::Print() << std::endl;
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								 amrex::Real dt_lev,
								 int /*iteration*/, int /*ncycle*/)
{
	BL_PROFILE("RadhydroSimulation::advanceSingleTimestepAtLevel()");

	// get flux registers
	amrex::YAFluxRegister *fr_as_crse = nullptr;
	amrex::YAFluxRegister *fr_as_fine = nullptr;
	if (do_reflux != 0) {
		if (lev < finestLevel()) {
			fr_as_crse = flux_reg_[lev + 1].get();
			fr_as_crse->reset();
		}
		if (lev > 0) {
			fr_as_fine = flux_reg_[lev].get();
		}
	}

	// since we are starting a new timestep, need to swap old and new state vectors
	std::swap(state_old_[lev], state_new_[lev]);

	// advance hydro
	if (is_hydro_enabled_) {
		advanceHydroAtLevel(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	}

	// check hydro states
	checkHydroStates(lev);

	// subcycle radiation
	if (is_radiation_enabled_) {
		subcycleRadiationAtLevel(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	}

	// check state validity
	AMREX_ASSERT(!state_new_[lev].contains_nan(0, state_new_[lev].nComp()));
	AMREX_ASSERT(!state_new_[lev].contains_nan()); // check ghost zones
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceHydroAtLevel(int lev, amrex::Real time,
							amrex::Real dt_lev,
							amrex::YAFluxRegister *fr_as_crse,
							amrex::YAFluxRegister *fr_as_fine)
{
	BL_PROFILE("RadhydroSimulation::advanceHydroAtLevel()");

	amrex::Real fluxScaleFactor = NAN;
	if (integratorOrder_ == 2) {
		fluxScaleFactor = 0.5;
	} else if (integratorOrder_ == 1) {
		fluxScaleFactor = 1.0;
	}

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_[lev], state_old_[lev], lev, time);

	// check state validity
	AMREX_ASSERT(!state_old_[lev].contains_nan(0, state_old_[lev].nComp()));
	AMREX_ASSERT(!state_old_[lev].contains_nan()); // check ghost cells

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto fluxArrays = computeHydroFluxes(stateOld, indexRange, ncompHydro_);

		// Stage 1 of RK2-SSP
		HydroSystem<problem_t>::PredictStep(
		    stateOld, stateNew,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    dt_lev, geom[lev].CellSizeArray(), indexRange, ncompHydro_);

		if (do_reflux) {
			// increment flux registers
			auto expandedFluxes = expandFluxArrays(fluxArrays, 0, state_new_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev,
					       fluxScaleFactor * dt_lev);
		}

		AMREX_ASSERT(HydroSystem<problem_t>::CheckStatesValid(indexRange, stateNew));
	}

	if (integratorOrder_ == 2) {
		// update ghost zones [intermediate stage stored in state_new_]
		fillBoundaryConditions(state_new_[lev], state_new_[lev], lev, time + dt_lev);

		// check intermediate state validity
		AMREX_ASSERT(!state_new_[lev].contains_nan(0, state_new_[lev].nComp()));
		AMREX_ASSERT(!state_new_[lev].contains_nan()); // check ghost zones

		// advance all grids on local processor (Stage 2 of integrator)
		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
			auto const &stateOld = state_old_[lev].const_array(iter);
			auto const &stateInter = state_new_[lev].const_array(iter);
			auto const &stateNew = state_new_[lev].array(iter);
			auto fluxArrays = computeHydroFluxes(stateInter, indexRange, ncompHydro_);

			// Stage 2 of RK2-SSP
			HydroSystem<problem_t>::AddFluxesRK2(
				stateNew, stateOld, stateInter,
				{AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					fluxArrays[2].const_array())},
				dt_lev, geom[lev].CellSizeArray(), indexRange, ncompHydro_);

			if (do_reflux) {
				// increment flux registers
				auto expandedFluxes = expandFluxArrays(fluxArrays, 0, state_new_[lev].nComp());
				incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev,
							fluxScaleFactor * dt_lev);
			}

			AMREX_ASSERT(HydroSystem<problem_t>::CheckStatesValid(indexRange, stateNew));
		}
	}
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::expandFluxArrays(
    std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, const int nstartNew, const int ncompNew)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	BL_PROFILE("RadhydroSimulation::expandFluxArrays()");

	// This is needed because reflux arrays must have the same number of components as
	// state_new_[lev]
	auto copyFlux = [nstartNew, ncompNew](amrex::FArrayBox const &oldFlux) {
		amrex::Box const &fluxRange = oldFlux.box();
		amrex::FArrayBox newFlux(fluxRange, ncompNew, amrex::The_Async_Arena());
		newFlux.setVal<amrex::RunOn::Device>(0.);
		// copy oldFlux (starting at 0) to newFlux (starting at nstart)
		AMREX_ASSERT(ncompNew > oldFlux.nComp());
		newFlux.copy<amrex::RunOn::Device>(oldFlux, 0, nstartNew, oldFlux.nComp());
		return newFlux;
	};
	return {AMREX_D_DECL(copyFlux(fluxes[0]), copyFlux(fluxes[1]), copyFlux(fluxes[2]))};
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeHydroFluxes(
    amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	BL_PROFILE("RadhydroSimulation::computeHydroFluxes()");

	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
#endif

	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(consVar, x1Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X2>(consVar, x2Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X3>(consVar, x3Flux, indexRange, nvars);)

	return {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::hydroFluxFunction(
    amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux,
    const amrex::Box &indexRange, const int nvars)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &flatteningRange = amrex::grow(indexRange, 2); // +1 greater than ppmRange
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1Flat(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	HydroSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	if (reconstructionOrder_ == 3) {
		// mixed interface/cell-centered kernel
		HydroSystem<problem_t>::template ReconstructStatesPPM<DIR>(
			primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
			x1ReconstructRange, nvars);
	} else if (reconstructionOrder_ == 1) {
		// interface-centered kernel
		HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(
			primVar.array(), x1LeftState.array(), x1RightState.array(),
			x1ReconstructRange, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}

	// cell-centered kernel
	HydroSystem<problem_t>::template ComputeFlatteningCoefficients<DIR>(
	    primVar.array(), x1Flat.array(), flatteningRange);

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(
	    primVar.array(), x1Flat.array(), x1LeftState.array(), x1RightState.array(),
	    reconstructRange, nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	HydroSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(),
	    x1FluxRange); // watch out for argument order!!
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew)
{
	// copy radiation state variables from stateNew to stateOld
	amrex::MultiFab::Copy(stateOld, stateNew, nstartHyperbolic_, nstartHyperbolic_, ncompHyperbolic_, 0);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::subcycleRadiationAtLevel(int lev, amrex::Real time,
							     amrex::Real dt_lev_hydro,
							     amrex::YAFluxRegister *fr_as_crse,
							     amrex::YAFluxRegister *fr_as_fine)
{
	// compute radiation timestep
	amrex::Real domain_signal_max = RadSystem<problem_t>::c_hat_;
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / domain_signal_max);

	amrex::Long nsubSteps = 0;
	amrex::Real dt_radiation = NAN;

	if (is_hydro_enabled_ && !(constantDt_ > 0.)) {
		// adjust to get integer number of substeps
		nsubSteps = std::ceil(dt_lev_hydro / dtrad_tmp);
		dt_radiation = dt_lev_hydro / static_cast<double>(nsubSteps);
	} else { // no hydro, or using constant dt (this is necessary for radiation test problems)
		dt_radiation = dt_lev_hydro;
		nsubSteps = 1;
	}

	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps < 1e4);
	AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);

	if (Verbose() != 0) {
		amrex::Print() << "\tRadiation substeps: " << nsubSteps << "\tdt: " << dt_radiation
			       << "\n";
	}

	// perform subcycle
	amrex::Real time_subcycle = time;
	for (int i = 0; i < nsubSteps; ++i) {
		if (i > 0) {
			// since we are starting a new substep, we need to copy radiation state from
			// 	new state vector to old state vector
			// (this is not necessary for the i=0 substep because we have already swapped
			//  the full hydro+radiation state vectors at the beginning of the level advance)
			swapRadiationState(state_old_[lev], state_new_[lev]);
		}

		// advance hyperbolic radiation subsystem starting from state_old_ to state_new_
		advanceRadiationSubstepAtLevel(lev, time_subcycle, dt_radiation, i, nsubSteps,
							  fr_as_crse, fr_as_fine);

		// new radiation state is stored in state_new_
		// new hydro state is stored in state_new_ (always the case during radiation update)

		// matter-radiation exchange source terms
		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateNew = state_new_[lev].array(iter);
			auto const &prob_lo = geom[lev].ProbLoArray();
			// update state_new_[lev] in place (updates both radiation and hydro vars)
			operatorSplitSourceTerms(stateNew, indexRange, time_subcycle, dt_radiation, dx, prob_lo);
		}

		// new hydro+radiation state is stored in state_new_

		// check hydro states post-update
		// [If this fails, the *most likely* cause is that the flow is radiation-pressure-dominated,
		//  and the hydro timestep is larger than the timescale of radiative acceleration.
		//  In this case, in the radiation work term \vec{v} dot (-\vec{F_rad}/c^2),
		//  v cannot be approximated as constant!]
		checkHydroStates(lev);

		// update 'time_subcycle'
		time_subcycle += dt_radiation;
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceRadiationSubstepAtLevel(
    int lev, amrex::Real time, amrex::Real dt_radiation, int const iter, int const /*nsubsteps*/,
	amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine)
{
	if (Verbose()) {
		amrex::Print() << "\tsubstep " << iter << " t = " << time
					   << "\t t+dt = " << time+dt_radiation << std::endl;
	}

	// get cell sizes
	auto const &dx = geom[lev].CellSizeArray();

	// We use the RK2-SSP method here. It needs two registers: one to store the old timestep,
	// and another to store the intermediate stage (which is reused for the final stage).

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_[lev], state_old_[lev], lev, time);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] =
			computeRadiationFluxes(stateOld, indexRange, ncompHyperbolic_, dx);

		// Stage 1 of RK2-SSP
		RadSystem<problem_t>::PredictStep(
			stateOld, stateNew,
			{AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(),
				fluxArrays[2].array())},
			{AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(),
				fluxDiffusiveArrays[1].const_array(),
				fluxDiffusiveArrays[2].const_array())},
			dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			// increment flux registers
			// WARNING: as written, diffusive flux correction is not compatible with reflux!!
			auto expandedFluxes =
				expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev,
						0.5 * dt_radiation);
		}
	}

	// update ghost zones [intermediate stage stored in state_new_]
	fillBoundaryConditions(state_new_[lev], state_new_[lev], lev, time + dt_radiation);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateInter = state_new_[lev].const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] =
			computeRadiationFluxes(stateInter, indexRange, ncompHyperbolic_, dx);

		// Stage 2 of RK2-SSP
		RadSystem<problem_t>::AddFluxesRK2(
			stateNew, stateOld, stateInter,
			{AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(),
				fluxArrays[2].array())},
			{AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(),
				fluxDiffusiveArrays[1].const_array(),
				fluxDiffusiveArrays[2].const_array())},
			dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			// increment flux registers
			// WARNING: as written, diffusive flux correction is not compatible with reflux!!
			auto expandedFluxes =
				expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev,
						0.5 * dt_radiation);
		}
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::operatorSplitSourceTerms(
    amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, 
	const amrex::Real time, const double dt,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	amrex::FArrayBox radEnergySource(indexRange, 1,
					 amrex::The_Async_Arena()); // cell-centered scalar
	amrex::FArrayBox advectionFluxes(indexRange, 3,
					 amrex::The_Async_Arena()); // cell-centered vector

	radEnergySource.setVal<amrex::RunOn::Device>(0.);
	advectionFluxes.setVal<amrex::RunOn::Device>(0.);

	// cell-centered radiation energy source (used only in test problems)
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange,
						 dx, prob_lo, time + dt);

	// cell-centered source terms
	RadSystem<problem_t>::AddSourceTerms(stateNew, radEnergySource.const_array(),
					     advectionFluxes.const_array(), indexRange, dt);
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeRadiationFluxes(
    amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>,
		  std::array<amrex::FArrayBox, AMREX_SPACEDIM>>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
	amrex::FArrayBox x1FluxDiffusive(x1FluxRange, nvars, amrex::The_Async_Arena());
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
	amrex::FArrayBox x2FluxDiffusive(x2FluxRange, nvars, amrex::The_Async_Arena());
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
	amrex::FArrayBox x3FluxDiffusive(x3FluxRange, nvars, amrex::The_Async_Arena());
#endif

	AMREX_D_TERM(
	    fluxFunction<FluxDir::X1>(consVar, x1Flux, x1FluxDiffusive, indexRange, nvars, dx);
	    , fluxFunction<FluxDir::X2>(consVar, x2Flux, x2FluxDiffusive, indexRange, nvars, dx);
	    , fluxFunction<FluxDir::X3>(consVar, x3Flux, x3FluxDiffusive, indexRange, nvars, dx);)

	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxDiffusiveArrays{AMREX_D_DECL(
	    std::move(x1FluxDiffusive), std::move(x2FluxDiffusive), std::move(x3FluxDiffusive))};

	return std::make_tuple(std::move(fluxArrays), std::move(fluxDiffusiveArrays));
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
						 amrex::FArrayBox &x1Flux,
						 amrex::FArrayBox &x1FluxDiffusive,
						 const amrex::Box &indexRange, const int nvars,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	// mixed interface/cell-centered kernel
	// RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	//    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	//    x1ReconstructRange, nvars);
	// PLM and donor cell are interface-centered kernels
	// RadSystem<problem_t>::template ReconstructStatesConstant<DIR>(
	//     primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//     nvars);
	RadSystem<problem_t>::template ReconstructStatesPLM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange, nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(),
							  x1LeftState.array(), x1RightState.array(),
							  x1FluxRange, consState,
							  dx); // watch out for argument order!!
}

#endif // RADIATION_SIMULATION_HPP_