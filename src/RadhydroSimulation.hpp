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

#include "AMReX_FabArray.H"
#include "AMReX_GpuControl.H"
#include "AMReX_IArrayBox.H"
#include "AMReX_IndexType.H"
#include "AMReX.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"
#include "AMReX_YAFluxRegister.H"

#include "CloudyCooling.hpp"
#include "hyperbolic_system.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "simulation.hpp"
#include "SimulationData.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class RadhydroSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_cc_;
	using AMRSimulation<problem_t>::state_new_cc_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::ncomp_cc_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::BCs_cc_;
	using AMRSimulation<problem_t>::componentNames_cc_;
	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::Verbose;
	using AMRSimulation<problem_t>::constantDt_;
	using AMRSimulation<problem_t>::boxArray;
	using AMRSimulation<problem_t>::DistributionMap;
	using AMRSimulation<problem_t>::refRatio;
	using AMRSimulation<problem_t>::cellUpdates_;
	using AMRSimulation<problem_t>::CountCells;
	using AMRSimulation<problem_t>::WriteCheckpointFile;
	using AMRSimulation<problem_t>::GetData;
	using AMRSimulation<problem_t>::FillPatchWithData;

	SimulationData<problem_t> userData_;

	static constexpr int nvarTotal_cc_ = RadSystem<problem_t>::nvar_;
	static constexpr int ncompHydro_ = HydroSystem<problem_t>::nvar_; // hydro
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;

	amrex::Real radiationCflNumber_ = 0.3;
	int maxSubsteps_ = 10; // maximum number of radiation subcycles per hydro step
	
	bool computeReferenceSolution_ = false;
	amrex::Real errorNorm_ = NAN;
	amrex::Real densityFloor_ = 0.;
	amrex::Real pressureFloor_ = 0.;
	int fofcMaxIterations_ = 3; // maximum number of flux correction iterations -- only 1 is needed in almost all cases, but in rare cases a second iteration is needed

	int integratorOrder_ = 2; // 1 == forward Euler; 2 == RK2-SSP (default)
	int reconstructionOrder_ = 3; // 1 == donor cell; 2 == PLM; 3 == PPM (default)
	int radiationReconstructionOrder_ = 3; // 1 == donor cell; 2 == PLM; 3 == PPM (default)
	int useDualEnergy_ = 1; // 0 == disabled; 1 == use auxiliary internal energy equation (default)

	amrex::Long radiationCellUpdates_ = 0; // total number of radiation cell-updates

	// member functions
	explicit RadhydroSimulation(amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : AMRSimulation<problem_t>(boundaryConditions) {
    // check modules cannot be enabled if they are not been implemented yet
    static_assert(!Physics_Traits<problem_t>::is_chemistry_enabled, "Chemistry is not supported, yet.");
    
    // add hydro state variables
    if constexpr (Physics_Traits<problem_t>::is_hydro_enabled ||
                  Physics_Traits<problem_t>::is_radiation_enabled) {
      std::vector<std::string> hydroNames = {"gasDensity", "x-GasMomentum", "y-GasMomentum", "z-GasMomentum", "gasEnergy", "gasInternalEnergy"};
      componentNames_cc_.insert(componentNames_cc_.end(), hydroNames.begin(), hydroNames.end());
      ncomp_cc_ += hydroNames.size();
    }
    // add passive scalar variables
    if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0){
      std::vector<std::string> scalarNames = getScalarVariableNames();
      componentNames_cc_.insert(componentNames_cc_.end(), scalarNames.begin(), scalarNames.end());
      ncomp_cc_ += scalarNames.size();
    }
    // add radiation state variables
    if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
      std::vector<std::string> radNames = {"radEnergy", "x-RadFlux", "y-RadFlux", "z-RadFlux"};
      componentNames_cc_.insert(componentNames_cc_.end(), radNames.begin(), radNames.end());
      ncomp_cc_ += radNames.size();
    }

    // read in runtime parameters
		readParmParse();
	}

	[[nodiscard]] static auto getScalarVariableNames() -> std::vector<std::string>;
	void readParmParse();

	void checkHydroStates(amrex::MultiFab &mf, char const *file, int line);
	void computeMaxSignalLocal(int level) override;
	auto computeExtraPhysicsTimestep(int lev) -> amrex::Real override;
	void preCalculateInitialConditions() override;
  	void setInitialConditionsOnGrid(quokka::grid grid_elem) override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
									  int ncycle) override;
	void computeAfterTimestep() override;
	void computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int /*ncycle*/);
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref,
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;

	// fix-up states
	void FixupState(int level) override;

	// implement FillPatch function
	void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, FillPatchType fptype) override;

	// functions to operate on state vector before/after interpolating between levels
	static void PreInterpState(amrex::MultiFab &mf, int scomp, int ncomp);
	static void PostInterpState(amrex::MultiFab &mf, int scomp, int ncomp);

	// compute axis-aligned 1D profile of user_f(x, y, z)
	template <typename F>
	auto computeAxisAlignedProfile(int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	void addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes,
    			   std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes,
				   const int srccomp, const int dstcomp);

	auto expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, int nstartNew,
			      int ncompNew) -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	void advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev,
				 amrex::YAFluxRegister *fr_as_crse,
				 amrex::YAFluxRegister *fr_as_fine);

	auto advanceHydroAtLevel(amrex::MultiFab &state_old_tmp,
							amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
							int lev, amrex::Real time,
							amrex::Real dt_lev) -> bool;

	void addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time,
				 amrex::Real dt_lev);

	auto isCflViolated(int lev, amrex::Real time, amrex::Real dt_actual) -> bool;			 

	// radiation subcycle
	void swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew);
	auto computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int;
	void advanceRadiationSubstepAtLevel(int lev, amrex::Real time,
						   amrex::Real dt_radiation, int iter_count, int nsubsteps,
						   amrex::YAFluxRegister *fr_as_crse,
						   amrex::YAFluxRegister *fr_as_fine);
	void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro,
				      amrex::YAFluxRegister *fr_as_crse,
				      amrex::YAFluxRegister *fr_as_fine);

	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew,
			const amrex::Box &indexRange, amrex::Real time, double dt,
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);

	auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar,
				    const amrex::Box &indexRange, int nvars,
				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
	    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>,
			  std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;

	auto computeHydroFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
	    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	auto computeFOHydroFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
    	-> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::MultiFab const &primVar,
			  amrex::MultiFab &leftState,
			  amrex::MultiFab &rightState,
			  amrex::MultiFab &x1Flux,
			  amrex::MultiFab &x1FaceVel,
			  amrex::MultiFab const &x1Flat,
			  amrex::MultiFab const &x2Flat,
			  amrex::MultiFab const &x3Flat,
			  int ng_reconstruct, int nvars);

	template <FluxDir DIR>
	void hydroFOFluxFunction(amrex::MultiFab const &primVar,
			  amrex::MultiFab &leftState,
			  amrex::MultiFab &rightState,
			  amrex::MultiFab &x1Flux,
			  amrex::MultiFab &x1FaceVel,
			  int ng_reconstruct, int nvars);

	void replaceFluxes(
			std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
			std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,	amrex::iMultiFab &redoFlag);
};

template <typename problem_t>
auto RadhydroSimulation<problem_t>::getScalarVariableNames() -> std::vector<std::string> {
	// return vector of names for the passive scalars
	// this can be specialized by the user to provide more descriptive names
	// (these names are used to label the variables in the plotfiles)

	std::vector<std::string> names;
	int nscalars = HydroSystem<problem_t>::nscalars_;
	names.reserve(nscalars);
	for(int n = 0; n < nscalars; ++n) {
		// write string 'scalar_1', etc.
		names.push_back(fmt::format("scalar_{}", n));
	}
	return names;
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::readParmParse() {
	// set hydro runtime parameters
	{
		amrex::ParmParse hpp("hydro");
		hpp.query("reconstruction_order", reconstructionOrder_);
		hpp.query("use_dual_energy", useDualEnergy_);
	}

	// set radiation runtime parameters
	{
		amrex::ParmParse rpp("radiation");
		rpp.query("reconstruction_order", radiationReconstructionOrder_);
		rpp.query("cfl", radiationCflNumber_);
	}
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int
{
	// compute radiation timestep
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real c_hat = RadSystem<problem_t>::c_hat_;
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat);
	int nsubSteps = std::ceil(dt_lev_hydro / dtrad_tmp);
	return nsubSteps;
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	BL_PROFILE("RadhydroSimulation::computeMaxSignalLocal()");

	// hydro: loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew = state_new_cc_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);

		if constexpr (Physics_Traits<problem_t>::is_hydro_enabled && !(Physics_Traits<problem_t>::is_radiation_enabled)) {
			// hydro only
			HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal,
								      indexRange);
		} else if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			// radiation hydro, or radiation only
			RadSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal,
								    indexRange);
			if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
				auto maxSignalHydroFAB = amrex::FArrayBox(indexRange);
				auto const &maxSignalHydro = maxSignalHydroFAB.array();
				HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignalHydro, indexRange);
				const int maxSubsteps = maxSubsteps_;
				// ensure that we use the smaller of the two timesteps
				amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
					amrex::Real const maxSignalRadiation = maxSignal(i,j,k) / static_cast<double>(maxSubsteps);
					maxSignal(i, j, k) = std::max(maxSignalRadiation, maxSignalHydro(i, j, k));
				});
			}
		} else {
			// no physics modules enabled, why are we running?
			amrex::Abort("At least one of hydro or radiation must be enabled! Cannot "
				     "compute a time step.");
		}
	}
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeExtraPhysicsTimestep(int const level) -> amrex::Real
{
	BL_PROFILE("RadhydroSimulation::computeExtraPhysicsTimestep()");
	// users can override this to enforce additional timestep constraints
	return std::numeric_limits<amrex::Real>::max();
}

#if !defined(NDEBUG)
#define CHECK_HYDRO_STATES(mf) checkHydroStates(mf, __FILE__, __LINE__)
#else
#define CHECK_HYDRO_STATES(mf) 
#endif

template <typename problem_t>
void RadhydroSimulation<problem_t>::checkHydroStates(amrex::MultiFab &mf, char const *file, int line)
{
	BL_PROFILE("RadhydroSimulation::checkHydroStates()");

	bool validStates = HydroSystem<problem_t>::CheckStatesValid(mf);
	amrex::ParallelDescriptor::ReduceBoolAnd(validStates);

	if(!validStates) {
		amrex::Print() << "Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")\n";
		amrex::Print() << "Writing checkpoint for debugging...\n";
		amrex::MFIter::allowMultipleMFIters(true);
		WriteCheckpointFile();
		amrex::Abort("Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")");
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::preCalculateInitialConditions() {
  // default empty implementation
  // user should implement using problem-specific template specialization
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // default empty implementation
  // user should implement using problem-specific template specialization
}

template <typename problem_t> void RadhydroSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int ncycle)
{
	// user should implement if desired
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::addStrangSplitSources(amrex::MultiFab &state,
								 int lev, amrex::Real time, amrex::Real dt)
{
	// user should implement
	// (when Strang splitting is enabled, dt is actually 0.5*dt_lev)
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname,
								amrex::MultiFab &mf, const int ncomp) const
{
	// compute derived variables and save in 'mf' -- user should implement
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
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of total energy
	amrex::Real const Egas0 = initSumCons[RadSystem<problem_t>::gasEnergy_index];
	amrex::Real const Egas = state_new_cc_[0].sum(RadSystem<problem_t>::gasEnergy_index) * vol;

	amrex::Real Etot0 = NAN;
	amrex::Real Etot = NAN;
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		amrex::Real const Erad0 = initSumCons[RadSystem<problem_t>::radEnergy_index];
		Etot0 = Egas0 + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad0;
		amrex::Real const Erad = state_new_cc_[0].sum(RadSystem<problem_t>::radEnergy_index) * vol;
		Etot = Egas + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad;
	} else {
		Etot0 = Egas0;
		Etot = Egas;
	}

	amrex::Real const abs_err = (Etot - Etot0);
	amrex::Real const rel_err = abs_err / Etot0;

	amrex::Print() << "\nInitial gas+radiation energy = " << Etot0 << std::endl;
	amrex::Print() << "Final gas+radiation energy = " << Etot << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err << std::endl;
	amrex::Print() << std::endl;

	if (computeReferenceSolution_) {
		// compute reference solution
		const int ncomp = state_new_cc_[0].nComp();
		const int nghost = state_new_cc_[0].nGrow();
		amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, nghost);
		computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray());

		// compute error norm
		amrex::MultiFab residual(boxArray(0), DistributionMap(0), ncomp, nghost);
		amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, nghost);
		amrex::MultiFab::Saxpy(residual, -1., state_new_cc_[0], 0, 0, ncomp, nghost);

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

	// compute average number of radiation subcycles per timestep
	double const avg_rad_subcycles = static_cast<double>(radiationCellUpdates_) / static_cast<double>(cellUpdates_);
	amrex::Print() << "avg. num. of radiation subcycles = " << avg_rad_subcycles << std::endl;
	amrex::Print() << std::endl;
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								 amrex::Real dt_lev, int ncycle)
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
	std::swap(state_old_cc_[lev], state_new_cc_[lev]);

	// check hydro states before update (this can be caused by the flux register!)
	CHECK_HYDRO_STATES(state_old_cc_[lev]);

	// advance hydro
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
		advanceHydroAtLevelWithRetries(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	} else {
		// copy hydro vars from state_old_cc_ to state_new_cc_
		// (otherwise radiation update will be wrong!)
		amrex::MultiFab::Copy(state_new_cc_[lev], state_old_cc_[lev], 0, 0, ncompHydro_, 0);
	}

	// check hydro states after hydro update
	CHECK_HYDRO_STATES(state_new_cc_[lev]);
	
	// subcycle radiation
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		subcycleRadiationAtLevel(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	}

	// check hydro states after radiation update
	CHECK_HYDRO_STATES(state_new_cc_[lev]);

	// compute any operator-split terms here (user-defined)
	computeAfterLevelAdvance(lev, time, dt_lev, ncycle);

	// check hydro states after user work
	CHECK_HYDRO_STATES(state_new_cc_[lev]);

	// check state validity
	AMREX_ASSERT(!state_new_cc_[lev].contains_nan(0, state_new_cc_[lev].nComp()));
}

// fix-up any unphysical states created by AMR operations
// (e.g., caused by the flux register or from interpolation)
template <typename problem_t>
void RadhydroSimulation<problem_t>::FixupState(int lev)
{
	BL_PROFILE("RadhydroSimulation::FixupState()");

	// fix hydro state
	HydroSystem<problem_t>::EnforceDensityFloor(densityFloor_, state_new_cc_[lev]);
	// sync internal energy and total energy
	HydroSystem<problem_t>::SyncDualEnergy(state_new_cc_[lev]);
}

// Compute a new multifab 'mf' by copying in state from valid region and filling
// ghost cells
// NOTE: This has to be implemented here because PreInterpState and PostInterpState
// are implemented in this class (and must be *static* functions).
template <typename problem_t>
void RadhydroSimulation<problem_t>::FillPatch(int lev, amrex::Real time,
                                         amrex::MultiFab &mf, int icomp,
                                         int ncomp, FillPatchType fptype) {
  BL_PROFILE("AMRSimulation::FillPatch()");

  amrex::Vector<amrex::MultiFab *> cmf;
  amrex::Vector<amrex::MultiFab *> fmf;
  amrex::Vector<amrex::Real> ctime;
  amrex::Vector<amrex::Real> ftime;

  if (lev == 0) {
    // in this case, should return either state_new_cc_[lev] or state_old_cc_[lev]
    GetData(lev, time, fmf, ftime);
  } else {
    // in this case, should return either state_new_cc_[lev] or state_old_cc_[lev]
    GetData(lev, time, fmf, ftime);
    // returns old state, new state, or both depending on 'time'
    GetData(lev - 1, time, cmf, ctime);
  }

  FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, fptype,
		PreInterpState, PostInterpState);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::PreInterpState(amrex::MultiFab &mf, int scomp, int ncomp)
{
	BL_PROFILE("RadhydroSimulation::PreInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto Etot = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto kinetic_energy = (px*px + py*py + pz*pz) / (2.0*rho);

		// replace hydro total energy with specific internal energy (SIE)
		const auto e = (Etot - kinetic_energy) / rho;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = e;
	});
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::PostInterpState(amrex::MultiFab &mf, int scomp, int ncomp)
{
	BL_PROFILE("RadhydroSimulation::PostInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto e = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto Eint = rho * e;
		const auto kinetic_energy = (px*px + py*py + pz*pz) / (2.0*rho);

		// recompute hydro total energy from Eint + KE
		const auto Etot = Eint + kinetic_energy;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Etot;
	});
}

template <typename problem_t>
template <typename F>
auto RadhydroSimulation<problem_t>::computeAxisAlignedProfile(const int axis, F const &user_f)
	-> amrex::Gpu::HostVector<amrex::Real>
{
	// compute a 1D profile of user_f(i, j, k, state) along the given axis.
	BL_PROFILE("RadhydroSimulation::computeAxisAlignedProfile()");

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level+1);
	for(int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
	}

	// evaluate user_f on all levels
	for(int lev = 0; lev <= finest_level; ++lev) {
		for(amrex::MFIter iter(q[lev]); iter.isValid(); ++iter) {
			auto const &box = iter.validbox();
			auto const &state = state_new_cc_[lev].const_array(iter);
			auto const &result = q[lev].array(iter);
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				result(i, j, k) = user_f(i, j, k, state);
			});
		}
	}

	// average down
	for (int crse_lev = finest_level - 1; crse_lev >= 0; --crse_lev) {
  		amrex::average_down(q[crse_lev + 1], q[crse_lev],
			geom[crse_lev + 1], geom[crse_lev], 0, q[crse_lev].nComp(), refRatio(crse_lev));
  	}

	// compute 1D profile from level 0 multifab
	amrex::Box domain = geom[0].Domain();
	auto profile = amrex::sumToLine(q[0], 0, q[0].nComp(), domain, axis);

	// normalize profile
	amrex::Long numCells = domain.numPts() / domain.length(axis);
	for (int i = 0; i < profile.size(); ++i) {
		profile[i] /= static_cast<amrex::Real>(numCells);
	}

	return profile;
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceHydroAtLevelWithRetries(int lev, amrex::Real time,
							amrex::Real dt_lev,
							amrex::YAFluxRegister *fr_as_crse,
							amrex::YAFluxRegister *fr_as_fine)
{
	// timestep retries
	const int max_retries = 4;
	bool success = false;
	amrex::Real cur_time;

	for (int retry_count = 0; retry_count <= max_retries; ++retry_count) {
		// reduce timestep by a factor of 2^retry_count
		const int nsubsteps = std::pow(2, retry_count);
		const amrex::Real dt_step = dt_lev / nsubsteps;
		cur_time = time;

		if (retry_count > 0 && Verbose()) {
			amrex::Print() << "\t>> Re-trying hydro advance at level " << lev
						   << " with reduced timestep (nsubsteps = " << nsubsteps
						   << ", dt_new = " << dt_step << ")\n";
		}

		// create temporary multifab for old state
		amrex::MultiFab state_old_cc_tmp(grids[lev], dmap[lev], ncomp_cc_, nghost_);
		amrex::Copy(state_old_cc_tmp, state_old_cc_[lev], 0, 0, ncomp_cc_, nghost_);

		// copy flux register state into temporaries
		// ...
		amrex::MultiFab &fluxRegFineData = fr_as_fine->getFineData();
		amrex::MultiFab &fluxRegCrseData = fr_as_crse->getCrseData();

		// subcycle advanceHydroAtLevel, checking return value
		for (int substep = 0; substep < nsubsteps; ++substep) {
			if (substep > 0) {
				// since we are starting a new substep, we need to copy hydro state from
				//  the new state vector to old state vector
				amrex::Copy(state_old_cc_tmp, state_new_cc_[lev], 0, 0, ncompHydro_, nghost_);

				// TODO(ben): reset the flux registers
				// ...
			}

			success = advanceHydroAtLevel(state_old_cc_tmp, fr_as_crse, fr_as_fine, lev, time, dt_step);
			cur_time += dt_step;

			if (!success) {
				if (Verbose()) {
					amrex::Print() << "\t>> WARNING: Hydro advance failed on level " << lev << "\n";
				}
				break;
			}
		}

		if (success) {

			// we are done, do not attempt more retries
			break;
		}
	}
	AMREX_ALWAYS_ASSERT(success); // crash if we exceeded max_retries
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::isCflViolated(int lev, amrex::Real time,
							amrex::Real dt_actual) -> bool
{
	// check wheter dt_actual would violate CFL condition using the post-update hydro state

	// compute max signal speed
	amrex::Real max_signal = HydroSystem<problem_t>::maxSignalSpeedLocal(state_new_cc_[lev]);
	amrex::ParallelDescriptor::ReduceRealMax(max_signal);

	// compute dt_cfl
	auto dx = geom[lev].CellSizeArray();
	const amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	const amrex::Real dt_cfl = cflNumber_ * (dx_min / max_signal);

	// check whether dt_actual > dt_cfl (CFL violation)
	const amrex::Real max_factor = 1.1;
	const bool cflViolation = dt_actual > (max_factor * dt_cfl);
	if (cflViolation && Verbose()) {
		amrex::Print() << "\t>> CFL violation detected on level " << lev
					   << " with dt_lev = " << dt_actual
					   << " and dt_cfl = " << dt_cfl << "\n"
					   << "\t   max_signal = " << max_signal << "\n";
	}
	return cflViolation;
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp,
							amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
							int lev, amrex::Real time,
							amrex::Real dt_lev) -> bool
{
	BL_PROFILE("RadhydroSimulation::advanceHydroAtLevel()");

	amrex::Real fluxScaleFactor = NAN;
	if (integratorOrder_ == 2) {
		fluxScaleFactor = 0.5;
	} else if (integratorOrder_ == 1) {
		fluxScaleFactor = 1.0;
	}

	auto dx = geom[lev].CellSizeArray();

	// do Strang split source terms (first half-step)
	addStrangSplitSources(state_old_cc_tmp, lev, time, 0.5*dt_lev);

	// create temporary multifab for intermediate state
	amrex::MultiFab state_inter_cc_(grids[lev], dmap[lev], ncomp_cc_, nghost_);
	state_inter_cc_.setVal(0); // prevent assert in fillBoundaryConditions when radiation is enabled

	// Stage 1 of RK2-SSP
	{
		// update ghost zones [old timestep]
		fillBoundaryConditions(state_old_cc_tmp, state_old_cc_tmp, lev, time, PreInterpState, PostInterpState);

		// check state validity
		AMREX_ASSERT(!state_old_cc_tmp.contains_nan(0, state_old_cc_tmp.nComp()));
		AMREX_ASSERT(!state_old_cc_tmp.contains_nan()); // check ghost cells

		// advance all grids on local processor (Stage 1 of integrator)
		auto const &stateOld = state_old_cc_tmp;
		auto &stateNew = state_inter_cc_;
		auto [fluxArrays, faceVel] = computeHydroFluxes(stateOld, ncompHydro_, lev);

		amrex::MultiFab rhs(grids[lev], dmap[lev], ncompHydro_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 0);
		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, faceVel);
		HydroSystem<problem_t>::PredictStep(stateOld, stateNew, rhs, dt_lev, ncompHydro_, redoFlag);

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // just in case
		int ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-1] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
			}

			// replace fluxes around troubled cells with Godunov fluxes
			auto [FOfluxArrays, FOfaceVel] = computeFOHydroFluxes(stateOld, ncompHydro_, lev);
			replaceFluxes(fluxArrays, FOfluxArrays, redoFlag);
			replaceFluxes(faceVel, FOfaceVel, redoFlag); // needed for dual energy
			redoFlag.setVal(quokka::redoFlag::none); // reset redoFlag

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, faceVel);
			HydroSystem<problem_t>::PredictStep(stateOld, stateNew, rhs, dt_lev, ncompHydro_, redoFlag);
		
			amrex::Gpu::streamSynchronizeAll(); // just in case
			if (redoFlag.max(0) == quokka::redoFlag::redo) {
				// FOFC failed
				return false;
			}
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceDensityFloor(densityFloor_, stateNew);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateNew);
		}

		if (do_reflux == 1) {
			// increment flux registers
			incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
		}
	}
	amrex::Gpu::streamSynchronizeAll();

	// Stage 2 of RK2-SSP
	{
		// update ghost zones [intermediate stage stored in state_inter_cc_]
		fillBoundaryConditions(state_inter_cc_, state_inter_cc_, lev, time + dt_lev, PreInterpState,
					PostInterpState);

		// check intermediate state validity
		AMREX_ASSERT(!state_inter_cc_.contains_nan(0, state_inter_cc_.nComp()));
		AMREX_ASSERT(!state_inter_cc_.contains_nan()); // check ghost zones

		auto const &stateOld = state_old_cc_tmp;
		auto const &stateInter = state_inter_cc_;
		auto &stateFinal = state_new_cc_[lev];
		auto [fluxArrays, faceVel] = computeHydroFluxes(stateInter, ncompHydro_, lev);

		amrex::MultiFab rhs(grids[lev], dmap[lev], ncompHydro_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 0);
		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateInter, dx, faceVel);
		HydroSystem<problem_t>::AddFluxesRK2(stateFinal, stateOld, stateInter, rhs, dt_lev, ncompHydro_, redoFlag);

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // just in case
		int ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-2] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
			}

			// replace fluxes around troubled cells with Godunov fluxes
			auto [FOfluxArrays, FOfaceVel] = computeFOHydroFluxes(stateInter, ncompHydro_, lev);
			replaceFluxes(fluxArrays, FOfluxArrays, redoFlag);
			replaceFluxes(faceVel, FOfaceVel, redoFlag); // needed for dual energy
			redoFlag.setVal(quokka::redoFlag::none); // reset redoFlag

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateInter, dx, faceVel);
			HydroSystem<problem_t>::AddFluxesRK2(stateFinal, stateOld, stateInter, rhs, dt_lev, ncompHydro_, redoFlag);

			amrex::Gpu::streamSynchronizeAll(); // just in case
			if (redoFlag.max(0) == quokka::redoFlag::redo) {
				// FOFC failed
				return false;
			}
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceDensityFloor(densityFloor_, stateFinal);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateFinal);
		}

		if (do_reflux == 1) {
			// increment flux registers
			incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
		}
	}
	amrex::Gpu::streamSynchronizeAll();

	// do Strang split source terms (second half-step)
	addStrangSplitSources(state_new_cc_[lev], lev, time + dt_lev, 0.5*dt_lev);

	// check if we have violated the CFL timestep
	return !isCflViolated(lev, time, dt_lev);
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::replaceFluxes(
	std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,	amrex::iMultiFab &redoFlag)
{
	BL_PROFILE("RadhydroSimulation::replaceFluxes()");

	for(int idim = 0; idim < AMREX_SPACEDIM; ++idim) { // loop over dimension
		// ensure that flux arrays have the same number of components
		AMREX_ASSERT(fluxes[idim].nComp() == FOfluxes[idim].nComp());
		int ncomp = fluxes[idim].nComp();

		auto const &FOflux_arrs = FOfluxes[idim].const_arrays();
		auto const &redoFlag_arrs = redoFlag.const_arrays();
		auto flux_arrs = fluxes[idim].arrays();

		// By convention, the fluxes are defined on the left edge of each zone,
		// i.e. flux_(i) is the flux *into* zone i through the interface on the
		// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
		// the interface on the right of zone i.

		amrex::IntVect ng{AMREX_D_DECL(0,0,0)};

		amrex::ParallelFor(redoFlag, ng, ncomp,
			[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
			if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
				// replace fluxes with first-order ones at faces of cell (i,j,k)
				flux_arrs[bx](i, j, k, n) = FOflux_arrs[bx](i, j, k, n);

				if (idim == 0) { // x-dir fluxes
					flux_arrs[bx](i + 1, j, k, n) = FOflux_arrs[bx](i + 1, j, k, n);
				} else if (idim == 1) { // y-dir fluxes
					flux_arrs[bx](i, j + 1, k, n) = FOflux_arrs[bx](i, j + 1, k, n);
				} else if (idim == 2) { // z-dir fluxes
					flux_arrs[bx](i, j, k + 1, n) = FOflux_arrs[bx](i, j, k + 1, n);
				}
			}
		});
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes,
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes, const int srccomp, const int dstcomp)
{
	BL_PROFILE("RadhydroSimulation::addFluxArrays()");

	for(int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto const &srcflux = srcfluxes[idim];
		auto &dstflux = dstfluxes[idim];
		amrex::Add(dstflux, srcflux, srccomp, dstcomp, srcflux.nComp(), 0);
	}
}


template <typename problem_t>
auto RadhydroSimulation<problem_t>::expandFluxArrays(
    std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, const int nstartNew, const int ncompNew)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	BL_PROFILE("RadhydroSimulation::expandFluxArrays()");

	// This is needed because reflux arrays must have the same number of components as
	// state_new_cc_[lev]
	auto copyFlux = [nstartNew, ncompNew](amrex::FArrayBox const &oldFlux) {
		amrex::Box const &fluxRange = oldFlux.box();
		amrex::FArrayBox newFlux(fluxRange, ncompNew, amrex::The_Async_Arena());
		newFlux.setVal<amrex::RunOn::Device>(0.);
		// copy oldFlux (starting at 0) to newFlux (starting at nstart)
		AMREX_ASSERT(ncompNew >= oldFlux.nComp());
		newFlux.copy<amrex::RunOn::Device>(oldFlux, 0, nstartNew, oldFlux.nComp());
		return newFlux;
	};
	return {AMREX_D_DECL(copyFlux(fluxes[0]), copyFlux(fluxes[1]), copyFlux(fluxes[2]))};
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeHydroFluxes(
    amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	BL_PROFILE("RadhydroSimulation::computeHydroFluxes()");

	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int flatteningGhost = 2;
	const int reconstructRange = 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_);
	std::array<amrex::MultiFab, 3> flatCoefs;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < 3; ++idim) {
		flatCoefs[idim] = amrex::MultiFab(ba, dm, 1, flatteningGhost);
	}

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, 0);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, 0);
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar, primVar, nghost_);

	// compute flattening coefficients
	AMREX_D_TERM(HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X1>(
			primVar, flatCoefs[0], flatteningGhost);
		, HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X2>(
			primVar, flatCoefs[1], flatteningGhost);
		, HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X3>(
			primVar, flatCoefs[2], flatteningGhost); )

	// compute flux functions
	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(primVar, leftState[0], rightState[0], flux[0], facevel[0],
					flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructRange, nvars);
		     , hydroFluxFunction<FluxDir::X2>(primVar, leftState[1], rightState[1], flux[1], facevel[1],
					flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructRange, nvars);
		     , hydroFluxFunction<FluxDir::X3>(primVar, leftState[2], rightState[2], flux[2], facevel[2],
					flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructRange, nvars); )

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// return flux and face-centered velocities
	return std::make_pair(std::move(flux), std::move(facevel));
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::hydroFluxFunction(
    amrex::MultiFab const &primVar,
	amrex::MultiFab &leftState,
	amrex::MultiFab &rightState,
	amrex::MultiFab &flux,
	amrex::MultiFab &faceVel,
	amrex::MultiFab const &x1Flat,
	amrex::MultiFab const &x2Flat,
	amrex::MultiFab const &x3Flat,
	const int ng_reconstruct,
	const int nvars)
{
	if (reconstructionOrder_ == 3) {
		HydroSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 2) {
		HydroSystem<problem_t>::template ReconstructStatesPLM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 1) {
		HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(primVar, x1Flat, x2Flat, x3Flat, leftState, rightState, ng_reconstruct, nvars);

	// interface-centered kernel
	HydroSystem<problem_t>::template ComputeFluxes<DIR>(flux, faceVel, leftState, rightState, primVar);
}


template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeFOHydroFluxes(
    amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	BL_PROFILE("RadhydroSimulation::computeFOHydroFluxes()");

	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int flatteningGhost = 2;
	const int reconstructRange = 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, 0);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, 0);
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar, primVar, nghost_);

	// compute flux functions
	AMREX_D_TERM(hydroFOFluxFunction<FluxDir::X1>(primVar, leftState[0], rightState[0], flux[0], facevel[0],
					reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X2>(primVar, leftState[1], rightState[1], flux[1], facevel[1],
					reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X3>(primVar, leftState[2], rightState[2], flux[2], facevel[2],
					reconstructRange, nvars); )

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// return flux and face-centered velocities
	return std::make_pair(std::move(flux), std::move(facevel));
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::hydroFOFluxFunction(
    amrex::MultiFab const &primVar,
	amrex::MultiFab &leftState,
	amrex::MultiFab &rightState,
	amrex::MultiFab &flux,
	amrex::MultiFab &faceVel,
	const int ng_reconstruct,
	const int nvars)
{
	// donor-cell reconstruction
	HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);

	// interface-centered kernel
	HydroSystem<problem_t>::template ComputeFluxes<DIR>(flux, faceVel, leftState, rightState, primVar);
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
	int nsubSteps = 0;
	amrex::Real dt_radiation = NAN;

	if (Physics_Traits<problem_t>::is_hydro_enabled && !(constantDt_ > 0.)) {
		// adjust to get integer number of substeps
		nsubSteps = computeNumberOfRadiationSubsteps(lev, dt_lev_hydro);
		dt_radiation = dt_lev_hydro / static_cast<double>(nsubSteps);
	} else { // no hydro, or using constant dt (this is necessary for radiation test problems)
		dt_radiation = dt_lev_hydro;
		nsubSteps = 1;
	}

	if (Verbose() != 0) {
		amrex::Print() << "\tRadiation substeps: " << nsubSteps << "\tdt: " << dt_radiation
			       << "\n";
	}
	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps <= (maxSubsteps_+1));
	AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);

	// perform subcycle
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real time_subcycle = time;
	for (int i = 0; i < nsubSteps; ++i) {
		if (i > 0) {
			// since we are starting a new substep, we need to copy radiation state from
			// 	new state vector to old state vector
			// (this is not necessary for the i=0 substep because we have already swapped
			//  the full hydro+radiation state vectors at the beginning of the level advance)
			swapRadiationState(state_old_cc_[lev], state_new_cc_[lev]);
		}

		// advance hyperbolic radiation subsystem starting from state_old_cc_ to state_new_cc_
		advanceRadiationSubstepAtLevel(lev, time_subcycle, dt_radiation, i, nsubSteps,
							  fr_as_crse, fr_as_fine);

		// new radiation state is stored in state_new_cc_
		// new hydro state is stored in state_new_cc_ (always the case during radiation update)

		// matter-radiation exchange source terms
		for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateNew = state_new_cc_[lev].array(iter);
			auto const &prob_lo = geom[lev].ProbLoArray();
			auto const &prob_hi = geom[lev].ProbHiArray();
			// update state_new_cc_[lev] in place (updates both radiation and hydro vars)
			operatorSplitSourceTerms(stateNew, indexRange, time_subcycle, dt_radiation, 
									 dx, prob_lo, prob_hi);
		}

		// new hydro+radiation state is stored in state_new_cc_

		// update 'time_subcycle'
		time_subcycle += dt_radiation;

		// update cell update counter
		radiationCellUpdates_ += CountCells(lev); // keep track of number of cell updates
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceRadiationSubstepAtLevel(
    int lev, amrex::Real time, amrex::Real dt_radiation, int const iter_count, int const /*nsubsteps*/,
	amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine)
{
	if (Verbose()) {
		amrex::Print() << "\tsubstep " << iter_count << " t = " << time << std::endl;
	}

	// get cell sizes
	auto const &dx = geom[lev].CellSizeArray();

	// We use the RK2-SSP method here. It needs two registers: one to store the old timestep,
	// and another to store the intermediate stage (which is reused for the final stage).

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_cc_[lev], state_old_cc_[lev], lev, time,
			PreInterpState, PostInterpState);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_cc_[lev].const_array(iter);
		auto const &stateNew = state_new_cc_[lev].array(iter);
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
				expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev,
						0.5 * dt_radiation);
		}
	}

	// update ghost zones [intermediate stage stored in state_new_cc_]
	fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, time + dt_radiation,
			PreInterpState, PostInterpState);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_cc_[lev].const_array(iter);
		auto const &stateInter = state_new_cc_[lev].const_array(iter);
		auto const &stateNew = state_new_cc_[lev].array(iter);
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
				expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
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
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{
	amrex::FArrayBox radEnergySource(indexRange, 1,
					 amrex::The_Async_Arena()); // cell-centered scalar
	amrex::FArrayBox advectionFluxes(indexRange, 3,
					 amrex::The_Async_Arena()); // cell-centered vector

	radEnergySource.setVal<amrex::RunOn::Device>(0.);
	advectionFluxes.setVal<amrex::RunOn::Device>(0.);

	// cell-centered radiation energy source
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange,
						 dx, prob_lo, prob_hi, time + dt);

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

	if (radiationReconstructionOrder_ == 3) {
		// mixed interface/cell-centered kernel
		RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    	primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    	x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 2) {
		// PLM and donor cell are interface-centered kernels
		RadSystem<problem_t>::template ReconstructStatesPLM<DIR>(
	    	primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 1) {
		RadSystem<problem_t>::template ReconstructStatesConstant<DIR>(
			primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
			nvars);
	} else {
		amrex::Abort("Invalid reconstruction order for radiation variables! Aborting...");
	}

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(),
							  x1LeftState.array(), x1RightState.array(),
							  x1FluxRange, consState,
							  dx); // watch out for argument order!!
}

#endif // RADIATION_SIMULATION_HPP_