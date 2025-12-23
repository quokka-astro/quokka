#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file QuokkaSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include "AMReX_GpuContainers.H"
#include "AMReX_Tuple.H"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std
{
namespace filesystem = experimental::filesystem;
}
#endif
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArray.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#if AMREX_SPACEDIM == 3
#include "hydro_MacProjector.H"
#endif
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#ifdef AMREX_USE_ASCENT
#include "AMReX_Conduit_Blueprint.H"
#include <ascent.hpp>
#include <conduit_node.hpp>
#endif

#include "SimulationData.hpp"
#include "chemistry/Chemistry.hpp"
#include "cooling/ResampledCooling.hpp"
#include "dust/dust_system.hpp"
#include "eos.H"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation/radiation_system.hpp"
#include "simulation.hpp"
#include "turbulence/TurbulentDriving.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class QuokkaSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_cc_;
	using AMRSimulation<problem_t>::state_new_cc_;
	using AMRSimulation<problem_t>::max_signal_speed_;
	using AMRSimulation<problem_t>::state_old_fc_;
	using AMRSimulation<problem_t>::state_new_fc_;
	using AMRSimulation<problem_t>::TracerPC;
	using AMRSimulation<problem_t>::particleRegister_;

	using AMRSimulation<problem_t>::nghost_cc_;
	using AMRSimulation<problem_t>::nghost_fc_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::BCs_cc_;
	using AMRSimulation<problem_t>::BCs_fc_;
	using AMRSimulation<problem_t>::componentNames_cc_;
	using AMRSimulation<problem_t>::componentNames_fc_flat_;
	using AMRSimulation<problem_t>::componentNames_fc_;
	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::CustomPlotFileName;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::istep;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::emf_reg_;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::incrementEMFRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::tNew_;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::do_tracers;
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
	using AMRSimulation<problem_t>::Gconst_;

	using AMRSimulation<problem_t>::densityFloor_;
	using AMRSimulation<problem_t>::tempFloor_;

	using AMRSimulation<problem_t>::max_level;
	using AMRSimulation<problem_t>::n_error_buf;

	using AMRSimulation<problem_t>::sfh_interval_;
	using AMRSimulation<problem_t>::sfh_time_interval_;

#if AMREX_SPACEDIM == 3
	using AMRSimulation<problem_t>::luminosityTables_;
#endif // AMREX_SPACEDIM == 3

	SimulationData<problem_t> userData_;

	// Photoelectric heating
	bool use_sfh_based_pe_heating_ = false;
	std::string sfh_to_pe_heating_table_filename_;
	amrex::Real sf_area_kpc2_ = -1.0;		      // area of the star formation region in kpc^2 (for computing the PE heating rate)
	amrex::Real const_sfr_Msun_per_year_per_kpc2_ = -1.0; // constant star formation rate in Msun/year/kpc^2 (for computing the PE heating rate); will
							      // override real star formation rate from the simulation if non-negative
	quokka::PeHeatingTables<> peHeatingTables_;

	int enableCooling_ = 0;
	int enableChemistry_ = 0;
	int enableTurbulence_ = 0;
	Real max_density_allowed = std::numeric_limits<amrex::Real>::max();
	Real min_density_allowed = std::numeric_limits<amrex::Real>::min();

	quokka::ResampledCooling::resampled_tables resampledTables_;
	std::string coolingTableType_;
	std::string coolingTableFilename_;

	std::map<std::string, std::string> turbParams_;

	static constexpr int nvarTotal_cc_ = Physics_Indices<problem_t>::nvarTotal_cc;
	static constexpr int nvars_ = HydroSystem<problem_t>::nvar_;
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;
	static constexpr int n_mhd_vars_per_dim_ = MHDSystem<problem_t>::nvar_per_dim_; // mhd
	static constexpr int numDustVars_ = Physics_NumVars::numDustVarsPerGroup;	// number of dust variables for each dust group

	static constexpr bool is_particle_enabled = Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None;

	amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nDustGroups> dust_alpha_ = {};
	amrex::Real dust_omega_ = 1.0;

	amrex::Real radiationCflNumber_ = 0.3;
	int maxSubsteps_ = 10;				// maximum number of radiation subcycles per hydro step
	amrex::Real dustGasInteractionCoeff_ = 2.5e-34; // erg cm^3 s^−1 K^−3/2

	amrex::Real errorNorm_ = NAN;
	amrex::Real pressureFloor_ = 0.;

	bool use_wavespeed_correction_ = false;
	bool print_rad_counter_ = false;
	amrex::Real radiation_iteration_tolerance_ = 1e-11;    // tolerance for the Newton-Raphson iteration residuals
	amrex::Real radiation_iteration_tolerance_rel_ = -1.0; // tolerance for the relative change between two consecutive Newton-Raphson iterations

	bool projectInitialBField_ = false;
	bool updateInitialMagneticEnergy_ = true;

	int lowLevelDebuggingOutput_ = 0;	// 0 == do nothing; 1 == output intermediate multifabs used in hydro each timestep (ONLY USE FOR DEBUGGING)
	int integratorOrder_ = 2;		// 1 == forward Euler; 2 == RK2-SSP (default)
	int reconstructionOrder_ = 3;		// 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM (extrema-preserving)
	int radiationReconstructionOrder_ = 3;	// 1 == donor cell; 2 == PLM; 3 == PPM (default); 5 == xPPM
	int emfReconstructionOrder_ = 5;	// 1 == donor cell; 2 == PLM; 3 == PPM; 5 == xPPM (extrema-preserving, default)
	int useDualEnergy_ = 1;			// 0 == disabled; 1 == use auxiliary internal energy equation (default)
	int abortOnFofcFailure_ = 1;		// 0 == keep going, 1 == abort hydro advance if FOFC fails
	amrex::Real artificialViscosityK_ = 0.; // artificial viscosity coefficient (default == None)
	// number of ghost cells for face velocity computation (default == 2)
	// we now need 3 total to accommodate the higher-order reconstruction in computeEMF
	int nghost_vel_ = Physics_Traits<problem_t>::is_mhd_enabled ? 3 : 2;

	EMFComputeScheme emfComputingScheme_ = EMFComputeScheme::FelkerStone2017;
	EMFAvgScheme emfAveragingScheme_ = EMFAvgScheme::LondrilloDelZanna2004; // method to use to average EMF at edges

	amrex::Long radiationCellUpdates_ = 0; // total number of radiation cell-updates
	std::unique_ptr<quokka::turbulence::turbulentDriving<problem_t>> td;

	// member functions
	explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : AMRSimulation<problem_t>(BCs_cc, BCs_fc)
	{
		initialize();
	}

	explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) { initialize(); }

	void initialize()
	{
		static_assert(!(Physics_Traits<problem_t>::is_mhd_enabled && Physics_Traits<problem_t>::is_radiation_enabled),
			      "MHD + Radiation is not supported yet.");
#if (AMREX_SPACEDIM != 3)
		static_assert(!(Physics_Traits<problem_t>::is_mhd_enabled), "MHD is only supported in 3D.");
#endif // (AMREX_SPACEDIM != 3)

		defineComponentNames();
		defineDefaultPlotfileVariables();
		// read in runtime parameters
		readParmParse();
		// set gamma
		amrex::ParmParse eos("eos");
		eos.add("eos_gamma", quokka::EOS_Traits<problem_t>::gamma);
		// initialize Microphysics params
		init_extern_parameters();
		// initialize Microphysics EOS
		amrex::Real small_temp = 1e-10;
		amrex::Real small_dens = 1e-100;
		eos_init(small_temp, small_dens);
	}

	[[nodiscard]] static auto getScalarVariableNames() -> std::vector<std::string>;
	void defineComponentNames();
	void defineDefaultPlotfileVariables();
	void readParmParse();
	void rereadRuntimeParameters(); // Re-read parameters to ensure runtime values override compile-time settings

	void checkHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, char const *file, int line);
	void computeMaxSignalLocal(int level) override;
	void printCellProperties(int lev, amrex::IntVect const &index) override;
	void preCalculateInitialConditions() override;
	void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;
	void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;
	void postInitialization() override;

	// Optionally project already-initialised face-centred magnetic fields onto a divergence-free space
	void projectFaceCenteredMagneticField();
	void updateInitialMagneticEnergyFromFaceField();
	void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;
	void createInitialRadParticles() override;
#if AMREX_SPACEDIM == 3
	void createInitialCICParticles() override;
	void createInitialCICRadParticles() override;
	void createInitialStochasticStellarPopParticles() override;
	void createInitialSinkParticles() override;
	void createInitialTestParticles() override;
#endif // AMREX_SPACEDIM == 3
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) override;
	void computeBeforeTimestep() override;
	void computeAfterTimestep() override;
	void computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/);
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);
	void computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction dir);
	auto computeErrorNorm(bool use_rel_err = true) -> amrex::Real;
	auto computeComponentErrors() -> std::vector<std::tuple<std::string, amrex::Real, amrex::Real>>;
	void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames,
						int lev, int interval) override;

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;

	// compute projected vars
	[[nodiscard]] auto ComputeProjections(amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> override;

	// compute statistics
	auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;

	// fix-up states
	void FixupState(int level) override;

	// implement FillPatch function
	void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
		       FillPatchType fptype) override;

	// functions to operate on state vector before/after interpolating between levels
	static void PreInterpState(amrex::MultiFab &mf, int scomp, int ncomp);
	static void PostInterpState(amrex::MultiFab &mf, int scomp, int ncomp);

	// compute axis-aligned 1D profile of user_f(x, y, z)
	template <typename F> auto computeAxisAlignedProfile(int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	// fill rhs for Poisson solve
	void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;

	void print_multifab_fc(amrex::MultiFab &mf, std::string const &name, int lev, int idim);

	// add gravitational acceleration to hydro state
	void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;

	void addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes, int srccomp,
			   int dstcomp);

	auto expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, int nstartNew, int ncompNew)
	    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	void printCoordinates(int lev, const amrex::IntVect &cell_idx);

	void advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
					    amrex::EdgeFluxRegister *emf_as_crse = nullptr, amrex::EdgeFluxRegister *emf_as_fine = nullptr);

	auto advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_old_fc_tmp,
				 amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse,
				 amrex::EdgeFluxRegister *emf_as_fine, int lev, amrex::Real time, amrex::Real dt_lev) -> bool;

	void addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt_lev);
	auto addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt_lev) -> bool;

	auto computePhotoelectricHeatingRate(Real current_time) -> amrex::Real;

	auto isCflViolated(int lev, amrex::Real time, amrex::Real dt_actual) -> bool;

	// radiation subcycle
	void swapRadiationState(amrex::MultiFab &stateOld_cc, amrex::MultiFab const &stateNew_cc);
	auto computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int;
	void advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::FluxRegister *fr_as_crse,
					  amrex::FluxRegister *fr_as_fine);
	void advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::FluxRegister *fr_as_crse,
					 amrex::FluxRegister *fr_as_fine);

	void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine);

	auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, int nvars,
				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
	    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;

	auto computeHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int lev)
	    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
			  std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	auto computeFOHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int lev)
	    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
			  std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	template <FluxDir DIR>
	void computeCCPerpBfieldComps(amrex::MultiFab &cc_bfield_perp_comps_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc) const;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::MultiFab &primVar, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState,
			       amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel,
			       amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, amrex::MultiFab const &x1Flat,
			       amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, int ng_reconstruct_total, int nvars);

	template <FluxDir DIR>
	void hydroFOFluxFunction(amrex::MultiFab &primVar, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState,
				 amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel,
				 amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &x1ConsVar_fc_mf, int ng_reconstruct_total,
				 int nvars);

	void replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,
			   amrex::iMultiFab &redoFlag);

	void replaceEMFs(std::array<amrex::MultiFab, AMREX_SPACEDIM> &emf_components, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FO_emf_components,
			 amrex::iMultiFab &redoFlag);

	// void PrintRadEnergySource(amrex::MultiFab const &radEnergySource);
};

template <typename problem_t> void QuokkaSimulation<problem_t>::defineComponentNames()
{

	// cell-centred
	// add hydro state variables
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		std::vector<std::string> hydroNames = {"gasDensity", "x-GasMomentum", "y-GasMomentum", "z-GasMomentum", "gasEnergy", "gasInternalEnergy"};
		componentNames_cc_.insert(componentNames_cc_.end(), hydroNames.begin(), hydroNames.end());
	}
	// add passive scalar variables
	if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
		std::vector<std::string> scalarNames = getScalarVariableNames();
		componentNames_cc_.insert(componentNames_cc_.end(), scalarNames.begin(), scalarNames.end());
	}
	// add dust state variables
	if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
		std::vector<std::string> dustNames = {};
		for (int i = 0; i < Physics_Traits<problem_t>::nDustGroups; ++i) {
			dustNames.push_back("dustDensity-Group" + std::to_string(i));
			dustNames.push_back("x-DustMomentum-Group" + std::to_string(i));
			dustNames.push_back("y-DustMomentum-Group" + std::to_string(i));
			dustNames.push_back("z-DustMomentum-Group" + std::to_string(i));
		}
		componentNames_cc_.insert(componentNames_cc_.end(), dustNames.begin(), dustNames.end());
	}
	// add radiation state variables
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		std::vector<std::string> radNames = {};
		for (int i = 0; i < Physics_Traits<problem_t>::nGroups; ++i) {
			radNames.push_back("radEnergy-Group" + std::to_string(i));
			radNames.push_back("x-RadFlux-Group" + std::to_string(i));
			radNames.push_back("y-RadFlux-Group" + std::to_string(i));
			radNames.push_back("z-RadFlux-Group" + std::to_string(i));
		}
		componentNames_cc_.insert(componentNames_cc_.end(), radNames.begin(), radNames.end());
	}

	// face-centred

	// add mhd state variables
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
			componentNames_fc_flat_.push_back({quokka::face_dir_str[idim] + "-BField"});
			componentNames_fc_[idim].push_back({quokka::face_dir_str[idim] + "-BField"});
		}
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::defineDefaultPlotfileVariables()
{
	// Initialize plotfileVarsToInclude_cc_ with all cell-centered variables
	this->plotfileVarsToInclude_cc_.insert(this->plotfileVarsToInclude_cc_.end(), this->componentNames_cc_.begin(), this->componentNames_cc_.end());

	// Add all face-centered variables
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int icomp = 0; icomp < Physics_Indices<problem_t>::nvarTotal_fc; ++icomp) {
			const std::string &varname = this->componentNames_fc_flat_[icomp];
			this->plotfileVarsToInclude_cc_.push_back(varname);
		}
	}

	// Add all derived variables
	this->plotfileVarsToInclude_cc_.insert(this->plotfileVarsToInclude_cc_.end(), this->derivedNames_.begin(), this->derivedNames_.end());

	// Detect name collisions and abort if any are found
	std::set<std::string> seen_names;
	for (const std::string &varname : this->plotfileVarsToInclude_cc_) {
		if (!seen_names.insert(varname).second) {
			amrex::Abort("Duplicate variable name '" + varname +
				     "' found in plotfile variables list. "
				     "This indicates a naming collision between cell-centered, face-centered, or derived variables.");
		}
	}
}

// initialize metadata
template <typename problem_t> void AMRSimulation<problem_t>::initializeSimulationMetadata()
{
	if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
		// if unit system is CONSTANTS, the units are not well defined unless all four constants, G, k_B, c, and a_rad, are defined. However, in a hydro
		// simulation, only k_B is defined. In a radiation-hydrodynamics simulation, only k_B, c, and a_rad are defined. Besides, CONSTANTS is only used
		// for testing purposes, so we don't care about the units in that case.
		simulationMetadata_["units"]["unit_length"] = NAN;
		simulationMetadata_["units"]["unit_mass"] = NAN;
		simulationMetadata_["units"]["unit_time"] = NAN;
		simulationMetadata_["units"]["unit_temperature"] = NAN;

		// constants
		simulationMetadata_["constants"]["k_B"] = Physics_Traits<problem_t>::boltzmann_constant;
		simulationMetadata_["constants"]["G"] = Physics_Traits<problem_t>::gravitational_constant;
		if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			simulationMetadata_["constants"]["c"] = Physics_Traits<problem_t>::c_light;
			simulationMetadata_["constants"]["c_hat"] = Physics_Traits<problem_t>::c_light * RadSystem_Traits<problem_t>::c_hat_over_c;
			simulationMetadata_["constants"]["a_rad"] = Physics_Traits<problem_t>::radiation_constant;
		}
	} else {
		// units
		simulationMetadata_["units"]["unit_length"] = unit_length;
		simulationMetadata_["units"]["unit_mass"] = unit_mass;
		simulationMetadata_["units"]["unit_time"] = unit_time;
		simulationMetadata_["units"]["unit_temperature"] = unit_temperature;

		// constants
		double k_B = NAN;
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			k_B = C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// Have to do a conversion because EOS class is not accessible here
			k_B = C::k_B /
			      (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
			       (Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
		simulationMetadata_["constants"]["k_B"] = k_B;
		simulationMetadata_["constants"]["G"] = Gconst_;
		if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			simulationMetadata_["constants"]["c"] = RadSystem<problem_t>::c_light_;
			simulationMetadata_["constants"]["c_hat"] = RadSystem<problem_t>::c_hat_;
			simulationMetadata_["constants"]["a_rad"] = RadSystem<problem_t>::radiation_constant_;
		}
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::getScalarVariableNames() -> std::vector<std::string>
{
	// return vector of names for the passive scalars
	// this can be specialized by the user to provide more descriptive names
	// (these names are used to label the variables in the plotfiles)

	std::vector<std::string> names;
	int nscalars = HydroSystem<problem_t>::nscalars_;
	names.reserve(nscalars);
	for (int n = 0; n < nscalars; ++n) {
		// write string 'scalar_1', etc.
		names.push_back(fmt::format("scalar_{}", n));
	}
	return names;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::readParmParse()
{
	// set hydro runtime parameters
	{
		amrex::ParmParse const hpp("hydro");
		hpp.query("low_level_debugging_output", lowLevelDebuggingOutput_);
		hpp.query("rk_integrator_order", integratorOrder_);
		hpp.query("reconstruction_order", reconstructionOrder_);
		hpp.query("use_dual_energy", useDualEnergy_);
		hpp.query("abort_on_fofc_failure", abortOnFofcFailure_);
		hpp.query("artificial_viscosity_coefficient", artificialViscosityK_);
	}

	// set MHD runtime parameters
	{
		amrex::ParmParse const hpp("mhd");
		hpp.query("emf_reconstruction_order", emfReconstructionOrder_);
		hpp.query("emf_compute_scheme", emfComputingScheme_);
		hpp.query("emf_averaging_scheme", emfAveragingScheme_);
		hpp.query("project_initial_b_field", projectInitialBField_);
		hpp.query("update_initial_b_energy", updateInitialMagneticEnergy_);
	}

	// set cooling runtime parameters
	{
		amrex::ParmParse const hpp("cooling");
		int alwaysReadTables = 0;
		hpp.query("enabled", enableCooling_);
		hpp.query("cooling_table_type", coolingTableType_);
		hpp.query("read_tables_even_if_disabled", alwaysReadTables);
		if (coolingTableType_.empty()) {
			coolingTableType_ = "resampled";
		}
		if ((enableCooling_ == 1) || (alwaysReadTables == 1)) {
			hpp.query("hdf5_data_file", coolingTableFilename_);
			if (coolingTableType_ == "resampled") {
				// read resampled cooling tables
				amrex::Print() << "Reading resampled cooling tables...\n";
				quokka::ResampledCooling::readResampledData(coolingTableFilename_, resampledTables_);
			} else {
				amrex::Abort("Invalid cooling table type! Only 'resampled' is supported.");
			}
		}
	}

	// set turbulence runtime parameters
	{
		amrex::ParmParse const hpp("turbulence");
		hpp.query("enabled", enableTurbulence_);
		hpp.query("length", turbParams_["length"]);
		hpp.query("target_vdisp", turbParams_["target_vdisp"]);
		hpp.query("ampl_factor", turbParams_["ampl_factor"]);
		hpp.query("ampl_auto_adjust", turbParams_["ampl_auto_adjust"]);
		hpp.query("k_driv", turbParams_["k_driv"]);
		hpp.query("k_min", turbParams_["k_min"]);
		hpp.query("k_max", turbParams_["k_max"]);
		hpp.query("sol_weight", turbParams_["sol_weight"]);
		hpp.query("spect_form", turbParams_["spect_form"]);
		hpp.query("power_law_exp", turbParams_["power_law_exp"]);
		hpp.query("angles_exp", turbParams_["angles_exp"]);
		hpp.query("random_seed", turbParams_["random_seed"]);
		hpp.query("nsteps_per_t_turb", turbParams_["nsteps_per_t_turb"]);
		turbParams_["ndim"] = std::to_string(AMREX_SPACEDIM);

		if (enableTurbulence_ == 1) {
			td = std::make_unique<quokka::turbulence::turbulentDriving<problem_t>>(turbParams_);
		}
	}

	// set photoelectric heating runtime parameters
	{
		amrex::ParmParse const pp;
		pp.query("use_sfh_based_pe_heating", use_sfh_based_pe_heating_);
		pp.query("sfh_to_pe_heating_table", sfh_to_pe_heating_table_filename_);
		pp.query("sf_area_kpc2", sf_area_kpc2_);
		pp.query("const_sfr_Msun_per_year_per_kpc2", const_sfr_Msun_per_year_per_kpc2_);
		// It's allowed to turn on sfh and not turn on use_sfh_based_pe_heating, but the opposite is not allowed.
		if (use_sfh_based_pe_heating_) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    !sfh_to_pe_heating_table_filename_.empty(),
			    "When use_sfh_based_pe_heating is set to true, a PE heating table must be specified via sfh_to_pe_heating_table");
			if (const_sfr_Msun_per_year_per_kpc2_ < 0.0) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE((sfh_interval_ > 0) || (sfh_time_interval_ > 0),
								 "When use_sfh_based_pe_heating is set to true, star formation history must be turned on by "
								 "specifying sfh_interval or sfh_time_interval");
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sf_area_kpc2_ > 0.0,
								 "When use_sfh_based_pe_heating is set to true, sf_area_kpc2 must be set to a positive value");
			} else {
				// Using a constant star formation rate does not require recording the star formation history.
			}
		} else {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
			    sfh_to_pe_heating_table_filename_.empty(),
			    "use_sfh_based_pe_heating is set to false but sfh_to_pe_heating_table is specified. This indicates a misconfiguration.");
		}
	}

	// Load PE heating table if specified
	if (use_sfh_based_pe_heating_) {
		amrex::Print() << "Loading PE heating table from: " << sfh_to_pe_heating_table_filename_ << "\n";

		// Use linear spacing for PE heating values (can be changed if needed)
		peHeatingTables_.pe_heating = quokka::DataTable<1, 1>::CSVReader(sfh_to_pe_heating_table_filename_, quokka::SpacingType::fast_log);

		amrex::Print() << "PE heating table loaded successfully.\n";
		amrex::Print() << fmt::format("\tTable dimension: {}\n", peHeatingTables_.pe_heating.size(0));
		amrex::Print() << fmt::format("\tNumber of outputs: {}\n", peHeatingTables_.pe_heating.num_outputs());

		// Validate table metadata matches expected hardcoded values
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(peHeatingTables_.pe_heating.input_name(0) == "age",
						 fmt::format("PE heating table input must be 'age', got '{}'", peHeatingTables_.pe_heating.input_name(0)));
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    peHeatingTables_.pe_heating.input_unit(0) == "year",
		    fmt::format("PE heating table input unit must be 'year', got '{}'", peHeatingTables_.pe_heating.input_unit(0)));
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    peHeatingTables_.pe_heating.output_unit(0) == "erg/s/Msun",
		    fmt::format("PE heating table output unit must be 'erg/s/Msun', got '{}'", peHeatingTables_.pe_heating.output_unit(0)));

		// Set global pointer for access from particle functions
		quokka::g_pe_heating_tables_ptr<> = &peHeatingTables_;
	}

#ifdef CHEMISTRY
	// set chemistry runtime parameters
	{
		amrex::ParmParse const hpp("chemistry");
		hpp.query("enabled", enableChemistry_);
		hpp.query("max_density_allowed", max_density_allowed); // chemistry is not accurate for densities > 3e-6
		hpp.query("min_density_allowed", min_density_allowed); // don't do chemistry in cells with densities below the minimum density specified
	}
#endif

	// set dust runtime parameters
	{
		for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
			dust_alpha_[g] = 0.0;
		}
		amrex::ParmParse const dpp("dust");
		std::vector<amrex::Real> alpha_vec;
		if (dpp.queryarr("alpha", alpha_vec) != 0) {
			AMREX_ASSERT(alpha_vec.size() == Physics_Traits<problem_t>::nDustGroups);
			for (int g = 0; g < Physics_Traits<problem_t>::nDustGroups; ++g) {
				dust_alpha_[g] = alpha_vec[g];
			}
		}
		dpp.query("omega", dust_omega_);
	}

	// set radiation runtime parameters
	{
		amrex::ParmParse const rpp("radiation");
		rpp.query("reconstruction_order", radiationReconstructionOrder_);
		rpp.query("cfl", radiationCflNumber_);
		rpp.query("dust_gas_interaction_coeff", dustGasInteractionCoeff_);
		rpp.query("print_iteration_counts", print_rad_counter_);
		rpp.query("iteration_tolerance", radiation_iteration_tolerance_);
		rpp.query("iteration_tolerance_rel", radiation_iteration_tolerance_rel_);
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::rereadRuntimeParameters()
{
	// Re-read runtime parameters to ensure they override any compile-time settings
	// This is called at the beginning of evolve() to ensure user input takes precedence

	// Call parent class rereadRuntimeParameters
	AMRSimulation<problem_t>::rereadRuntimeParameters();

	// Re-read QuokkaSimulation-specific parameters
	readParmParse();

	// Re-read particle parameters
	quokka::particleParmParse();
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int
{
	// compute radiation timestep
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real c_hat = RadSystem<problem_t>::c_hat_;
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat);
	int const nsubSteps = std::ceil(dt_lev_hydro / dtrad_tmp);
	return nsubSteps;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	const BL_PROFILE("QuokkaSimulation::computeMaxSignalLocal()");

	// hydro: loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew_cc = state_new_cc_[level].const_array(iter);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> stateNew_fc;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < 3; ++idim) {
				stateNew_fc[idim] = state_new_fc_[level][idim].const_array(iter);
			}
		}
		auto const &maxSignal = max_signal_speed_[level].array(iter);

		if constexpr (Physics_Traits<problem_t>::is_hydro_enabled && !(Physics_Traits<problem_t>::is_radiation_enabled)) {
			// hydro/mhd
			HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew_cc, stateNew_fc, maxSignal, indexRange);
		} else if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			// radiation hydro/mhd, or radiation only
			RadSystem<problem_t>::ComputeMaxSignalSpeed(stateNew_cc, maxSignal, indexRange);
			if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
				auto maxSignalHydroFAB = amrex::FArrayBox(indexRange);
				auto const &maxSignalHydro = maxSignalHydroFAB.array();
				HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew_cc, stateNew_fc, maxSignalHydro, indexRange);
				const int maxSubsteps = maxSubsteps_;
				// ensure that we use the smaller of the two timesteps
				amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					amrex::Real const maxSignalRadiation = maxSignal(i, j, k) / static_cast<double>(maxSubsteps);
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

template <typename problem_t> void QuokkaSimulation<problem_t>::printCellProperties(int lev, amrex::IntVect const &index)
{
	// print density, velocity magnitude, temperature, adiabatic sound speed
	amrex::Vector<amrex::Real> cell_values = amrex::get_cell_data(state_new_cc_[lev], index);

	// cell_values is *only* filled on the MPI rank that holds the box with this cell
	// (NOTE: for Cray MPICH, standard output is NOT ordered with respect to different ranks.)
	if (!cell_values.empty()) {
		const amrex::Real rho = cell_values[HydroSystem<problem_t>::density_index];
		const amrex::Real px1 = cell_values[HydroSystem<problem_t>::x1Momentum_index];
		const amrex::Real px2 = cell_values[HydroSystem<problem_t>::x2Momentum_index];
		const amrex::Real px3 = cell_values[HydroSystem<problem_t>::x3Momentum_index];
		const amrex::Real Etot = cell_values[HydroSystem<problem_t>::energy_index];
		const amrex::Real vx1 = px1 / rho;
		const amrex::Real vx2 = px2 / rho;
		const amrex::Real vx3 = px3 / rho;
		const amrex::Real vsq = (vx1 * vx1) + (vx2 * vx2) + (vx3 * vx3);
		const amrex::Real vel_mag = std::sqrt(vsq);
		const amrex::Real Ekin = 0.5 * rho * vsq;
		const amrex::Real Eint = Etot - Ekin;
		const amrex::Real P = quokka::EOS<problem_t>::ComputePressure(rho, Eint);
		const amrex::Real cs = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P);

		amrex::AllPrint() << fmt::format("...[level {}] \tcell density = {:e}, |v| = {:e}, cs = {:e}\n", lev, rho, vel_mag, cs);
	}
}

#if !defined(NDEBUG)
#define CHECK_HYDRO_STATES(mf, mf_fc) checkHydroStates(mf, mf_fc, __FILE__, __LINE__)
#else
#define CHECK_HYDRO_STATES(mf, mf_fc)
#endif

template <typename problem_t>
void QuokkaSimulation<problem_t>::checkHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, char const *file, int line)
{
	const BL_PROFILE("QuokkaSimulation::checkHydroStates()");

	bool validStates = HydroSystem<problem_t>::CheckStatesValid(mf, mf_fc);
	amrex::ParallelDescriptor::ReduceBoolAnd(validStates);

	if (!validStates) {
		amrex::Print() << "Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")\n";
		amrex::Print() << "Writing checkpoint for debugging...\n";
		amrex::MFIter::allowMultipleMFIters(1);
		WriteCheckpointFile();
		amrex::Abort("Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")");
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::preCalculateInitialConditions()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if face-centered vars are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialRadParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialRadParticles()");
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if Rad_particles are used
}

#if AMREX_SPACEDIM == 3

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialCICParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialCICParticles()");
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if CIC_particles are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialCICRadParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialCICRadParticles()");
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if CICRad_particles are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialStochasticStellarPopParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialStochasticStellarPopParticles()");
	// Optional implementation
	// StochasticStellarPop particles are created on-the-fly from fluid cells. The user can optionally implement this function to create particles at the
	// beginning of the simulation.
	// note: an implementation is only effective if StochasticStellarPop_particles are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialSinkParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialSinkParticles()");
	// Optional implementation
	// Sink particles are created on-the-fly from fluid cells. The user can optionally implement this function to create particles at the
	// beginning of the simulation.
	// note: an implementation is only effective if Sink_particles are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialTestParticles()
{
	const BL_PROFILE("QuokkaSimulation::createInitialTestParticles()");
	// Optional implementation
	// Test particles are created on-the-fly from fluid cells. The user can optionally implement this function to create particles at the
	// beginning of the simulation.
	// note: an implementation is only effective if Test_particles are used
}
#endif // AMREX_SPACEDIM == 3

template <typename problem_t> void QuokkaSimulation<problem_t>::computeBeforeTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	// user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt)
{
	// user should implement
	// (when Strang splitting is enabled, dt is actually 0.5*dt_lev)
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computePhotoelectricHeatingRate(amrex::Real current_time) -> amrex::Real
{
	amrex::Real heating_rate = 0.0;

	// Check if PE heating tables are initialized
	// Note that this function is always called as long as cooling is turned on, so it is okay if g_pe_heating_tables_ptr is null
	if (quokka::g_pe_heating_tables_ptr<> == nullptr || !quokka::g_pe_heating_tables_ptr<>->is_initialized()) {
		return heating_rate; // Return 0 if tables not loaded
	}

	// Get GPU-friendly const tables
	auto const gpu_tables = quokka::g_pe_heating_tables_ptr<>->const_tables();

	if (const_sfr_Msun_per_year_per_kpc2_ > 0.0) {
		// Constant star formation rate
		heating_rate = quokka::PeHeatingFromConstSfr(const_sfr_Msun_per_year_per_kpc2_, gpu_tables);
	} else {
		// Real star formation history
		heating_rate = particleRegister_.computePhotoelectricHeatingRate(current_time, gpu_tables, sf_area_kpc2_);
	}

	return heating_rate;
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) -> bool
{
	// start by assuming cooling integrator is successful.
	bool cool_success = true;
	if (enableCooling_ == 1) {
		if (coolingTableType_.empty()) {
			coolingTableType_ = "resampled";
		}
		if (coolingTableType_ == "resampled") {
			const Real const_heating_rate_per_H = computePhotoelectricHeatingRate(time); // unit: erg/s/H
			cool_success = quokka::ResampledCooling::computeCooling<problem_t>(state, dt, resampledTables_, tempFloor_, const_heating_rate_per_H);
		} else {
			amrex::Abort("Invalid cooling table type! Only 'resampled' is supported.");
		}
	}

	// start by assuming chemistry burn is successful.
	bool burn_success = true; // NOLINT
#ifdef CHEMISTRY
	if (enableChemistry_ == 1) {
		// compute chemistry
		burn_success = quokka::chemistry::computeChemistry<problem_t>(state, dt, max_density_allowed, min_density_allowed);
	}
#endif

	if (enableTurbulence_ == 1) {
		auto const &cellSizes = geom[lev].CellSizeArray();
		td->applyDriving(state, time, dt, cellSizes);
	}
	if constexpr (Physics_Traits<problem_t>::is_dust_enabled) {
		DustSystem<problem_t>::computeDustDrag(state, dt, dust_omega_, dust_alpha_);
	}

	// compute user-specified sources
	addStrangSplitSources(state, lev, time, dt);

	return (burn_success && cool_success);
}

template <typename problem_t> void QuokkaSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const
{
	// compute derived variables and save in 'mf' -- user should implement
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::ComputeProjections(const amrex::Direction /*dir*/) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute projections and return as unordered_map -- user should implement
	return std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>{};
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	// compute statistics and return a std::map<std::string, amrex::Real> -- user should implement
	// IMPORTANT: the user is responsible for performing any necessary MPI reductions before returning
	return std::map<std::string, amrex::Real>{};
}

template <typename problem_t> void QuokkaSimulation<problem_t>::refineGrid(int /*lev*/, amrex::TagBoxArray & /*tags*/, amrex::Real /*time*/, int /*ngrow*/)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void QuokkaSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow)
{
	// call user-defined RefineGrid to set tags
	refineGrid(lev, tags, time, ngrow);

#if AMREX_SPACEDIM == 3
	// refine grids around particles
	if (lev < max_level) {
		particleRegister_.refineGridsAroundParticles(lev, tags, time, ngrow, n_error_buf[lev]);
	}
#endif
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	// user should implement
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	// user should implement
}

template <typename problem_t> void QuokkaSimulation<problem_t>::print_multifab_fc(amrex::MultiFab &mf, std::string const & /*name*/, int /*lev*/, int idim)
{
	amrex::Print() << "\nDDEBUG fc at direction " << idim << "\n";
	auto mf_fc = mf.arrays();
	amrex::ParallelFor(
	    mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { printf("%f\n", mf_fc[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex)); });
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computeComponentErrors() -> std::vector<std::tuple<std::string, amrex::Real, amrex::Real>>
{
	// returns a vector of tuples: (component name, absolute error, relative error)
	// absolute error is normalized by number of cells
	// relative error is NAN if reference norm is zero

	std::vector<std::tuple<std::string, amrex::Real, amrex::Real>> comp_errors{};

	// Compute cell-centered errors
	const int ncomp = state_new_cc_[0].nComp();
	amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, 0);
	computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray());
	bool has_valid_data = true;
	for (int icomp = 0; icomp < ncomp; ++icomp) {
		if (!std::isfinite(state_ref_level0.norm0(icomp))) {
			has_valid_data = false;
			break;
		}
	}

	if (!has_valid_data) {
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "\nWARNING: Cell-centered reference solution was not properly set.\n";
			amrex::Print() << "Please implement computeReferenceSolution() for your problem.\n\n";
		}
		return comp_errors; // Return empty vector
	}

	amrex::MultiFab residual(state_ref_level0.boxArray(), state_ref_level0.DistributionMap(), ncomp, 0);
	amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, 0);
	amrex::MultiFab::Saxpy(residual, -1., state_new_cc_[0], 0, 0, ncomp, 0);

	const auto n_cells = static_cast<amrex::Real>(residual.boxArray().numPts());

	for (int icomp = 0; icomp < ncomp; ++icomp) {
		const amrex::Real abs_err = residual.norm1(icomp) / n_cells;
		const amrex::Real ref_norm = state_ref_level0.norm1(icomp) / n_cells;

		amrex::Real rel_err = NAN;
		if (ref_norm > 0.0) {
			rel_err = abs_err / ref_norm;
		}

		comp_errors.emplace_back(componentNames_cc_[icomp], abs_err, rel_err);
	}

	// Compute face-centered errors (if MHD is enabled)
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			const int ncomp_fc = state_new_fc_[0][idim].nComp();
			amrex::BoxArray ba_fc = amrex::convert(boxArray(0), amrex::IntVect::TheDimensionVector(idim));
			// Shrink by 1 in the face-normal direction (idim) so that identical cell numbers are used for error calculation
			ba_fc.growHi(idim, -1);
			amrex::MultiFab state_ref_fc_level0(ba_fc, DistributionMap(0), ncomp_fc, 0);
			amrex::MultiFab state_new_fc_shrunk(ba_fc, DistributionMap(0), ncomp_fc, 0);
			computeReferenceSolution_fc(state_ref_fc_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray(), quokka::direction{idim});
			bool has_valid_data = true;
			for (int icomp = 0; icomp < ncomp_fc; ++icomp) {
				if (!std::isfinite(state_ref_fc_level0.norm0(icomp))) {
					has_valid_data = false;
					break;
				}
			}

			if (!has_valid_data) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					amrex::Print() << "\nWARNING: Face-centered reference solution was not properly set.\n";
					amrex::Print() << "Please implement computeReferenceSolution_fc() for your problem.\n\n";
				}
				continue; // Return empty vector
			}
			amrex::MultiFab::Copy(state_new_fc_shrunk, state_new_fc_[0][idim], 0, 0, ncomp_fc, 0);
			amrex::MultiFab residual_fc(state_ref_fc_level0.boxArray(), state_ref_fc_level0.DistributionMap(), ncomp_fc, 0);
			amrex::MultiFab::Copy(residual_fc, state_ref_fc_level0, 0, 0, ncomp_fc, 0);
			amrex::MultiFab::Saxpy(residual_fc, -1., state_new_fc_shrunk, 0, 0, ncomp_fc, 0);

			const auto n_cells_fc = static_cast<amrex::Real>(residual_fc.boxArray().numPts());

			for (int icomp_fc = 0; icomp_fc < ncomp_fc; ++icomp_fc) {
				const amrex::Real abs_err = residual_fc.norm1(icomp_fc) / n_cells_fc;
				const amrex::Real ref_norm = state_ref_fc_level0.norm1(icomp_fc) / n_cells_fc;

				amrex::Real rel_err = NAN;
				if (ref_norm > 0.0) {
					rel_err = abs_err / ref_norm;
				}

				comp_errors.emplace_back(componentNames_fc_[idim][icomp_fc], abs_err, rel_err);
			}
		}
	}
	if (this->suppress_output == 0) {
		amrex::Print() << "\nComponent Errors:\n";
		amrex::Print() << std::string(70, '=') << "\n";
		amrex::Print() << std::setw(25) << std::left << "Component" << std::setw(20) << std::right << "Absolute Error" << std::setw(20) << std::right
			       << "Relative Error" << "\n";
		amrex::Print() << std::string(70, '-') << "\n";
	}
	for (const auto &[name, abs_err, rel_err] : comp_errors) {
		if (this->suppress_output == 0) {
			amrex::Print() << std::setw(25) << std::left << name << std::setw(20) << std::right << std::scientific << std::setprecision(4)
				       << abs_err;
			if (std::isnan(rel_err)) {
				amrex::Print() << std::setw(20) << std::right << "N/A" << "\n";
			} else {
				amrex::Print() << std::setw(20) << std::right << std::scientific << std::setprecision(4) << rel_err << "\n";
			}
		}
	}
	if (this->suppress_output == 0) {
		amrex::Print() << std::string(70, '-') << "\n";
	}
	return comp_errors;
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computeErrorNorm(bool use_rel_err) -> amrex::Real
{
	// use_rel_err: if true, compute relative error norm; else compute absolute error norm
	// default: true

	auto comp_errors = computeComponentErrors();
	if (comp_errors.empty()) {
		if (this->suppress_output == 0) {
			amrex::Print() << "\nNo valid reference solution found. Cannot compute error norm.\n\n";
		}
		return NAN;
	}

	// Get number of cells to convert back from normalized errors to L1 norms
	const auto n_cells = static_cast<amrex::Real>(state_new_cc_[0].boxArray().numPts());
	amrex::Real sum_sq_err = 0.0;
	amrex::Real sum_sq_ref = 0.0;

	for (const auto &[name, abs_err, rel_err] : comp_errors) {
		// Convert normalized errors back to L1 norms
		amrex::Real L1_err = abs_err * n_cells;
		sum_sq_err += L1_err * L1_err;

		// Reconstruct reference L1 norm
		if (!std::isnan(rel_err) && rel_err != 0.0) {
			amrex::Real L1_ref = (abs_err / rel_err) * n_cells;
			sum_sq_ref += L1_ref * L1_ref;
		}
		// If rel_err is NaN or zero, reference was zero, contributes 0 to sum_sq_ref
	}

	const amrex::Real err_norm = std::sqrt(sum_sq_err);
	const amrex::Real sol_norm = std::sqrt(sum_sq_ref);

	amrex::Real error_norm = 0.0;
	if (use_rel_err && sol_norm > 0.0) {
		error_norm = err_norm / sol_norm;
		if (this->suppress_output == 0) {
			amrex::Print() << std::string(70, '=') << "\n";
			amrex::Print() << "\nRelative RMS L1 error norm = " << error_norm << "\n\n";
		}
	} else {
		error_norm = err_norm;
		if (this->suppress_output == 0) {
			amrex::Print() << std::string(70, '=') << "\n";
			if (sol_norm == 0.0) {
				amrex::Print() << "\nReference norm is zero; reporting absolute L1 error norm = " << error_norm << "\n\n";
			} else {
				amrex::Print() << "\nAbsolute L1 error norm = " << error_norm << "\n\n";
			}
		}
	}

	return error_norm;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of total energy
	amrex::Real const Egas0 = initSumCons[RadSystem<problem_t>::gasEnergy_index];
	amrex::Real const Egas = state_new_cc_[0].sum(RadSystem<problem_t>::gasEnergy_index) * vol;

	amrex::Real Etot0 = NAN;
	amrex::Real Etot = NAN;
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		amrex::Real Erad0 = 0.;
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			Erad0 += initSumCons[RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g];
		}
		Etot0 = Egas0 + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad0;
		amrex::Real Erad = 0.;
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			Erad += state_new_cc_[0].sum(RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
		}
		Etot = Egas + (RadSystem<problem_t>::c_light_ / RadSystem<problem_t>::c_hat_) * Erad;
	} else {
		Etot0 = Egas0;
		Etot = Egas;
	}

	amrex::Real const abs_err = (Etot - Etot0);
	amrex::Real rel_err = NAN;
	if (Etot0 != 0) {
		rel_err = abs_err / Etot0;
	}

	if (this->suppress_output == 0) {
		amrex::Print() << "\nInitial gas+radiation energy = " << Etot0 << '\n';
		amrex::Print() << "Final gas+radiation energy = " << Etot << '\n';
		amrex::Print() << "\tabsolute conservation error = " << abs_err << '\n';
		amrex::Print() << "\trelative conservation error = " << rel_err << '\n';
		amrex::Print() << '\n';
	}

	amrex::Print() << '\n';

	// compute average number of radiation subcycles per timestep
	if (cellUpdates_ > 0) {
		if (this->suppress_output == 0) {
			double const avg_rad_subcycles = static_cast<double>(radiationCellUpdates_) / static_cast<double>(cellUpdates_);
			amrex::Print() << "avg. num. of radiation subcycles = " << avg_rad_subcycles << '\n';
			amrex::Print() << '\n';
		}
	} else {
		amrex::Print() << "No cell updates performed!\n";
		amrex::Print() << '\n';
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	const BL_PROFILE("QuokkaSimulation::advanceSingleTimestepAtLevel()");

	// get flux and EMF registers
	amrex::FluxRegister *fr_as_crse = nullptr;
	amrex::FluxRegister *fr_as_fine = nullptr;
	amrex::EdgeFluxRegister *emf_as_crse = nullptr;
	amrex::EdgeFluxRegister *emf_as_fine = nullptr;
	if (do_reflux != 0) {
		if (lev < finestLevel()) {
			fr_as_crse = flux_reg_[lev + 1].get();
			if (fr_as_crse != nullptr) {
				fr_as_crse->setVal(0.0);
			}
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				emf_as_crse = emf_reg_[lev + 1].get();
				if (emf_as_crse != nullptr) {
					emf_as_crse->reset();
				}
			}
		}
		if (lev > 0) {
			fr_as_fine = flux_reg_[lev].get();
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				emf_as_fine = emf_reg_[lev].get();
			}
		}
	}

	// since we are starting a new timestep, need to swap old and new state vectors
	std::swap(state_old_cc_[lev], state_new_cc_[lev]);
	if (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		std::swap(state_old_fc_[lev], state_new_fc_[lev]);
	}

	// check hydro states before update (this can be caused by the flux register!)
	CHECK_HYDRO_STATES(state_old_cc_[lev], state_old_fc_[lev]);

	// advance hydro
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
		advanceHydroAtLevelWithRetries(lev, time, dt_lev, fr_as_crse, fr_as_fine, emf_as_crse, emf_as_fine);
	} else {
		// copy hydro vars from state_old_cc_ to state_new_cc_
		// (otherwise radiation update will be wrong!)
		amrex::MultiFab::Copy(state_new_cc_[lev], state_old_cc_[lev], 0, 0, nvars_, 0);
	}

	// check hydro states after hydro update
	CHECK_HYDRO_STATES(state_new_cc_[lev], state_new_fc_[lev]);

	// subcycle radiation
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		subcycleRadiationAtLevel(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	}

	// check hydro states after radiation update
	CHECK_HYDRO_STATES(state_new_cc_[lev], state_new_fc_[lev]);

	// compute any operator-split terms here (user-defined)
	computeAfterLevelAdvance(lev, time, dt_lev, ncycle);

	// check hydro states after user work
	CHECK_HYDRO_STATES(state_new_cc_[lev], state_new_fc_[lev]);

	// check state validity
	AMREX_ASSERT(!state_new_cc_[lev].contains_nan(0, state_new_cc_[lev].nComp()));
}

template <typename problem_t> void QuokkaSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs_mf, const int lev)
{
	// add hydro density to Poisson rhs
	auto const &state = state_new_cc_[lev].const_arrays();
	auto rhs = rhs_mf.arrays();
	const Real G = Gconst_;

	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// *add* density to rhs_mf
		// (N.B. particles **will not work** if you overwrite the density here!)
		rhs[bx](i, j, k) += 4.0 * M_PI * G * state[bx](i, j, k, HydroSystem<problem_t>::density_index);
	});
	amrex::Gpu::streamSynchronizeAll();
}

template <typename problem_t> void QuokkaSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi_mf, const int lev, const amrex::Real dt)
{
#if (AMREX_SPACEDIM == 3)
	// apply Poisson gravity operator on level 'lev'
	auto const &dx = geom[lev].CellSizeArray();
	auto const &phi = phi_mf.const_arrays();
	auto state = state_new_cc_[lev].arrays();

	amrex::ParallelFor(phi_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// add operator-split gravitational acceleration
		const amrex::Real rho = state[bx](i, j, k, HydroSystem<problem_t>::density_index);
		amrex::Real px = state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		amrex::Real py = state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		amrex::Real pz = state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const amrex::Real KE_old = 0.5 * (px * px + py * py + pz * pz) / rho;

		// g = -grad \phi
		amrex::Real gx = -0.5 * (phi[bx](i + 1, j, k) - phi[bx](i - 1, j, k)) / dx[0];
		amrex::Real gy = -0.5 * (phi[bx](i, j + 1, k) - phi[bx](i, j - 1, k)) / dx[1];
		amrex::Real gz = -0.5 * (phi[bx](i, j, k + 1) - phi[bx](i, j, k - 1)) / dx[2];

		px += dt * rho * gx;
		py += dt * rho * gy;
		pz += dt * rho * gz;
		const amrex::Real KE_new = 0.5 * (px * px + py * py + pz * pz) / rho;
		const amrex::Real dKE = KE_new - KE_old;

		state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px;
		state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py;
		state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz;
		state[bx](i, j, k, HydroSystem<problem_t>::energy_index) += dKE;
	});
#else
	amrex::ignore_unused(phi_mf, lev, dt);
#endif // (AMREX_SPACEDIM == 3)
}

template <typename problem_t> void QuokkaSimulation<problem_t>::projectFaceCenteredMagneticField()
{
#if AMREX_SPACEDIM == 3
	static_assert(Physics_Traits<problem_t>::is_mhd_enabled, "Magnetic field initialisation requires MHD to be enabled.");

	if (!projectInitialBField_) {
		return;
	}

	auto const has_ext_dir_hydro_bc = [&]() {
		constexpr int hydro_first = Physics_Indices<problem_t>::hydroFirstIndex;
		constexpr int num_hydro_vars = Physics_NumVars::numHydroVars;
		for (int offset = 0; offset < num_hydro_vars; ++offset) {
			int const component_index = hydro_first + offset;
			if (component_index >= static_cast<int>(BCs_cc_.size())) {
				continue;
			}
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				int const lo_bc = BCs_cc_[component_index].lo(dir);
				int const hi_bc = BCs_cc_[component_index].hi(dir);
				if (lo_bc == amrex::BCType::ext_dir || lo_bc == amrex::BCType::ext_dir_cc || hi_bc == amrex::BCType::ext_dir ||
				    hi_bc == amrex::BCType::ext_dir_cc) {
					return true;
				}
			}
		}
		return false;
	};

	if (has_ext_dir_hydro_bc()) {
		amrex::Print() << "Skipping initial magnetic field projection because ext_dir hydrodynamic boundary conditions are not supported.\n";
		return;
	}

	int const finest = this->finest_level;
	if (finest < 0) {
		return;
	}
	amrex::Vector<amrex::Geometry> geom_levels(finest + 1);
	amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>> projector_umac(finest + 1);
	amrex::Vector<amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM>> projector_beta(finest + 1);
	amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>> beta_storage(finest + 1);
	amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>> bfield_storage(finest + 1);

	auto const fill_face_boundaries = [&](int lev) {
		auto const time = (lev < static_cast<int>(tNew_.size())) ? tNew_[lev] : static_cast<amrex::Real>(0.0);
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			fillBoundaryConditions(state_new_fc_[lev][dir], state_new_fc_[lev][dir], lev, time, quokka::centering::fc,
					       static_cast<quokka::direction>(dir), AMRSimulation<problem_t>::InterpHookNone,
					       AMRSimulation<problem_t>::InterpHookNone, FillPatchType::fillpatch_function);
		}
	};

	auto const fill_all_boundaries = [&]() {
		for (int lev = 0; lev <= finest; ++lev) {
			fill_face_boundaries(lev);
		}
	};

	for (int lev = 0; lev <= finest; ++lev) {
		AMREX_ASSERT(lev < static_cast<int>(state_new_fc_.size()));
		geom_levels[lev] = this->Geom(lev);
		auto const &ba = boxArray(lev);
		auto const &dm = DistributionMap(lev);

		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			amrex::IntVect faceType = amrex::IntVect::TheDimensionVector(dir);
			beta_storage[lev][dir] = std::make_unique<amrex::MultiFab>(amrex::convert(ba, faceType), dm, 1, 0);
			beta_storage[lev][dir]->setVal(1.0);
			bfield_storage[lev][dir] =
			    std::make_unique<amrex::MultiFab>(state_new_fc_[lev][dir], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1);
			projector_umac[lev][dir] = bfield_storage[lev][dir].get();
			projector_beta[lev][dir] = beta_storage[lev][dir].get();
		}
	}

	fill_all_boundaries();

	constexpr amrex::Real tolerance_ratio = 2.0e-14;
	constexpr amrex::Real small_b = 1.0e-30;

	amrex::Real min_dx_global = std::numeric_limits<amrex::Real>::max();
	amrex::Real max_bmag_global = small_b;
	amrex::Real initial_div_norm = 0.0;

	for (int lev = 0; lev <= finest; ++lev) {
		amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> face_ptrs;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			face_ptrs[dir] = bfield_storage[lev][dir].get();
		}

		amrex::MultiFab div_initial(boxArray(lev), DistributionMap(lev), 1, 0);
		amrex::computeDivergence(div_initial, face_ptrs, geom_levels[lev]);
		initial_div_norm = std::max(initial_div_norm, div_initial.norm0(0, 0, false));

		amrex::MultiFab b_cc(boxArray(lev), DistributionMap(lev), AMREX_SPACEDIM, 0);
		amrex::average_face_to_cellcenter(b_cc, 0, face_ptrs);

		amrex::MultiFab b_mag(boxArray(lev), DistributionMap(lev), 1, 0);
		for (amrex::MFIter mfi(b_mag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
			auto const &bcc_arr = b_cc.const_array(mfi);
			auto bmag_arr = b_mag.array(mfi);
			amrex::Box const &box = mfi.tilebox();
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const bx_loc = bcc_arr(i, j, k, 0);
				amrex::Real const by_loc = bcc_arr(i, j, k, 1);
				amrex::Real const bz_loc = bcc_arr(i, j, k, 2);
				bmag_arr(i, j, k, 0) = std::sqrt(bx_loc * bx_loc + by_loc * by_loc + bz_loc * bz_loc);
			});
		}
		max_bmag_global = std::max(max_bmag_global, b_mag.norm0(0, 0, false));

		auto const &dx = geom_levels[lev].CellSizeArray();
		amrex::Real const dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
		min_dx_global = std::min(min_dx_global, dx_min);
	}

	if (min_dx_global <= 0.0) {
		min_dx_global = std::numeric_limits<amrex::Real>::min();
	}
	amrex::Real const abs_tol = tolerance_ratio * std::max(max_bmag_global, small_b) / min_dx_global;
	amrex::Real rel_tol = 0.0;
	if (initial_div_norm > 0.0) {
		rel_tol = abs_tol / initial_div_norm;
		rel_tol = std::min(rel_tol, static_cast<amrex::Real>(1.0));
	}

	auto const compute_dimensionless_divergence = [&]() {
		amrex::Real max_norm = 0.0;
		for (int lev = 0; lev <= finest; ++lev) {
			amrex::MultiFab divB(boxArray(lev), DistributionMap(lev), 1, 0);
			amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> bfield_ptrs;
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				bfield_ptrs[dir] = bfield_storage[lev][dir].get();
			}
			amrex::computeDivergence(divB, bfield_ptrs, geom_levels[lev]);

			auto const &dx = geom_levels[lev].CellSizeArray();
			amrex::Real const dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
			amrex::Real const scale_b = std::max(max_bmag_global, small_b);

			amrex::MultiFab ratio(divB.boxArray(), divB.DistributionMap(), 1, 0);
			for (amrex::MFIter mfi(divB, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
				auto const &div_arr = divB.const_array(mfi);
				auto ratio_arr = ratio.array(mfi);
				amrex::Box const &box = mfi.tilebox();
				amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					amrex::Real const div_loc = div_arr(i, j, k, 0);
					ratio_arr(i, j, k, 0) = dx_min * std::abs(div_loc) / scale_b;
				});
			}

			max_norm = std::max(max_norm, ratio.norm0(0, 0, false));
		}
		return max_norm;
	};

	amrex::LPInfo info;
	info.setAgglomeration(false);
	info.setConsolidation(false);

	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_lo;
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_hi;
	constexpr int nvarPerDim_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
	AMREX_ALWAYS_ASSERT(nvarPerDim_fc > 0);

	auto const to_linop_bc = [](int bc_value) -> amrex::LinOpBCType {
		if (bc_value == amrex::BCType::reflect_even || bc_value == amrex::BCType::foextrap) {
			return amrex::LinOpBCType::Neumann; // Neumann zero-valued BC
		}
		if (bc_value == amrex::BCType::int_dir) {
			return amrex::LinOpBCType::Periodic; // periodic BC
		}
		return amrex::LinOpBCType::Dirichlet; // zero-valued or fallback BC (includes reflect_odd)
	};

	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		int const component_index = dir * nvarPerDim_fc + MHDSystem<problem_t>::bfield_index;
		bc_lo[dir] = to_linop_bc(BCs_fc_[component_index].lo(dir));
		bc_hi[dir] = to_linop_bc(BCs_fc_[component_index].hi(dir));
	}

	Hydro::MacProjector macproj(projector_umac, amrex::MLMG::Location::FaceCenter, projector_beta, amrex::MLMG::Location::FaceCenter,
				    amrex::MLMG::Location::CellCenter, geom_levels, info);
	macproj.setDomainBC(bc_lo, bc_hi);
	for (int lev = 0; lev <= finest; ++lev) {
		macproj.setLevelBC(lev, nullptr);
	}
	macproj.setVerbose(Verbose() ? 1 : 0);
	macproj.getMLMG().setBottomVerbose(0);
	amrex::Print() << "\nProjecting initial magnetic field...\n";
	macproj.project(rel_tol, abs_tol);
	fill_all_boundaries();

	amrex::Real max_divB_norm = compute_dimensionless_divergence();
	amrex::Real const divB_tolerance = tolerance_ratio;
	amrex::Print() << "projectFaceCenteredMagneticField: max(dx * |div B| / |B|) = " << max_divB_norm << ", tolerance = " << divB_tolerance << '\n';
	if (max_divB_norm > divB_tolerance) {
		int forced_iters = 1;
		int attempt = 0;
		constexpr int max_attempts = 100;
		while (max_divB_norm > divB_tolerance) {
			if (attempt >= max_attempts) {
				amrex::Print() << "projectFaceCenteredMagneticField: L_inf(||div B||) = " << max_divB_norm << ", tolerance = " << divB_tolerance
					       << '\n';
				amrex::Abort("Magnetic field MAC projection failed to satisfy divergence tolerance.");
			}
			++attempt;
			macproj.getMLMG().setFixedIter(forced_iters);
			macproj.getMLMG().setMaxIter(forced_iters);
			macproj.project(0.0, 0.0);
			fill_all_boundaries();
			max_divB_norm = compute_dimensionless_divergence();
			amrex::Print() << "projectFaceCenteredMagneticField retry " << attempt << " (" << forced_iters
				       << " iter(s)): L_inf(||div B||) = " << max_divB_norm << ", tolerance = " << divB_tolerance << '\n';
			if (max_divB_norm > divB_tolerance) {
				forced_iters = std::min(forced_iters * 2, 64);
			}
		}
		macproj.getMLMG().setFixedIter(0);
		macproj.getMLMG().setMaxIter(200);
	}
#endif
}

template <typename problem_t> void QuokkaSimulation<problem_t>::updateInitialMagneticEnergyFromFaceField()
{
#if AMREX_SPACEDIM == 3
	if constexpr (!Physics_Traits<problem_t>::is_mhd_enabled) {
		return;
	}

	if (!updateInitialMagneticEnergy_) {
		return;
	}

	int const finest = this->finest_level;
	if (finest < 0) {
		return;
	}

	for (int lev = 0; lev <= finest; ++lev) {
		std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_storage;
		amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> face_ptrs;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			face_storage[dir] =
			    std::make_unique<amrex::MultiFab>(state_new_fc_[lev][dir], amrex::make_alias, Physics_Indices<problem_t>::mhdFirstIndex, 1);
			face_ptrs[dir] = face_storage[dir].get();
		}

		amrex::MultiFab b_cc(boxArray(lev), DistributionMap(lev), AMREX_SPACEDIM, 0);
		amrex::average_face_to_cellcenter(b_cc, 0, face_ptrs);

		auto const bcc_arrays = b_cc.const_arrays();
		auto cons_arrays = state_new_cc_[lev].arrays();
		amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho = cons_arrays[bx](i, j, k, HydroSystem<problem_t>::density_index);
			amrex::Real const px = cons_arrays[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			amrex::Real const py = cons_arrays[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			amrex::Real const pz = cons_arrays[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			amrex::Real kinetic = 0.0;
			if (rho > 0.0) {
				amrex::Real const inv_rho = 1.0 / rho;
				kinetic = 0.5 * inv_rho * (px * px + py * py + pz * pz);
			}

			amrex::Real const eint = cons_arrays[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index);
			amrex::Real const bx_loc = bcc_arrays[bx](i, j, k, 0);
			amrex::Real const by_loc = bcc_arrays[bx](i, j, k, 1);
			amrex::Real const bz_loc = bcc_arrays[bx](i, j, k, 2);
			amrex::Real const magnetic_energy = 0.5 * (bx_loc * bx_loc + by_loc * by_loc + bz_loc * bz_loc);

			cons_arrays[bx](i, j, k, HydroSystem<problem_t>::energy_index) = eint + kinetic + magnetic_energy;
		});

		amrex::Gpu::streamSynchronizeAll();

		if (lev < static_cast<int>(state_old_cc_.size())) {
			int const dest_comp = HydroSystem<problem_t>::energy_index;
			state_old_cc_[lev].ParallelCopy(state_new_cc_[lev], dest_comp, dest_comp, 1, state_old_cc_[lev].nGrow(), state_new_cc_[lev].nGrow());
		}
	}
#else
	amrex::ignore_unused(updateInitialMagneticEnergy_);
#endif
}

template <typename problem_t> void QuokkaSimulation<problem_t>::postInitialization()
{
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		projectFaceCenteredMagneticField();
		updateInitialMagneticEnergyFromFaceField();

		int const finest_level = finestLevel();
		for (int lev = 0; lev <= finest_level; ++lev) {
			if (lev >= static_cast<int>(state_old_fc_.size())) {
				continue;
			}
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				state_old_fc_[lev][dir].ParallelCopy(state_new_fc_[lev][dir], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc,
								     state_old_fc_[lev][dir].nGrow(), state_new_fc_[lev][dir].nGrow());
			}
		}
	}
}

// fix-up any unphysical states created by AMR operations
// (e.g., caused by the flux register or from interpolation)
template <typename problem_t> void QuokkaSimulation<problem_t>::FixupState(int lev)
{
	const BL_PROFILE("QuokkaSimulation::FixupState()");

	// fix hydro state
	HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, state_new_cc_[lev]);

	// sync internal energy and total energy
	HydroSystem<problem_t>::SyncDualEnergy(state_new_cc_[lev], state_new_fc_[lev]);
}

// Compute a new multifab 'mf' by copying in state from valid region and filling
// ghost cells
// NOTE: This has to be implemented here because PreInterpState and PostInterpState
// are implemented in this class (and must be *static* functions).
template <typename problem_t>
void QuokkaSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
					    FillPatchType fptype)
{
	const BL_PROFILE("AMRSimulation::FillPatch()");

	amrex::Vector<amrex::MultiFab *> cmf;
	amrex::Vector<amrex::MultiFab *> fmf;
	amrex::Vector<amrex::Real> ctime;
	amrex::Vector<amrex::Real> ftime;

	if (lev == 0) {
		// in this case, should return either state_new_[lev] or state_old_[lev]
		GetData(lev, time, fmf, ftime, cen, dir);
	} else {
		// in this case, should return either state_new_[lev] or state_old_[lev]
		// returns old state, new state, or both depending on 'time'
		GetData(lev, time, fmf, ftime, cen, dir);
		GetData(lev - 1, time, cmf, ctime, cen, dir);
	}

	if (cen == quokka::centering::cc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_cc_, cen, dir, fptype, PreInterpState, PostInterpState);
	} else if (cen == quokka::centering::fc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_fc_, cen, dir, fptype, PreInterpState, PostInterpState);
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::PreInterpState(amrex::MultiFab &mf, int /*scomp*/, int /*ncomp*/)
{
	const BL_PROFILE("QuokkaSimulation::PreInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto Etot = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto kinetic_energy = (px * px + py * py + pz * pz) / (2.0 * rho);

		// replace hydro total energy with specific internal energy (SIE)
		const auto e = (Etot - kinetic_energy) / rho;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = e;
	});
}

template <typename problem_t> void QuokkaSimulation<problem_t>::PostInterpState(amrex::MultiFab &mf, int /*scomp*/, int /*ncomp*/)
{
	const BL_PROFILE("QuokkaSimulation::PostInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto e = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto Eint = rho * e;
		const auto kinetic_energy = (px * px + py * py + pz * pz) / (2.0 * rho);

		// recompute hydro total energy from Eint + KE
		const auto Etot = Eint + kinetic_energy;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Etot;
	});
}

template <typename problem_t>
template <typename F>
auto QuokkaSimulation<problem_t>::computeAxisAlignedProfile(const int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>
{
	// compute a 1D profile of user_f(i, j, k, state) along the given axis.
	const BL_PROFILE("QuokkaSimulation::computeAxisAlignedProfile()");

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
	}

	// evaluate user_f on all levels
	for (int lev = 0; lev <= finest_level; ++lev) {
		for (amrex::MFIter iter(q[lev]); iter.isValid(); ++iter) {
			auto const &box = iter.validbox();
			auto const &state = state_new_cc_[lev].const_array(iter);
			auto const &result = q[lev].array(iter);
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) { result(i, j, k) = user_f(i, j, k, state); });
		}
	}

	// average down
	for (int crse_lev = finest_level - 1; crse_lev >= 0; --crse_lev) {
		amrex::average_down(q[crse_lev + 1], q[crse_lev], geom[crse_lev + 1], geom[crse_lev], 0, q[crse_lev].nComp(), refRatio(crse_lev));
	}

	// compute 1D profile from level 0 multifab
	amrex::Box domain = geom[0].Domain();
	auto profile = amrex::sumToLine(q[0], 0, q[0].nComp(), domain, axis);

	// normalize profile
	amrex::Long const numCells = domain.numPts() / domain.length(axis);
	for (double &bin : profile) {
		bin /= static_cast<amrex::Real>(numCells);
	}

	return profile;
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::FluxRegister *fr_as_crse,
								 amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse,
								 amrex::EdgeFluxRegister *emf_as_fine)
{
	const BL_PROFILE_REGION("HydroSolver");
	const int max_retries = 6;

	if constexpr (!Physics_Traits<problem_t>::is_mhd_enabled) {
		amrex::ignore_unused(emf_as_crse, emf_as_fine);
	}

	amrex::MultiFab accepted_state_cc(grids[lev], dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
	amrex::Copy(accepted_state_cc, state_old_cc_[lev], 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> accepted_state_fc;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_fc = amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
			accepted_state_fc[idim].define(ba_fc, dmap[lev], Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
			amrex::Copy(accepted_state_fc[idim], state_old_fc_[lev][idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
		}
	}

	auto restoreHydroState = [&]() {
		amrex::Copy(state_new_cc_[lev], accepted_state_cc, 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::Copy(state_new_fc_[lev][idim], accepted_state_fc[idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
			}
		}
	};

	auto updateAcceptedHydroState = [&]() {
		amrex::Copy(accepted_state_cc, state_new_cc_[lev], 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::Copy(accepted_state_fc[idim], state_new_fc_[lev][idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
			}
		}
	};

	const int max_total_substeps = static_cast<int>(1U << static_cast<unsigned>(max_retries));
	int completed_units = 0;
	int cur_retry_level = 0;

	while (completed_units < max_total_substeps && cur_retry_level <= max_retries) {
		const int total_substeps = static_cast<int>(1U << static_cast<unsigned>(cur_retry_level));
		const int units_per_substep = max_total_substeps / total_substeps;
		const int start_substep = completed_units / units_per_substep;
		const amrex::Real dt_substep = dt_lev / static_cast<amrex::Real>(total_substeps);
		const amrex::Real dt_attempt_remaining = dt_substep * static_cast<amrex::Real>(total_substeps - start_substep);

		if (cur_retry_level > 0 && Verbose()) {
			amrex::Print() << "\t>> Re-trying hydro advance at level " << lev << " with reduced timestep (remaining dt = " << dt_attempt_remaining
				       << ", total substeps = " << total_substeps << ", completed substeps = " << start_substep << ", dt_new = " << dt_substep
				       << ")\n";
		}

		amrex::MultiFab state_old_cc_tmp(grids[lev], dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
		amrex::Copy(state_old_cc_tmp, accepted_state_cc, 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);

		std::array<amrex::MultiFab, AMREX_SPACEDIM> state_old_fc_tmp;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				auto ba_fc = amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
				state_old_fc_tmp[idim].define(ba_fc, dmap[lev], Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
				amrex::Copy(state_old_fc_tmp[idim], accepted_state_fc[idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
			}
		}

		bool attempt_failed = false;

		for (int substep_index = start_substep; substep_index < total_substeps; ++substep_index) {
			if (substep_index > start_substep) {
				amrex::Copy(state_old_cc_tmp, state_new_cc_[lev], 0, 0, nvars_, nghost_cc_);
				if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
					for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
						amrex::Copy(state_old_fc_tmp[idim], state_new_fc_[lev][idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc,
							    nghost_fc_);
					}
				}
			}

			const amrex::Real current_substep_time = time + static_cast<amrex::Real>(substep_index) * dt_substep;
			const bool substep_success = advanceHydroAtLevel(state_old_cc_tmp, state_old_fc_tmp, fr_as_crse, fr_as_fine, emf_as_crse, emf_as_fine,
									 lev, current_substep_time, dt_substep);
			if (!substep_success) {
				attempt_failed = true;
				break;
			}

			completed_units += units_per_substep;
			completed_units = std::min(completed_units, max_total_substeps);

			updateAcceptedHydroState();
		}

		if (attempt_failed) {
			restoreHydroState();
			++cur_retry_level;
			continue;
		}

		break;
	}

	if (completed_units < max_total_substeps) {
		// crash, we have exceeded max_retries
		amrex::Print() << "\nQUOKKA FATAL ERROR\n"
			       << "Hydro update exceeded max_retries on level " << lev << ". Cannot continue, crashing...\n"
			       << std::endl; // NOLINT(performance-avoid-endl)

		// write plotfile or Ascent Blueprint file
		amrex::ParallelDescriptor::Barrier();
#ifdef AMREX_USE_ASCENT
		conduit::Node mesh;
		amrex::SingleLevelToBlueprint(state_new_cc_[lev], componentNames_cc_, geom[lev], time, istep[lev] + 1, mesh);
		conduit::Node bpMeshHost;
		bpMeshHost.set(mesh); // copy to host mem (needed for Blueprint HDF5 output)
		amrex::WriteBlueprintFiles(bpMeshHost, "debug_hydro_state_fatal", istep[lev] + 1, "hdf5");
#else
		WriteSingleLevelPlotfileSimplified("debug_hydro_state_fatal", state_new_cc_[lev], componentNames_cc_, lev, 1);
#endif
		amrex::ParallelDescriptor::Barrier();

		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::ParallelDescriptor::Abort();
		}
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::isCflViolated(int lev, amrex::Real /*time*/, amrex::Real dt_actual) -> bool
{
	// check whether dt_actual would violate CFL condition using the post-update hydro state

	// compute max signal speed
	amrex::Real max_signal = HydroSystem<problem_t>::maxSignalSpeedLocal(state_new_cc_[lev], state_new_fc_[lev]);
	amrex::ParallelDescriptor::ReduceRealMax(max_signal);

	// compute dt_cfl
	auto dx = geom[lev].CellSizeArray();
	const amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	const amrex::Real dt_cfl = cflNumber_ * (dx_min / max_signal);

	// check whether dt_actual > dt_cfl (CFL violation)
	const amrex::Real max_factor = 1.1;
	const bool cflViolation = dt_actual > (max_factor * dt_cfl);
	if (cflViolation && Verbose()) {
		amrex::Print() << "\t>> CFL violation detected on level " << lev << " with dt_lev = " << dt_actual << " and dt_cfl = " << dt_cfl << "\n"
			       << "\t   max_signal = " << max_signal << "\n";
	}
	return cflViolation;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::printCoordinates(int lev, const amrex::IntVect &cell_idx)
{

	amrex::Real x_coord = geom[lev].ProbLo(0) + (cell_idx[0] + 0.5) * geom[lev].CellSize(0);
	amrex::Real y_coord = NAN;
	amrex::Real z_coord = NAN;

	if (AMREX_SPACEDIM > 1) {
		y_coord = geom[lev].ProbLo(1) + (cell_idx[1] + 0.5) * geom[lev].CellSize(1);

		if (AMREX_SPACEDIM > 2) {
			z_coord = geom[lev].ProbLo(2) + (cell_idx[2] + 0.5) * geom[lev].CellSize(2);
		}
	}
	amrex::Print() << "Coordinates: (" << x_coord << ", " << y_coord << ", " << z_coord << "): ";
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_old_fc_tmp,
						      amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse,
						      amrex::EdgeFluxRegister *emf_as_fine, int lev, amrex::Real time, amrex::Real dt_lev) -> bool
{
	const BL_PROFILE("QuokkaSimulation::advanceHydroAtLevel()");

	if constexpr (!Physics_Traits<problem_t>::is_mhd_enabled) {
		amrex::ignore_unused(emf_as_crse, emf_as_fine);
	}

	const amrex::Real stage1Weight = (integratorOrder_ == 2) ? 0.5 : 1.0;

	auto ba_cc = grids[lev];
	auto dm = dmap[lev];
	auto dx = geom[lev].CellSizeArray();

	// do Strang split source terms (first half-step)
	auto burn_success_first = addStrangSplitSourcesWithBuiltin(state_old_cc_tmp, lev, time, 0.5 * dt_lev);

	// check if reactions failed for source terms. If it failed, return false.
	if (!burn_success_first) {
		return burn_success_first;
	}

	// create temporary multifab for intermediate state
	amrex::MultiFab state_inter_cc_(grids[lev], dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
	state_inter_cc_.setVal(0); // prevent assert in fillBoundaryConditions when radiation is enabled
	std::array<amrex::MultiFab, AMREX_SPACEDIM> state_inter_fc_;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_fc = amrex::convert(ba_cc, amrex::IntVect::TheDimensionVector(idim));
			state_inter_fc_[idim].define(ba_fc, dm, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_);
			state_inter_fc_[idim].setVal(0);
		}
	}

	// create temporary multifabs for combined RK2 flux and time-average face velocity
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux_rk2;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> avgFaceVel;
	const int nghost_vel = 2; // 2 ghost faces are needed for tracer particles

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_fc = amrex::convert(ba_cc, amrex::IntVect::TheDimensionVector(idim));
		// initialize flux MultiFab
		flux_rk2[idim] = amrex::MultiFab(ba_fc, dm, nvars_, 0);
		flux_rk2[idim].setVal(0);
		// initialize velocity MultiFab
		avgFaceVel[idim] = amrex::MultiFab(ba_fc, dm, 1, nghost_vel);
		avgFaceVel[idim].setVal(0);
	}
	std::array<amrex::MultiFab, AMREX_SPACEDIM> ec_emf_components_rk_ave;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_ec = amrex::convert(ba_cc, amrex::IntVect(AMREX_D_DECL(1, 1, 1)) - amrex::IntVect::TheDimensionVector(idim));
			ec_emf_components_rk_ave[idim].define(ba_ec, dm, 1, 0);
			ec_emf_components_rk_ave[idim].setVal(0.0);
		}
	}

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_cc_tmp, state_old_cc_tmp, lev, time, quokka::centering::cc, quokka::direction::na, PreInterpState, PostInterpState);
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fillBoundaryConditions(state_old_fc_tmp[idim], state_old_fc_tmp[idim], lev, time, quokka::centering::fc, quokka::direction{idim},
					       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone,
					       FillPatchType::fillpatch_function);
		}
	}

	// LOW LEVEL DEBUGGING: output state_old_cc_tmp (with ghost cells)
	if (lowLevelDebuggingOutput_ == 1) {
#ifdef AMREX_USE_ASCENT
		// write Blueprint HDF5 files
		conduit::Node mesh;
		amrex::SingleLevelToBlueprint(state_old_cc_tmp, componentNames_cc_, geom[lev], time, istep[lev] + 1, mesh);
		amrex::WriteBlueprintFiles(mesh, "debug_stage1_filled_state_old", istep[lev] + 1, "hdf5");
#else
		// write AMReX plotfile
		// WriteSingleLevelPlotfile(CustomPlotFileName("debug_stage1_filled_state_old", istep[lev]+1),
		//    state_old_cc_tmp, componentNames_cc_, geom[lev], time, istep[lev]+1);
#endif
	}

	// check state validity
	AMREX_ASSERT(!state_old_cc_tmp.contains_nan(0, state_old_cc_tmp.nComp()));
	AMREX_ASSERT(!state_old_cc_tmp.contains_nan()); // check ghost cells

	auto [FOfluxArrays, FOfaceVel, FOfast_mhd_wavespeeds] = computeFOHydroFluxes(state_old_cc_tmp, state_old_fc_tmp, nvars_, lev);

	std::array<amrex::MultiFab, AMREX_SPACEDIM> ec_emf_components_fo;
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_ec = amrex::convert(ba_cc, amrex::IntVect(AMREX_D_DECL(1, 1, 1)) - amrex::IntVect::TheDimensionVector(idim));
			ec_emf_components_fo[idim].define(ba_ec, dm, 1, 0);
		}
		MHDSystem<problem_t>::ComputeEMF(ec_emf_components_fo, state_old_cc_tmp, FOfaceVel, state_old_fc_tmp, FOfast_mhd_wavespeeds,
						 emfReconstructionOrder_, emfAveragingScheme_, emfComputingScheme_);
	}

	// Stage 1 of RK2-SSP
	{
		//  advance all grids on local processor (Stage 1 of integrator)
		auto const &stateOld_cc = state_old_cc_tmp;
		auto &stateNew_cc = state_inter_cc_;

		auto const &stateOld_fc = state_old_fc_tmp;
		auto &stateNew_fc = state_inter_fc_;

		auto [fluxArrays, faceVel, fast_mhd_wavespeeds] = computeHydroFluxes(stateOld_cc, stateOld_fc, nvars_, lev);

		std::array<amrex::MultiFab, AMREX_SPACEDIM> ec_emf_components_rk_stage1;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				auto ba_ec = amrex::convert(ba_cc, amrex::IntVect(AMREX_D_DECL(1, 1, 1)) - amrex::IntVect::TheDimensionVector(idim));
				ec_emf_components_rk_stage1[idim].define(ba_ec, dm, 1, 0);
			}
			MHDSystem<problem_t>::ComputeEMF(ec_emf_components_rk_stage1, stateOld_cc, faceVel, stateOld_fc, fast_mhd_wavespeeds,
							 emfReconstructionOrder_, emfAveragingScheme_, emfComputingScheme_);
		}

		amrex::MultiFab rhs(grids[lev], dmap[lev], nvars_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 1);
		redoFlag.setVal(quokka::redoFlag::none);

		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, nvars_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld_cc, stateOld_fc, dx, faceVel, redoFlag);
		HydroSystem<problem_t>::PredictStep(stateOld_cc, stateNew_cc, rhs, dt_lev, nvars_, redoFlag);

		// LOW LEVEL DEBUGGING: output rhs
		if (lowLevelDebuggingOutput_ == 1) {
			// write rhs
			std::string plotfile_name = CustomPlotFileName("debug_stage1_rhs_fluxes", istep[lev] + 1);
			WriteSingleLevelPlotfileSimplified("debug_stage1_rhs_fluxes", rhs, componentNames_cc_, lev, 1);

			// write fluxes
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string fullprefix =
				    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("Flux_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(fluxArrays[idim], fullprefix);
			}
			// write face velocities
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string fullprefix = amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_",
										       std::string("FaceVel_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(faceVel[idim], fullprefix);
			}
		}

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // ensure device-side ops are finished

		amrex::Long const ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-1] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
				const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
				// Calculate the coordinates based on the cell index and cell size
				printCoordinates(lev, cell_idx);
				amrex::print_state(stateNew_cc, cell_idx);
			}

			// synchronize redoFlag across ranks
			redoFlag.FillBoundary(geom[lev].periodicity());

			replaceFluxes(fluxArrays, FOfluxArrays, redoFlag);
			replaceFluxes(faceVel, FOfaceVel, redoFlag); // needed for dual energy
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				replaceEMFs(ec_emf_components_rk_stage1, ec_emf_components_fo, redoFlag); // replace emf components
			}

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, nvars_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld_cc, stateOld_fc, dx, faceVel, redoFlag);
			HydroSystem<problem_t>::PredictStep(stateOld_cc, stateNew_cc, rhs, dt_lev, nvars_, redoFlag);

			amrex::Gpu::streamSynchronizeAll(); // just in case
			amrex::Long const ncells_bad = static_cast<int>(redoFlag.sum(0));
			if (ncells_bad > 0) {
				// FOFC failed
				if (Verbose()) {
					const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
					// print cell state
					amrex::Print() << "[FOFC-1] Flux correction failed:\n";
					printCoordinates(lev, cell_idx);
					amrex::print_state(stateNew_cc, cell_idx);
					amrex::Print() << "[FOFC-1] failed for " << ncells_bad << " cells on level " << lev << "\n";
				}
				if (abortOnFofcFailure_ != 0) {
					return false;
				}
			}
		}

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::MultiFab::Saxpy(ec_emf_components_rk_ave[idim], stage1Weight, ec_emf_components_rk_stage1[idim], 0, 0, 1, 0);
			}
			MHDSystem<problem_t>::SolveInductionEqn(stateOld_fc, stateNew_fc, ec_emf_components_rk_stage1, dt_lev, geom[lev].CellSizeArray());
		}

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::Saxpy(flux_rk2[idim], stage1Weight, fluxArrays[idim], 0, 0, nvars_, 0);
			amrex::MultiFab::Saxpy(avgFaceVel[idim], stage1Weight, faceVel[idim], 0, 0, 1, 0);
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, stateNew_cc);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateNew_cc, stateNew_fc);
		}
	}
	amrex::Gpu::streamSynchronizeAll();

	// Stage 2 of RK2-SSP
	if (integratorOrder_ == 2) {
		//  update ghost zones [intermediate stage stored in state_inter_cc_]
		fillBoundaryConditions(state_inter_cc_, state_inter_cc_, lev, time + dt_lev, quokka::centering::cc, quokka::direction::na, PreInterpState,
				       PostInterpState);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				fillBoundaryConditions(state_inter_fc_[idim], state_inter_fc_[idim], lev, time, quokka::centering::fc, quokka::direction{idim},
						       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone,
						       FillPatchType::fillpatch_function);
			}
		}

		// check intermediate state validity
		AMREX_ASSERT(!state_inter_cc_.contains_nan(0, state_inter_cc_.nComp()));
		AMREX_ASSERT(!state_inter_cc_.contains_nan()); // check ghost zones

		auto const &stateOld_cc = state_old_cc_tmp;
		auto const &stateInter_cc = state_inter_cc_;
		auto &stateFinal_cc = state_new_cc_[lev];

		auto const &stateOld_fc = state_old_fc_tmp;
		auto const &stateInter_fc = state_inter_fc_;
		auto &stateFinal_fc = state_new_fc_[lev];

		auto [fluxArrays, faceVel, fast_mhd_wavespeeds] = computeHydroFluxes(stateInter_cc, stateInter_fc, nvars_, lev);

		std::array<amrex::MultiFab, AMREX_SPACEDIM> ec_emf_components_rk_stage2;
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				auto ba_ec = amrex::convert(ba_cc, amrex::IntVect(AMREX_D_DECL(1, 1, 1)) - amrex::IntVect::TheDimensionVector(idim));
				ec_emf_components_rk_stage2[idim].define(ba_ec, dm, 1, 0);
			}
			MHDSystem<problem_t>::ComputeEMF(ec_emf_components_rk_stage2, stateInter_cc, faceVel, stateInter_fc, fast_mhd_wavespeeds,
							 emfReconstructionOrder_, emfAveragingScheme_, emfComputingScheme_);
		}

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::Saxpy(flux_rk2[idim], 0.5, fluxArrays[idim], 0, 0, nvars_, 0);
			amrex::MultiFab::Saxpy(avgFaceVel[idim], 0.5, faceVel[idim], 0, 0, 1, 0);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				amrex::MultiFab::Saxpy(ec_emf_components_rk_ave[idim], 0.5, ec_emf_components_rk_stage2[idim], 0, 0, 1, 0);
			}
		}

		amrex::MultiFab rhs(grids[lev], dmap[lev], nvars_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 1);
		redoFlag.setVal(quokka::redoFlag::none);

		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, flux_rk2, dx, nvars_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld_cc, stateOld_fc, dx, avgFaceVel, redoFlag);
		HydroSystem<problem_t>::PredictStep(stateOld_cc, stateFinal_cc, rhs, dt_lev, nvars_, redoFlag);

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // just in case
		amrex::Long const ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-2] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
				const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
				printCoordinates(lev, cell_idx);
				amrex::print_state(stateFinal_cc, cell_idx);
			}

			// synchronize redoFlag across ranks
			redoFlag.FillBoundary(geom[lev].periodicity());

			replaceFluxes(flux_rk2, FOfluxArrays, redoFlag);
			replaceFluxes(avgFaceVel, FOfaceVel, redoFlag); // needed for dual energy

			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				replaceEMFs(ec_emf_components_rk_ave, ec_emf_components_fo, redoFlag); // replaces EMF components
			}

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, flux_rk2, dx, nvars_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld_cc, stateOld_fc, dx, avgFaceVel, redoFlag);
			HydroSystem<problem_t>::PredictStep(stateOld_cc, stateFinal_cc, rhs, dt_lev, nvars_, redoFlag);

			amrex::Gpu::streamSynchronizeAll(); // just in case
			amrex::Long const ncells_bad = redoFlag.sum(0);
			if (ncells_bad > 0) {
				// FOFC failed
				if (Verbose()) {
					const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
					// print cell state
					amrex::Print() << "[FOFC-2] Flux correction failed:\n";
					printCoordinates(lev, cell_idx);
					amrex::print_state(stateFinal_cc, cell_idx);
					amrex::Print() << "[FOFC-2] failed for " << ncells_bad << " cells on level " << lev << "\n";
				}
				if (abortOnFofcFailure_ != 0) {
					return false;
				}
			}
		}

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			MHDSystem<problem_t>::SolveInductionEqn(stateOld_fc, stateFinal_fc, ec_emf_components_rk_ave, dt_lev, geom[lev].CellSizeArray());
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, stateFinal_cc);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateFinal_cc, stateFinal_fc);
		}

	} else { // we are only doing forward Euler
		amrex::Copy(state_new_cc_[lev], state_inter_cc_, 0, 0, nvars_, 0);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::Copy(state_new_fc_[lev][idim], state_inter_fc_[idim], 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, 0);
			}
		}
	}
	amrex::Gpu::streamSynchronizeAll();

	// do Strang split source terms (second half-step)
	auto burn_success_second = addStrangSplitSourcesWithBuiltin(state_new_cc_[lev], lev, time + dt_lev, 0.5 * dt_lev);

	bool const cfl_ok = !isCflViolated(lev, time, dt_lev);
	bool const final_success = (cfl_ok && burn_success_second);

	if (do_reflux == 1 && final_success) {
		incrementFluxRegisters(fr_as_crse, fr_as_fine, flux_rk2, lev, dt_lev);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			// E = -v x B, our emf is v x B, so we need to pass -dt
			incrementEMFRegisters(emf_as_crse, emf_as_fine, ec_emf_components_rk_ave, lev, -1.0 * dt_lev);
		}
	}

	if (do_tracers != 0 && final_success) {
		TracerPC->AdvectWithUmac(avgFaceVel.data(), lev, dt_lev);
	}

	return final_success;
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,
						amrex::iMultiFab &redoFlag)
{
	const BL_PROFILE("QuokkaSimulation::replaceFluxes()");

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) { // loop over dimension
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

		amrex::IntVect ng{AMREX_D_DECL(1, 1, 1)}; // must include 1 ghost zone

		amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
			if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
				// replace fluxes with first-order ones at the faces of cell (i,j,k)
				if (flux_arrs[bx].contains(i, j, k)) {
					flux_arrs[bx](i, j, k, n) = FOflux_arrs[bx](i, j, k, n);
				}
				if (idim == 0) { // x-dir fluxes
					if (flux_arrs[bx].contains(i + 1, j, k)) {
						flux_arrs[bx](i + 1, j, k, n) = FOflux_arrs[bx](i + 1, j, k, n);
					}
				} else if (idim == 1) { // y-dir fluxes
					if (flux_arrs[bx].contains(i, j + 1, k)) {
						flux_arrs[bx](i, j + 1, k, n) = FOflux_arrs[bx](i, j + 1, k, n);
					}
				} else if (idim == 2) { // z-dir fluxes
					if (flux_arrs[bx].contains(i, j, k + 1)) {
						flux_arrs[bx](i, j, k + 1, n) = FOflux_arrs[bx](i, j, k + 1, n);
					}
				}
			}
		});
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::replaceEMFs(std::array<amrex::MultiFab, AMREX_SPACEDIM> &emf_components,
					      std::array<amrex::MultiFab, AMREX_SPACEDIM> &FO_emf_components, amrex::iMultiFab &redoFlag)
{
	const BL_PROFILE("QuokkaSimulation::replaceFluxes()");

	for (int iedge = 0; iedge < 3; ++iedge) { // loop over edges
		// ensure that flux arrays have the same number of components
		AMREX_ASSERT(emf_components[iedge].nComp() == FO_emf_components[iedge].nComp());
		int ncomp = emf_components[iedge].nComp();

		auto const &FO_emf_components_arrs = FO_emf_components[iedge].const_arrays();
		auto const &redoFlag_arrs = redoFlag.const_arrays();
		auto emf_components_arrs = emf_components[iedge].arrays();

		amrex::IntVect ng{AMREX_D_DECL(1, 1, 1)}; // must include 1 ghost zone ?

		amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
			if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
				// replace fluxes with first-order ones at the faces of cell (i,j,k)
				if (emf_components_arrs[bx].contains(i, j, k)) {
					emf_components_arrs[bx](i, j, k, n) = FO_emf_components_arrs[bx](i, j, k, n);
				}
				if (iedge == 0) { // x-edge components
					if (emf_components_arrs[bx].contains(i, j + 1, k)) {
						emf_components_arrs[bx](i, j + 1, k, n) = FO_emf_components_arrs[bx](i, j + 1, k, n);
					}
					if (emf_components_arrs[bx].contains(i, j, k + 1)) {
						emf_components_arrs[bx](i, j, k + 1, n) = FO_emf_components_arrs[bx](i, j, k + 1, n);
					}
					if (emf_components_arrs[bx].contains(i, j + 1, k + 1)) {
						emf_components_arrs[bx](i, j + 1, k + 1, n) = FO_emf_components_arrs[bx](i, j + 1, k + 1, n);
					}
				} else if (iedge == 1) { // y-edge components
					if (emf_components_arrs[bx].contains(i + 1, j, k)) {
						emf_components_arrs[bx](i + 1, j, k, n) = FO_emf_components_arrs[bx](i + 1, j, k, n);
					}
					if (emf_components_arrs[bx].contains(i, j, k + 1)) {
						emf_components_arrs[bx](i, j, k + 1, n) = FO_emf_components_arrs[bx](i, j, k + 1, n);
					}
					if (emf_components_arrs[bx].contains(i + 1, j, k + 1)) {
						emf_components_arrs[bx](i + 1, j, k + 1, n) = FO_emf_components_arrs[bx](i + 1, j, k + 1, n);
					}
				} else if (iedge == 2) { // z-edge components
					if (emf_components_arrs[bx].contains(i + 1, j, k)) {
						emf_components_arrs[bx](i + 1, j, k, n) = FO_emf_components_arrs[bx](i + 1, j, k, n);
					}
					if (emf_components_arrs[bx].contains(i, j + 1, k)) {
						emf_components_arrs[bx](i, j + 1, k, n) = FO_emf_components_arrs[bx](i, j + 1, k, n);
					}
					if (emf_components_arrs[bx].contains(i + 1, j + 1, k)) {
						emf_components_arrs[bx](i + 1, j + 1, k, n) = FO_emf_components_arrs[bx](i + 1, j + 1, k, n);
					}
				}
			}
		});
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes,
						const int srccomp, const int dstcomp)
{
	const BL_PROFILE("QuokkaSimulation::addFluxArrays()");

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto const &srcflux = srcfluxes[idim];
		auto &dstflux = dstfluxes[idim];
		amrex::Add(dstflux, srcflux, srccomp, dstcomp, srcflux.nComp(), 0);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, const int nstartNew, const int ncompNew)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	const BL_PROFILE("QuokkaSimulation::expandFluxArrays()");

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
auto QuokkaSimulation<problem_t>::computeHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc,
						     const int nvars, const int lev)
    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	const BL_PROFILE("QuokkaSimulation::computeHydroFluxes()");

	const auto ba = grids[lev];
	const auto dm = dmap[lev];

	// default is 2. we need +1 ghost to get fc-vels in the ghost-zones (for piecewise-constant reconstruction)
	// for MHD we need to accommodate the higher order reconstruction we need to do in computeEMF
	const int reconstructGhost = nghost_vel_ + 1;

	// // we need two additional ghost cells in order to compute two ghost face velocities
	const int flatteningGhost = reconstructGhost + 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	amrex::MultiFab cc_bfield_perp_comps(ba, dm, 2, nghost_cc_);
	std::array<amrex::MultiFab, 3> flatCoefs;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState_bfield;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState_bfield;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fast_mhd_wavespeeds;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		flatCoefs[idim] = amrex::MultiFab(ba, dm, 1, flatteningGhost);
	}

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
		leftState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost);
		rightState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost - 1);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructGhost - 1);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			fast_mhd_wavespeeds[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost - 1);
		}
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar_cc, consVar_fc, primVar, nghost_cc_);

	// compute flattening coefficients
	AMREX_D_TERM(HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X1>(primVar, flatCoefs[0], flatteningGhost);
		     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X2>(primVar, flatCoefs[1], flatteningGhost);
		     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X3>(primVar, flatCoefs[2], flatteningGhost);)

	// compute flux functions
	AMREX_D_TERM(
	    hydroFluxFunction<FluxDir::X1>(primVar, cc_bfield_perp_comps, leftState[0], rightState[0], leftState_bfield[0], rightState_bfield[0], flux[0],
					   facevel[0], fast_mhd_wavespeeds[0], consVar_fc, flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructGhost, nvars);
	    , hydroFluxFunction<FluxDir::X2>(primVar, cc_bfield_perp_comps, leftState[1], rightState[1], leftState_bfield[1], rightState_bfield[1], flux[1],
					     facevel[1], fast_mhd_wavespeeds[1], consVar_fc, flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructGhost, nvars);
	    ,
	    hydroFluxFunction<FluxDir::X3>(primVar, cc_bfield_perp_comps, leftState[2], rightState[2], leftState_bfield[2], rightState_bfield[2], flux[2],
					   facevel[2], fast_mhd_wavespeeds[2], consVar_fc, flatCoefs[0], flatCoefs[1], flatCoefs[2], reconstructGhost, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// write reconstructed states to disk for analysis (includes ghost zones)
	// this->writeReconstructedStatesToDisk(leftState, rightState, lev, istep[lev]);

	// write face velocities to disk for analysis
	// this->writeFaceVelocitiesToDisk(facevel, lev, istep[lev]);

	// LOW LEVEL DEBUGGING: output all of the temporary MultiFabs
	if (lowLevelDebuggingOutput_ == 1) {
		// write primitive cell-centered state
		std::string plotfile_name = CustomPlotFileName("debug_reconstruction", istep[lev] + 1);
		WriteSingleLevelPlotfileSimplified("debug_reconstruction", primVar, componentNames_cc_, lev, 1);

		// write flattening coefficients
		amrex::Vector<std::string> flatCompNames{"chi"};
		WriteSingleLevelPlotfileSimplified("debug_flattening_x", flatCoefs[0], flatCompNames, lev, 1);
		WriteSingleLevelPlotfileSimplified("debug_flattening_y", flatCoefs[1], flatCompNames, lev, 1);
		WriteSingleLevelPlotfileSimplified("debug_flattening_z", flatCoefs[2], flatCompNames, lev, 1);

		// write L interface states
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (amrex::ParallelDescriptor::IOProcessor()) {
				std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
			}
			std::string const fullprefix =
			    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("StateL_") + quokka::face_dir_str[idim]);
			amrex::VisMF::Write(leftState[idim], fullprefix);
		}
		// write R interface states
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (amrex::ParallelDescriptor::IOProcessor()) {
				std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
			}
			std::string const fullprefix =
			    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("StateR_") + quokka::face_dir_str[idim]);
			amrex::VisMF::Write(rightState[idim], fullprefix);
		}
	}

	// return flux and face-centered velocities
	return std::make_tuple(std::move(flux), std::move(facevel), std::move(fast_mhd_wavespeeds));
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_FORCE_INLINE void QuokkaSimulation<problem_t>::computeCCPerpBfieldComps(amrex::MultiFab &cc_bfield_perp_comps_mf,
									      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc) const
{
	// per-direction neighbor offsets for those two components
	std::array<int, 3> delta_x2{0, 0, 0};
	std::array<int, 3> delta_x3{0, 0, 0};

	amrex::MultiArray4<const amrex::Real> x2State_fc_bfield_in;
	amrex::MultiArray4<const amrex::Real> x3State_fc_bfield_in;
	if constexpr (DIR == FluxDir::X1) {
		x2State_fc_bfield_in = consVar_fc[1].const_arrays(); // y-faces
		x3State_fc_bfield_in = consVar_fc[2].const_arrays(); // z-faces
		delta_x2[1] = 1;
		delta_x3[2] = 1;
	} else if constexpr (DIR == FluxDir::X2) {
		x2State_fc_bfield_in = consVar_fc[2].const_arrays(); // z-faces
		x3State_fc_bfield_in = consVar_fc[0].const_arrays(); // x-faces
		delta_x2[2] = 1;
		delta_x3[0] = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		x2State_fc_bfield_in = consVar_fc[0].const_arrays(); // x-faces
		x3State_fc_bfield_in = consVar_fc[1].const_arrays(); // y-faces
		delta_x2[0] = 1;
		delta_x3[1] = 1;
	}

	auto cc_bfield_perp_comps_out = cc_bfield_perp_comps_mf.arrays();
	constexpr int b_comp = Physics_Indices<problem_t>::mhdFirstIndex;

	amrex::IntVect ng{AMREX_D_DECL(nghost_fc_, nghost_fc_, nghost_fc_)};
	amrex::ParallelFor(cc_bfield_perp_comps_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const amrex::Real bx2_m = x2State_fc_bfield_in[bx](i, j, k, b_comp);
		const amrex::Real bx2_p = x2State_fc_bfield_in[bx](i + delta_x2[0], j + delta_x2[1], k + delta_x2[2], b_comp);
		cc_bfield_perp_comps_out[bx](i, j, k, 0) = 0.5 * (bx2_m + bx2_p);

		const amrex::Real bx3_m = x3State_fc_bfield_in[bx](i, j, k, b_comp);
		const amrex::Real bx3_p = x3State_fc_bfield_in[bx](i + delta_x3[0], j + delta_x3[1], k + delta_x3[2], b_comp);
		cc_bfield_perp_comps_out[bx](i, j, k, 1) = 0.5 * (bx3_m + bx3_p);
	});
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::hydroFluxFunction(amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState,
						    amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield,
						    amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds,
						    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, amrex::MultiFab const &x1Flat,
						    amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, const int ng_reconstruct, const int nvars)
{
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		QuokkaSimulation<problem_t>::template computeCCPerpBfieldComps<DIR>(cc_bfield_perp_comps_mf, consVar_fc);
	}

	if (reconstructionOrder_ == 5) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
											   ng_reconstruct, 2);
		}
	} else if (reconstructionOrder_ == 3) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
											ng_reconstruct, 2);
		}
	} else if (reconstructionOrder_ == 2) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::minmod>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::minmod>(cc_bfield_perp_comps_mf, leftState_bfield,
													      rightState_bfield, ng_reconstruct, 2);
		}
	} else if (reconstructionOrder_ == 1) {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
											     ng_reconstruct, 2);
		}
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(primVar_mf, x1Flat, x2Flat, x3Flat, leftState, rightState, ng_reconstruct, nvars);

	// interface-centered kernel
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLD, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
											 rightState_bfield, primVar_mf, artificialViscosityK_, &x1FSpds,
											 &consVar_fc[static_cast<int>(DIR)], nghost_vel_);
	} else {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLC, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
											 rightState_bfield, primVar_mf, artificialViscosityK_, nullptr, nullptr,
											 nghost_vel_);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeFOHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc,
						       const int nvars, const int lev)
    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	const BL_PROFILE("QuokkaSimulation::computeFOHydroFluxes()");

	const auto ba = grids[lev];
	const auto dm = dmap[lev];

	// same as above: default is 2. we need +1 ghost to get fc-vels in the ghost-zones (for piecewise-constant reconstruction)
	// for MHD we need to accommodate the higher order reconstruction we need to do in computeEMF
	const int reconstructRange = nghost_vel_ + 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	amrex::MultiFab cc_bfield_perp_comps(ba, dm, 2, nghost_cc_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState_bfield;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState_bfield;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fast_mhd_wavespeeds;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		leftState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange);
		rightState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange - 1);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructRange - 1);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			fast_mhd_wavespeeds[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange - 1);
		}
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar_cc, consVar_fc, primVar, nghost_cc_);

	// compute flux functions
	AMREX_D_TERM(hydroFOFluxFunction<FluxDir::X1>(primVar, cc_bfield_perp_comps, leftState[0], rightState[0], leftState_bfield[0], rightState_bfield[0],
						      flux[0], facevel[0], fast_mhd_wavespeeds[0], consVar_fc, reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X2>(primVar, cc_bfield_perp_comps, leftState[1], rightState[1], leftState_bfield[1], rightState_bfield[1],
							flux[1], facevel[1], fast_mhd_wavespeeds[1], consVar_fc, reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X3>(primVar, cc_bfield_perp_comps, leftState[2], rightState[2], leftState_bfield[2], rightState_bfield[2],
							flux[2], facevel[2], fast_mhd_wavespeeds[2], consVar_fc, reconstructRange, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// return flux and face-centered velocities
	return std::make_tuple(std::move(flux), std::move(facevel), std::move(fast_mhd_wavespeeds));
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::hydroFOFluxFunction(amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState,
						      amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield,
						      amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &x1ConsVar_fc_mf, const int ng_reconstruct,
						      const int nvars)
{
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		QuokkaSimulation<problem_t>::template computeCCPerpBfieldComps<DIR>(cc_bfield_perp_comps_mf, x1ConsVar_fc_mf);
	}

	// donor-cell reconstruction
	HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield, ng_reconstruct,
										2);
	}

	// LLF solver
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::LLF_MHD, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
											    rightState_bfield, primVar_mf, artificialViscosityK_, &x1FSpds,
											    &x1ConsVar_fc_mf[static_cast<int>(DIR)], nghost_vel_);
	} else {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::LLF, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
											rightState_bfield, primVar_mf, artificialViscosityK_, nullptr, nullptr,
											nghost_vel_);
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew)
{
	// copy radiation state variables from stateNew_cc to stateOld_cc
	amrex::MultiFab::Copy(stateOld, stateNew, nstartHyperbolic_, nstartHyperbolic_, ncompHyperbolic_, 0);
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::FluxRegister *fr_as_crse,
							   amrex::FluxRegister *fr_as_fine)
{
	// compute radiation timestep
	int nsubSteps = 0;
	amrex::Real dt_radiation = NAN;
	auto const rad_tol = radiation_iteration_tolerance_;
	auto const rad_tol_rel = radiation_iteration_tolerance_rel_;
	auto const tempFloor = tempFloor_;

	if (Physics_Traits<problem_t>::is_hydro_enabled && !(constantDt_ > 0.)) {
		// adjust to get integer number of substeps
		nsubSteps = computeNumberOfRadiationSubsteps(lev, dt_lev_hydro);
		dt_radiation = dt_lev_hydro / static_cast<double>(nsubSteps);
	} else { // no hydro, or using constant dt (this is necessary for radiation test problems)
		dt_radiation = dt_lev_hydro;
		nsubSteps = 1;
	}

	if (Verbose() != 0) {
		amrex::Print() << "\tRadiation substeps: " << nsubSteps << "\tdt: " << dt_radiation << "\n";
	}
	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps <= (maxSubsteps_ + 1));
	AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);

	// perform subcycle
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real time_subcycle = time;
	for (int i = 0; i < nsubSteps; ++i) {
		if (i > 0) {
			// since we are starting a new substep, we need to copy radiation state from
			//     new state vector to old state vector
			// (this is not necessary for the i=0 substep because we have already swapped
			//  the full hydro+radiation state vectors at the beginning of the level advance)
			swapRadiationState(state_old_cc_[lev], state_new_cc_[lev]);
		}

		// We use the IMEX PD-ARS scheme to evolve the radiation subsystem and radiation-matter coupling.

		// Stage 1: advance hyperbolic radiation subsystem using Forward Euler method, starting from state_old_cc_ to state_new_cc_
		advanceRadiationForwardEuler(lev, time_subcycle, dt_radiation, i, nsubSteps, fr_as_crse, fr_as_fine);

		// new radiation state is stored in state_new_cc_
		// new hydro state is stored in state_new_cc_ (always the case during radiation update)

		// failure counter for: matter-radiation coupling, dust temperature, outer iteration
		amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
		// iteration counter for: radiation update, Newton-Raphson iterations, max Newton-Raphson iterations, decoupled gas-dust update
		amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
		int *p_iteration_failure_counter = iteration_failure_counter.data();
		int *p_iteration_counter = iteration_counter.data();

		// Create a MultiFab to hold radEnergySource for the current AMR level
		// radEnergySource should have the unit of luminosity density, erg s^-1 cm^-3
		int const nghost = 1; // depositRadiation needs 1 ghost cell
		amrex::MultiFab radEnergySource(grids[lev], dmap[lev], Physics_Traits<problem_t>::nGroups, nghost);

		if constexpr (IMEX_a22 > 0.0) {
			// matter-radiation exchange source terms of stage 1

			radEnergySource.setVal(0.0); // Initialize the MultiFab to zero

			// for debugging, print the radEnergySource array
			// if (i == 0) {
			// 	amrex::Print() << "Initial,              ";
			// 	PrintRadEnergySource(radEnergySource);
			// 	amrex::Print() << "\n";
			// }

			// Deposit radiation from all particles that have luminosity. When there are no particles with luminosity, this will do nothing.
			particleRegister_.depositRadiation(radEnergySource, lev, time_subcycle);

			// for debugging, print the radEnergySource array
			// if (i == 0) {
			// 	amrex::Print() << "after ParticleToMesh: ";
			// 	PrintRadEnergySource(radEnergySource);
			// 	amrex::Print() << "\n";
			// }

			for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
				const amrex::Box &indexRange = iter.validbox();
				auto const &stateNew_cc = state_new_cc_[lev].array(iter);
				auto const &prob_lo = geom[lev].ProbLoArray();
				auto const &prob_hi = geom[lev].ProbHiArray();

				auto const &radEnergySource_arr = radEnergySource.array(iter);
				RadSystem<problem_t>::SetRadEnergySource(radEnergySource_arr, indexRange, dx, prob_lo, prob_hi, time_subcycle + dt_radiation);

				// update state_new_cc_[lev] in place (updates both radiation and hydro vars)
				// Note that only a fraction (IMEX_a32) of the matter-radiation exchange source terms are added to hydro. This ensures that the
				// hydro properties get to t + IMEX_a32 dt in terms of matter-radiation exchange.
				if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
					RadSystem<problem_t>::AddSourceTermsSingleGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_radiation, 1,
											dustGasInteractionCoeff_, rad_tol, rad_tol_rel, tempFloor,
											p_iteration_counter, p_iteration_failure_counter);
				} else {
					RadSystem<problem_t>::AddSourceTermsMultiGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_radiation, 1,
										       dustGasInteractionCoeff_, rad_tol, rad_tol_rel, tempFloor,
										       p_iteration_counter, p_iteration_failure_counter);
				}
			}
		}

		// Stage 2: advance hyperbolic radiation subsystem using midpoint RK2 method, starting from state_old_cc_ to state_new_cc_
		advanceRadiationMidpointRK2(lev, time_subcycle, dt_radiation, i, nsubSteps, fr_as_crse, fr_as_fine);

		// new radiation state is stored in state_new_cc_
		// new hydro state is stored in state_new_cc_ (always the case during radiation update)

		radEnergySource.setVal(0.0); // Initialize the MultiFab to zero

		// Deposit radiation from particles into radEnergySource. When there are no particles with luminosity, this will do nothing.
		particleRegister_.depositRadiation(radEnergySource, lev, time_subcycle);

		// Add the matter-radiation exchange source terms to the radiation subsystem and evolve by (1 - IMEX_a32) * dt
		for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateNew_cc = state_new_cc_[lev].array(iter);
			auto const &prob_lo = geom[lev].ProbLoArray();
			auto const &prob_hi = geom[lev].ProbHiArray();

			auto const &radEnergySource_arr = radEnergySource.array(iter);
			RadSystem<problem_t>::SetRadEnergySource(radEnergySource_arr, indexRange, dx, prob_lo, prob_hi, time_subcycle + dt_radiation);

			// include cell-centered source terms; will update state_new_cc_[lev] in place (updates both radiation and hydro vars)
			if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
				RadSystem<problem_t>::AddSourceTermsSingleGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_radiation, 2,
										dustGasInteractionCoeff_, rad_tol, rad_tol_rel, tempFloor, p_iteration_counter,
										p_iteration_failure_counter);
			} else {
				RadSystem<problem_t>::AddSourceTermsMultiGroup(stateNew_cc, radEnergySource_arr, indexRange, dt_radiation, 2,
									       dustGasInteractionCoeff_, rad_tol, rad_tol_rel, tempFloor, p_iteration_counter,
									       p_iteration_failure_counter);
			}
		}

		if (print_rad_counter_) {
			auto *h_iteration_counter = iteration_counter.copyToHost();
			long global_solver_count = h_iteration_counter[0];  // number of Newton-Raphson solvings, NOLINT(google-runtime-int)
			long global_iteration_sum = h_iteration_counter[1]; // sum of Newton-Raphson iterations, NOLINT(google-runtime-int)
			int global_iteration_max = h_iteration_counter[2];  // max number of Newton-Raphson iterations, NOLINT(google-runtime-int)
			// sum of decoupled gas-dust Newton-Raphson iterations
			long global_decoupled_iteration_sum = h_iteration_counter[3]; // NOLINT(google-runtime-int)

			amrex::ParallelDescriptor::ReduceLongSum(global_solver_count);
			amrex::ParallelDescriptor::ReduceLongSum(global_iteration_sum);
			amrex::ParallelDescriptor::ReduceIntMax(global_iteration_max);
			amrex::ParallelDescriptor::ReduceLongSum(global_decoupled_iteration_sum);

			if (amrex::ParallelDescriptor::IOProcessor()) {
				const auto n_cells = CountCells(lev);
				amrex::Print() << "time_subcycle = " << time_subcycle << ", total number of cells updated is " << n_cells << "\n";
				if (n_cells > 0 && global_solver_count > 0) {
					const double global_iteration_mean =
					    static_cast<double>(global_iteration_sum) / static_cast<double>(global_solver_count);
					const double global_solving_mean =
					    static_cast<double>(global_solver_count) / static_cast<double>(n_cells) / 2.0; // 2 stages
					const double global_decoupled_iteration_mean =
					    static_cast<double>(global_decoupled_iteration_sum) / static_cast<double>(global_solver_count);
					amrex::Print() << "The average number of Newton-Raphson solvings per IMEX stage is " << global_solving_mean
						       << ", (mean, max) number of Newton-Raphson iterations are " << global_iteration_mean << ", "
						       << global_iteration_max << ".\n";
					if constexpr (ISM_Traits<problem_t>::enable_dust_gas_thermal_coupling_model) {
						amrex::Print() << "The fraction of gas-dust interactions that are decoupled is "
							       << global_decoupled_iteration_mean << "\n";
					}
				}
			}
		}

		auto *h_iteration_failure_counter = iteration_failure_counter.copyToHost();
		long nf_coupling = h_iteration_failure_counter[0]; // number of matter-radiation coupling failures, NOLINT(google-runtime-int)
		long nf_dust = h_iteration_failure_counter[1];	   // number of dust temperature failures, NOLINT(google-runtime-int)
		long nf_outer = h_iteration_failure_counter[2];	   // number of outer iterations failures, NOLINT(google-runtime-int)

		amrex::ParallelDescriptor::ReduceLongSum(nf_coupling);
		amrex::ParallelDescriptor::ReduceLongSum(nf_dust);
		amrex::ParallelDescriptor::ReduceLongSum(nf_outer);

		// Note that the nf_dust has to abort BEFORE nf_coupling, because the dust temperature is used in the matter-radiation coupling and if dust
		// temperature is negative, the matter-radiation coupling will fail to converge.
		if (nf_dust > 0) {
			amrex::Abort("Newton-Raphson iteration for dust temperature failed to converge or dust temperature is negative!");
		}
		if (nf_coupling > 0) {
			amrex::Abort("Newton-Raphson iteration for matter-radiation coupling failed to converge!");
		}
		if (nf_outer > 0) {
			amrex::Abort("Outer iteration for matter-radiation coupling failed to converge!");
		}

		// new hydro+radiation state is stored in state_new_cc_

		// update 'time_subcycle'
		time_subcycle += dt_radiation;

		// update cell update counter
		radiationCellUpdates_ += CountCells(lev); // keep track of number of cell updates
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int const /*iter_count*/,
							       int const /*nsubsteps*/, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine)
{
	// get cell sizes
	auto const &dx = geom[lev].CellSizeArray();

	std::array<amrex::MultiFab, AMREX_SPACEDIM> refluxFluxes;
	if (do_reflux) {
		auto initRefluxFluxes = [&](std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::BoxArray ba = state_new_cc_[lev].boxArray();
				ba.surroundingNodes(dir);
				fluxes[dir].define(ba, dmap[lev], state_new_cc_[lev].nComp(), 0);
				fluxes[dir].setVal(0.0);
			}
		};
		initRefluxFluxes(refluxFluxes);
	}

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_cc_[lev], state_old_cc_[lev], lev, time, quokka::centering::cc, quokka::direction::na, PreInterpState,
			       PostInterpState);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld_cc = state_old_cc_[lev].const_array(iter);
		auto const &stateNew_cc = state_new_cc_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] = computeRadiationFluxes(stateOld_cc, indexRange, ncompHyperbolic_, dx);

		// Stage 1 of RK2-SSP
		RadSystem<problem_t>::PredictStep(
		    stateOld_cc, stateNew_cc, {AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(), fluxArrays[2].array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(), fluxDiffusiveArrays[1].const_array(), fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			auto expandedFluxes = expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				refluxFluxes[dir][iter].copy<amrex::RunOn::Device>(expandedFluxes[dir], expandedFluxes[dir].box(), 0, expandedFluxes[dir].box(),
										   0, refluxFluxes[dir].nComp());
			}
		}
	}

	if (do_reflux) {
		incrementFluxRegisters(fr_as_crse, fr_as_fine, refluxFluxes, lev, 0.5 * dt_radiation);
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int const /*iter_count*/,
							      int const /*nsubsteps*/, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine)
{
	auto const &dx = geom[lev].CellSizeArray();

	std::array<amrex::MultiFab, AMREX_SPACEDIM> refluxFluxes;
	if (do_reflux) {
		auto initRefluxFluxes = [&](std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes) {
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				amrex::BoxArray ba = state_new_cc_[lev].boxArray();
				ba.surroundingNodes(dir);
				fluxes[dir].define(ba, dmap[lev], state_new_cc_[lev].nComp(), 0);
				fluxes[dir].setVal(0.0);
			}
		};
		initRefluxFluxes(refluxFluxes);
	}

	// update ghost zones [intermediate stage stored in state_new_cc_]
	fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, (time + dt_radiation), quokka::centering::cc, quokka::direction::na, PreInterpState,
			       PostInterpState);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld_cc = state_old_cc_[lev].const_array(iter);
		auto const &stateInter_cc = state_new_cc_[lev].const_array(iter);
		auto const &stateNew_cc = state_new_cc_[lev].array(iter);
		auto [fluxArraysOld, fluxDiffusiveArraysOld] = computeRadiationFluxes(stateOld_cc, indexRange, ncompHyperbolic_, dx);
		auto [fluxArrays, fluxDiffusiveArrays] = computeRadiationFluxes(stateInter_cc, indexRange, ncompHyperbolic_, dx);

		// Stage 2 of RK2-SSP
		RadSystem<problem_t>::AddFluxesRK2(
		    stateNew_cc, stateOld_cc, stateInter_cc, {AMREX_D_DECL(fluxArraysOld[0].array(), fluxArraysOld[1].array(), fluxArraysOld[2].array())},
		    {AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(), fluxArrays[2].array())},
		    {AMREX_D_DECL(fluxDiffusiveArraysOld[0].const_array(), fluxDiffusiveArraysOld[1].const_array(), fluxDiffusiveArraysOld[2].const_array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(), fluxDiffusiveArrays[1].const_array(), fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			auto expandedFluxes = expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
			for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
				refluxFluxes[dir][iter].copy<amrex::RunOn::Device>(expandedFluxes[dir], expandedFluxes[dir].box(), 0, expandedFluxes[dir].box(),
										   0, refluxFluxes[dir].nComp());
			}
		}
	}

	if (do_reflux) {
		incrementFluxRegisters(fr_as_crse, fr_as_fine, refluxFluxes, lev, 0.5 * dt_radiation);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars,
							 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>>
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

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, x1Flux, x1FluxDiffusive, indexRange, nvars, dx);
		     , fluxFunction<FluxDir::X2>(consVar, x2Flux, x2FluxDiffusive, indexRange, nvars, dx);
		     , fluxFunction<FluxDir::X3>(consVar, x3Flux, x3FluxDiffusive, indexRange, nvars, dx);)

	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxArrays = {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxDiffusiveArrays{
	    AMREX_D_DECL(std::move(x1FluxDiffusive), std::move(x2FluxDiffusive), std::move(x3FluxDiffusive))};

	return std::make_tuple(std::move(fluxArrays), std::move(fluxDiffusiveArrays));
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
					       const amrex::Box &indexRange, const int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
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
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_cc_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	if (radiationReconstructionOrder_ == 5) {
		// cell-centered kernel
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
										   x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 3) {
		// mixed interface/cell-centered kernel
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
										x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 2) {
		// PLM and donor cell are interface-centered kernels
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::MC>(primVar.array(), x1LeftState.array(), x1RightState.array(),
												  x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 1) {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar.array(), x1LeftState.array(), x1RightState.array(),
										     x1ReconstructRange, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order for radiation variables! Aborting...");
	}

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(), x1LeftState.array(), x1RightState.array(), x1FluxRange,
							  consState, dx, use_wavespeed_correction_); // watch out for argument order!!
}

// Save single-level plotfile
// This is a wrapper around the WriteSingleLevelPlotfile function in the AMReX library.
// The step number of the plotfile is set to istep[lev] and the time is set to the current time tNew_[lev].
// Example usage: write debug_rhs00000 debug_rhs00001 etc with interval plotfileInterval_
//   const int lev_debug = 0;
//   amrex::Vector<std::string> flatCompNames{"rhs"};
//   WriteSingleLevelPlotfileSimplified("debug_rhs", rhs[lev_debug], flatCompNames, lev_debug, plotfileInterval_);
template <typename problem_t>
void QuokkaSimulation<problem_t>::WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf,
								     const amrex::Vector<std::string> &compNames, int lev, int interval)
{
	if ((istep[lev] % interval) != 0) {
		return;
	}
	const auto plotfile_name = CustomPlotFileName(plotfile_prefix.c_str(), istep[lev]);
	WriteSingleLevelPlotfile(plotfile_name, mf, compNames, geom[lev], tNew_[lev], istep[lev]);
}

#endif // RADIATION_SIMULATION_HPP_
