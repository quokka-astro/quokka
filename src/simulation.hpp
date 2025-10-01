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
#include "AMReX_MFInterpolater.H"
#include "AMReX_Periodicity.H"
#include "AMReX_String.H"
#include <cfenv>
#include <cmath>
#include <cstdio>
#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std
{
namespace filesystem = experimental::filesystem;
}
#endif
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <variant>

// library headers
#include "AMReX.H"
#include "AMReX_AmrCore.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_AsyncOut.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_EdgeFluxRegister.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FillPatchUtil.H"
#include "AMReX_FillPatcher.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_INT.H"
#include "AMReX_IndexType.H"
#include "AMReX_IntVect.H"
#include "AMReX_Interpolater.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Orientation.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Utility.H"
#include "AMReX_Vector.H"
#include "AMReX_VisMF.H"
#include <AMReX_FluxRegister.H>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <yaml-cpp/yaml.h>

#include "AMReX_AmrParticles.H"
#include "particles/PhysicsParticles.hpp"

#if AMREX_SPACEDIM == 3
#include "AMReX_MLLinOp.H"
#include "AMReX_MLMG.H"
#include "AMReX_MLPoisson.H"
#include "AMReX_OpenBC.H"
#endif

#ifdef AMREX_USE_ASCENT
#include <AMReX_Conduit_Blueprint.H>
#include <ascent.hpp>
#endif

// internal headers
#include "fundamental_constants.H"
#include "grid.hpp"
#include "io/DiagBase.H"
#include "io/DiagPlotfile.H"
#include "io/DiagProjectionPlot.H"
#include "io/io_utils.hpp"
#include "io/projection.hpp"
#include "physics_info.hpp"

#ifdef QUOKKA_USE_OPENPMD
#include "io/openPMD.hpp"
#endif

#ifdef AMREX_USE_ASCENT
using namespace conduit;
using namespace ascent;
#endif

// Quokka version string to be stored in metadata. This is used in post-processing tools like YT to do version checks.
static constexpr auto QUOKKA_VERSION = "25.03";

template <> struct fmt::formatter<amrex::IntVect> : formatter<std::vector<int>> {
	// parse is inherited from formatter<std::vector<int>>.
	auto format(amrex::IntVect iv, format_context &ctx) const -> format_context::iterator
	{
		std::vector<int> const vec{AMREX_D_DECL(iv[0], iv[1], iv[2])};
		return formatter<std::vector<int>>::format(vec, ctx);
	};
};

using variant_t = std::variant<amrex::Real, std::string>;

namespace YAML
{
template <typename T> struct as_if<T, std::optional<T>> {
	explicit as_if(const Node &node_) : node(node_) {}
	const Node &node; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
	auto operator()() const -> std::optional<T>
	{
		std::optional<T> val;
		T t;
		if ((node.m_pNode != nullptr) && convert<T>::decode(node, t)) {
			val = std::move(t);
		}
		return val;
	}
};

// There is already a std::string partial specialisation,
// so we need a full specialisation here
template <> struct as_if<std::string, std::optional<std::string>> {
	explicit as_if(const Node &node_) : node(node_) {}
	const Node &node; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
	auto operator()() const -> std::optional<std::string>
	{
		std::optional<std::string> val;
		std::string t;
		if ((node.m_pNode != nullptr) && convert<std::string>::decode(node, t)) {
			val = std::move(t);
		}
		return val;
	}
};
} // namespace YAML

enum class FillPatchType { fillpatch_class, fillpatch_function };

// Main simulation class; solvers should inherit from this
template <typename problem_t> class AMRSimulation : public amrex::AmrCore
{
      public:
	amrex::Real maxDt_ = std::numeric_limits<double>::max();  // no limit by default
	amrex::Real initDt_ = std::numeric_limits<double>::max(); // no limit by default
	amrex::Real constantDt_ = 0.0;
	amrex::Vector<int> istep;	      // which step?
	amrex::Vector<int> nsubsteps;	      // how many substeps on each level?
	amrex::Vector<amrex::Real> tNew_;     // for state_new_cc_
	amrex::Vector<amrex::Real> tOld_;     // for state_old_cc_
	amrex::Vector<amrex::Real> dt_;	      // timestep for each level
	amrex::Real stopTime_ = 1.0;	      // default
	amrex::Real cflNumber_ = 0.3;	      // default
	amrex::Real particleCflNumber_ = 0.5; // default
	amrex::Real dtToleranceFactor_ = 1.1; // default
	amrex::Real dtCutoff_ = 0.0;	      // default: no cutoff (disabled when 0)
	amrex::Long cycleCount_ = 0;
	int printCycleTiming_ = 0;				     // default: don't print
	amrex::Long maxTimesteps_ = std::numeric_limits<int>::max(); // default: no limit
	amrex::Long maxWalltime_ = 0;				     // default: no limit
	int ascentInterval_ = -1;				     // -1 == no in-situ renders with Ascent
	int plotfileInterval_ = -1;				     // -1 == no output
	int projectionInterval_ = -1;				     // -1 == no output
	int statisticsInterval_ = -1;				     // -1 == no output
	int particleInterval_ = -1;				     // -1 == no output
	amrex::Real plotTimeInterval_ = -1.0;			     // time interval for plt file
	bool skipInitialPlotfile_ = false;			     // skip writing plotfile at t=0
	amrex::Real checkpointTimeInterval_ = -1.0;		     // time interval for checkpoints
	int checkpointInterval_ = -1;				     // -1 == no output
	int amrInterpMethod_ = 1;				     // 0 == piecewise constant, 1 == lincc_interp
	int restartRefineFactor_ = 1;				     // 1 == don't refine, >1 == refine by this factor on restart
	amrex::Real reltolPoisson_ = 1.0e-5;			     // default
	amrex::Real abstolPoisson_ = 1.0e-5;			     // default (scaled by minimum RHS value)
	int poissonSupercycleInterval_ = 1;			     // number of coarse steps between Poisson solves (default: 1)
	amrex::Vector<amrex::MultiFab> phi;

	amrex::Real densityFloor_ = 0.0; // default
	amrex::Real tempFloor_ = 0.0;	 // default

	YAML::Node simulationMetadata_;

	// constructor
	explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : BCs_cc_(BCs_cc), BCs_fc_(BCs_fc) { initialize(); }

	explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : BCs_cc_(BCs_cc), BCs_fc_(builtin_BCs_fc(BCs_cc)) { initialize(); }

	auto builtin_BCs_fc(amrex::Vector<amrex::BCRec> & /*BCs_cc*/) -> amrex::Vector<amrex::BCRec>
	{
		static_assert(!(Physics_Traits<problem_t>::is_mhd_enabled), "You are required to explicitly define the face-centered BCs when MHD is enabled.");
		amrex::Vector<amrex::BCRec> BCs_fc(0);
		return BCs_fc;
	}

	void initialize();
	void PerformanceHints();
	void readParameters();
	void rereadRuntimeParameters(); // Re-read parameters to ensure runtime values override compile-time settings
	void setInitialConditions();
	void setInitialConditionsAtLevel_cc(int level, amrex::Real time);
	void setInitialConditionsAtLevel_fc(int level, amrex::Real time);
	void evolve();
	void computeTimestep();
	auto computeTimestepAtLevel(int lev) -> amrex::ValLocPair<amrex::Real, amrex::IntVect>;

	void AverageFCToCC(amrex::MultiFab &mf_cc, const amrex::MultiFab &mf_fc, int idim, int dstcomp_start, int srccomp_start, int srccomp_total) const;

	virtual void computeMaxSignalLocal(int level) = 0;
	virtual void printCellProperties(int lev, amrex::IntVect const &index) = 0;
	virtual void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) = 0;
	virtual void preCalculateInitialConditions() = 0;
	virtual void setInitialConditionsOnGrid(quokka::grid const &grid_elem) = 0;
	virtual void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) = 0;
	virtual void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) = 0;
	virtual void createInitialRadParticles() = 0;
#if AMREX_SPACEDIM == 3
	virtual void createInitialCICParticles() = 0;
	virtual void createInitialCICRadParticles() = 0;
	virtual void createInitialStochasticStellarPopParticles() = 0;
	virtual void createInitialSinkParticles() = 0;
	virtual void createInitialTestParticles() = 0;
	// Test particles have integer components, and InitFromAsciiFile does not support integer components, so we do not allow creating them at the start
	// of the simulation
#endif // AMREX_SPACEDIM == 3
	virtual void computeBeforeTimestep() = 0;
	virtual void computeAfterTimestep() = 0;
	virtual void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) = 0;
	virtual void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) = 0;
	virtual void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) = 0;
	virtual void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf,
							const amrex::Vector<std::string> &compNames, int lev, int interval) = 0;

	// compute derived variables
	virtual void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const = 0;

	// compute projected vars
	[[nodiscard]] virtual auto ComputeProjections(amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> = 0;

	// compute statistics
	virtual auto ComputeStatistics() -> std::map<std::string, amrex::Real> = 0;

	// fix-up any unphysical states created by AMR operations
	// (e.g., caused by the flux register or from interpolation)
	virtual void FixupState(int level) = 0;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override = 0;

	// Make a new level using provided BoxArray and DistributionMapping
	void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;

	// Remake an existing level using provided BoxArray and DistributionMapping
	void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;

	// Delete level data
	void ClearLevel(int lev) override;

	// Make a new level from scratch using provided BoxArray and
	// DistributionMapping
	void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;

	// AMR utility functions
	template <typename PreInterpHook, typename PostInterpHook>
	void fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state, int lev, amrex::Real time, quokka::centering cen, quokka::direction dir,
				    PreInterpHook const &pre_interp, PostInterpHook const &post_interp, FillPatchType fptype = FillPatchType::fillpatch_class);

	template <typename PreInterpHook, typename PostInterpHook>
	void FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf, amrex::Vector<amrex::MultiFab *> &coarseData,
			       amrex::Vector<amrex::Real> &coarseTime, amrex::Vector<amrex::MultiFab *> &fineData, amrex::Vector<amrex::Real> &fineTime,
			       int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering &cen, quokka::direction dir, FillPatchType fptype,
			       PreInterpHook const &pre_interp, PostInterpHook const &post_interp);

	static void InterpHookNone(amrex::MultiFab &mf, int scomp, int ncomp);
	virtual void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
			       FillPatchType fptype);

	auto getAmrInterpolaterCellCentered() -> amrex::MFInterpolater *;
	auto getAmrInterpolaterFaceCentered() -> amrex::Interpolater *;
	void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering cen,
			     quokka::direction dir);
	void FillCoarsePatchFaceArray(int lev, amrex::Real time, amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> &mf_array, int icomp, int ncomp,
				      amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> &BCs_array);
	void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data, amrex::Vector<amrex::Real> &datatime, quokka::centering cen,
		     quokka::direction dir);
	void GetDataFaceArray(int lev, amrex::Real time, amrex::Array<amrex::Vector<amrex::MultiFab *>, AMREX_SPACEDIM> &data_array,
			      amrex::Vector<amrex::Real> &datatime);
	void AverageDown();
	void AverageDownTo(int crse_lev);
	void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);
	void calculateGpotAllLevels();
	void gravAccelAllLevels(amrex::Real dt);
	void ellipticSolveAllLevels(amrex::Real dt);

	void incrementFluxRegisters(amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays,
				    int lev, amrex::Real dt_lev);
	void incrementEMFRegisters(amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine,
				   std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_emf_components, int lev, amrex::Real dt_lev);

	void particleMeshInteraction(amrex::Real time, amrex::Real dt);

	// boundary condition
	AMREX_GPU_DEVICE static void setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
								 amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr, int bcomp,
								 int orig_comp); // template specialized by problem generator

	// boundary condition
	template <quokka::direction dir>
	AMREX_GPU_DEVICE static void setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp,
									int numcomp, amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr,
									int bcomp, int orig_comp); // template specialized by problem generator

	// compute volume integrals
	template <typename F> auto computeVolumeIntegral(F const &user_f) -> amrex::Real;

	// I/O functions
	[[nodiscard]] auto PlotFileName(int lev) const -> std::string;
	[[nodiscard]] auto CustomPlotFileName(const char *base, int lev) const -> std::string;
	[[nodiscard]] auto GetPlotfileVarNames() const -> amrex::Vector<std::string>;
	[[nodiscard]] auto GetPlotfileVarNames_fc() const -> std::array<amrex::Vector<std::string>, AMREX_SPACEDIM>;
	[[nodiscard]] auto PlotFileMF_cc(int included_ghosts) -> amrex::Vector<amrex::MultiFab>;
	[[nodiscard]] auto PlotFileMFAtLevel_cc(int lev, int included_ghosts) -> amrex::MultiFab;
	[[nodiscard]] auto PlotFileMF_fc(int nghost_fc_) -> std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM>;
	[[nodiscard]] auto PlotFileMFAtLevel_fc(int lev, int idim, int nghost_fc_) -> amrex::MultiFab;
	void createDiagnostics();
	void updateDiagnostics();
	void doDiagnostics();
	void WriteMetadataFile(std::string const &MetadataFileName) const;
	void ReadMetadataFile(std::string const &chkfilename);
	void WriteStatisticsFile();
	void WritePlotFile();
	void WriteParticleFile();
	void WriteCheckpointFile() const;
	void SetLastCheckpointSymlink(std::string const &checkpointname) const;
	void writeFaceVelocitiesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVel, int lev, int step);
	void writeReconstructedStatesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &leftState,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &rightState, int lev, int step);

	// ABOUTME: Used to handle universal refinement during checkpoint restart operations
	struct RefinementContext {
		int refinement_factor = 1;
		amrex::Geometry coarse_level0_geom;
		[[nodiscard]] auto needs_refinement() const -> bool { return refinement_factor > 1; }
	};

	void ReadCheckpointFile();

	// Helper methods for checkpoint restart refactoring
	auto detectRefinementContext(const amrex::BoxArray &restart_ba, const amrex::Geometry &current_geom) -> RefinementContext;
	auto readCheckpointHeader(const std::string &restart_file) -> amrex::Vector<amrex::BoxArray>;
	void interpolateMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source, const RefinementContext &context,
					    const amrex::Geometry &coarse_geom, const amrex::Geometry &fine_geom, const amrex::Vector<amrex::BCRec> &bcs);
	void interpolateFaceCenteredMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source, const RefinementContext &context,
							const amrex::Geometry &coarse_geom, const amrex::Geometry &fine_geom, quokka::direction dir);
	void loadMultiFabData(const RefinementContext &context);
	auto loadBalanceOnRestart(const amrex::BoxArray &input_ba, int lev) -> amrex::BoxArray;

	template <typename ParticleContainer>
	void restartParticleContainerWithRefinement(std::unique_ptr<ParticleContainer> &particles, std::string const &restart_chkfile,
						    std::string const &particle_type_name, amrex::Vector<amrex::BoxArray> const &header_box_arrays);

	template <typename ContainerType>
	void initializeParticleContainerFromCheckpoint(std::unique_ptr<ContainerType> &container, quokka::ParticleType particle_type,
						       amrex::Vector<amrex::BoxArray> const &header_box_arrays);

	auto getGitHashForQuokka() const -> std::string;
	auto getGitHashForAmrex() const -> std::string;
	auto getWalltime() -> amrex::Real;
	auto getCycleWalltime() -> amrex::Real;
	void setChkFile(std::string const &chkfile_number);
	[[nodiscard]] auto getOldMF_fc() const -> amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &;
	[[nodiscard]] auto getNewMF_fc() const -> amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &;

	// particle functions
#if AMREX_SPACEDIM == 3
	void kickParticlesAllLevels(amrex::Real dt);
#endif // AMREX_SPACEDIM == 3

	// simulation metadata
	void initializeSimulationMetadata();

#ifdef AMREX_USE_ASCENT
	void AscentCustomActions(conduit::Node const &blueprintMesh);
	void RenderAscent();
#endif
      protected:
	amrex::Vector<amrex::BCRec> BCs_cc_; // on level 0
	amrex::Vector<amrex::BCRec> BCs_fc_; // on level 0
	amrex::Vector<amrex::MultiFab> state_old_cc_;
	amrex::Vector<amrex::MultiFab> state_new_cc_;
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> state_old_fc_;
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> state_new_fc_;
	amrex::Vector<amrex::MultiFab> max_signal_speed_; // needed to compute CFL timestep

	// flux registers: store fluxes at coarse-fine interface for synchronization
	// this will be sized "nlevs_max+1"
	// NOTE: the flux register associated with flux_reg[lev] is associated with
	// the lev/lev-1 interface (and has grid spacing associated with lev-1)
	// therefore flux_reg[0] and flux_reg[nlevs_max] are never actually used in
	// the reflux operation
	amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg_;
	amrex::Vector<std::unique_ptr<amrex::EdgeFluxRegister>> emf_reg_;

	// This is for fillpatch during timestepping, but not for regridding.
	amrex::Vector<std::unique_ptr<amrex::FillPatcher<amrex::MultiFab>>> fillpatcher_;

	// Nghost = number of ghost cells for each array
	// For our new scheme MHD-scheme, we need 7 ghosts for MHD (4 base + 3 for EMF) or 6 otherwise
	int nghost_cc_ = Physics_Traits<problem_t>::is_mhd_enabled ? 7 : 6;
	int nghost_fc_ = nghost_cc_;

	amrex::Vector<std::string> componentNames_cc_;
	amrex::Vector<std::string> componentNames_fc_flat_;
	std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> componentNames_fc_;
	amrex::Vector<std::string> derivedNames_;
	amrex::Vector<std::string> plotfileVarsToInclude_cc_;
	bool areInitialConditionsDefined_ = false;

	/// output parameters
	std::string plot_file{"plt"};	       // plotfile prefix
	std::string chk_file{"chk"};	       // checkpoint prefix
	std::string stats_file{"history.txt"}; // statistics filename
	int plot_nfiles = -1;		       // default: -1 (i.e., one file per process)
	int checkpoint_nfiles = -1;	       // default: -1 (i.e., one file per process)
	/// input parameters (if >= 0 we restart from a checkpoint)
	std::string restart_chkfile;

	// Diagnostics
	amrex::Vector<std::unique_ptr<DiagBase>> m_diagnostics;
	amrex::Vector<std::string> m_diagVars;

	/// AMR-specific parameters
	int regrid_int = 2;	 // regrid interval (number of coarse steps)
	int do_reflux = 1;	 // 1 == reflux, 0 == no reflux
	int do_subcycle = 1;	 // 1 == subcycle, 0 == no subcyle
	int suppress_output = 0; // 1 == show timestepping, 0 == do not output each timestep

	// performance metrics
	amrex::Long cellUpdates_ = 0;
	amrex::Vector<amrex::Long> cellUpdatesEachLevel_;

	// gravity
	static constexpr amrex::Real Gconst_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::Gconst; // gravitational constant G, CGS units
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::gravitational_constant; // gravitational constant G, user defined
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// G / G_bar = u_l^3 / u_m / u_t^2
			return C::Gconst /
			       (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length /
				Physics_Traits<problem_t>::unit_mass / (Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time));
		}
	}();

	// unit length, mass, time, temperature
	static constexpr amrex::Real unit_length = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return Physics_Traits<problem_t>::unit_length;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return 1.0;
		} else {
			return NAN;
		}
	}();
	static constexpr amrex::Real unit_mass = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return Physics_Traits<problem_t>::unit_mass;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return 1.0;
		} else {
			return NAN;
		}
	}();
	static constexpr amrex::Real unit_time = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return Physics_Traits<problem_t>::unit_time;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return 1.0;
		} else {
			return NAN;
		}
	}();
	static constexpr amrex::Real unit_temperature = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			return Physics_Traits<problem_t>::unit_temperature;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return 1.0;
		} else {
			return NAN;
		}
	}();

	// tracer particles
      public:
	int do_tracers = 0;

      protected:
	void InitParticles();	 // create tracer particles
	void InitPhyParticles(); // create PhysicsParticles
	std::unique_ptr<amrex::AmrTracerParticleContainer> TracerPC;
	std::unique_ptr<quokka::RadParticleContainer<problem_t>> RadParticles;
#if AMREX_SPACEDIM == 3
	std::unique_ptr<quokka::CICParticleContainer> CICParticles;
	std::unique_ptr<quokka::CICRadParticleContainer<problem_t>> CICRadParticles;
	std::unique_ptr<quokka::StochasticStellarPopParticleContainer<problem_t>> StochasticStellarPopParticles;
	std::unique_ptr<quokka::SinkParticleContainer> SinkParticles;
	std::unique_ptr<quokka::TestParticleContainer<problem_t>> TestParticles;
#endif // AMREX_SPACEDIM == 3

	// external objects
#ifdef AMREX_USE_ASCENT
	Ascent ascent_;
#endif

	// Add PhysicsParticleRegister member
	quokka::PhysicsParticleRegister<problem_t> particleRegister_;
};

template <typename problem_t> auto AMRSimulation<problem_t>::getGitHashForQuokka() const -> std::string
{
	// NOTE: this is defined by a preprocessor macro in CMakeLists.txt
	return QUOKKA_GIT_HASH;
}

template <typename problem_t> auto AMRSimulation<problem_t>::getGitHashForAmrex() const -> std::string
{
	// NOTE: this is defined by a preprocessor macro in CMakeLists.txt
	return AMREX_GIT_HASH;
}

template <typename problem_t> void AMRSimulation<problem_t>::setChkFile(std::string const &chkfile_number) { restart_chkfile = chkfile_number; }

template <typename problem_t> auto AMRSimulation<problem_t>::getOldMF_fc() const -> const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> &
{
	return state_old_fc_;
}

template <typename problem_t> auto AMRSimulation<problem_t>::getNewMF_fc() const -> const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> &
{
	return state_new_fc_;
}

template <typename problem_t> void AMRSimulation<problem_t>::initialize()
{
	BL_PROFILE("AMRSimulation::initialize()"); // NOLINT(misc-const-correctness)

	readParameters();

	// print derived vars
	if (!derivedNames_.empty()) {
		amrex::Print() << "Using derived variables:\n";
		for (auto const &name : derivedNames_) {
			amrex::Print() << "\t" << name << "\n";
		}
		amrex::Print() << "\n";
	}

	int nlevs_max = max_level + 1;
	istep.resize(nlevs_max, 0);
	nsubsteps.resize(nlevs_max, 1);
	if (do_subcycle == 1) {
		for (int lev = 1; lev <= max_level; ++lev) {
			nsubsteps[lev] = MaxRefRatio(lev - 1);
		}
	}

	tNew_.resize(nlevs_max, 0.0);
	tOld_.resize(nlevs_max, -1.e100);
	dt_.resize(nlevs_max, 1.e100);
	state_new_cc_.resize(nlevs_max);
	state_old_cc_.resize(nlevs_max);
	state_new_fc_.resize(nlevs_max);
	state_old_fc_.resize(nlevs_max);
	max_signal_speed_.resize(nlevs_max);
	flux_reg_.resize(nlevs_max + 1);
	emf_reg_.resize(nlevs_max + 1);
	fillpatcher_.resize(nlevs_max + 1);
	cellUpdatesEachLevel_.resize(nlevs_max, 0);

	// check that grids will be properly nested on each level
	// (this is necessary since FillPatch only fills from non-ghost cells on
	// lev-1)
	auto checkIsProperlyNested = [this](int const lev, amrex::IntVect const &blockingFactor) {
		return amrex::ProperlyNested(refRatio(lev - 1), blockingFactor, nghost_cc_, amrex::IndexType::TheCellType(), &amrex::cell_cons_interp);
	};

	for (int lev = 1; lev <= max_level; ++lev) {
		if (!checkIsProperlyNested(lev, blocking_factor[lev])) {
			// level lev is not properly nested
			amrex::Print() << "Blocking factor is too small for proper grid nesting! "
					  "Increase blocking factor to >= ceil(nghost,ref_ratio)*ref_ratio."
				       << std::endl; // NOLINT(performance-avoid-endl)
			amrex::Abort("Grids not properly nested!");
		}
	}

	// check that ncells >= nghost_cc_ in each dimension
	// (this is necessary to prevent out-of-bounds access when filling ghost cells)
	const amrex::Box domain = Geom()[0].Domain();
	const amrex::IntVect ncells{
	    AMREX_D_DECL(domain.bigEnd(0) - domain.smallEnd(0) + 1, domain.bigEnd(1) - domain.smallEnd(1) + 1, domain.bigEnd(2) - domain.smallEnd(2) + 1)};

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		if (ncells[idim] < nghost_cc_) {
			amrex::Print() << "Number of cells in dimension " << idim << " (" << ncells[idim] << ") is less than nghost_cc_ (" << nghost_cc_
				       << ")! "
				       << "This will cause out-of-bounds access when filling ghost cells." << std::endl; // NOLINT(performance-avoid-endl)
			amrex::Abort("Insufficient grid resolution for ghost cell operations!");
		}
	}

	// add Quokka version to metadata
	simulationMetadata_["quokka_version"] = QUOKKA_VERSION;

	// add git commit to metadata
	simulationMetadata_["git_hash_quokka"] = getGitHashForQuokka();
	simulationMetadata_["git_hash_amrex"] = getGitHashForAmrex();

	// add units and physics-specific metadata
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		initializeSimulationMetadata();
	}

#ifdef AMREX_USE_ASCENT
	// initialize Ascent
	conduit::Node ascent_options;
	ascent_options["mpi_comm"] = MPI_Comm_c2f(amrex::ParallelContext::CommunicatorSub());
	ascent_.open(ascent_options);
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::PerformanceHints()
{
	// Check requested MPI ranks and available boxes
	for (int ilev = 0; ilev <= finestLevel(); ++ilev) {
		const amrex::Long nboxes = boxArray(ilev).size();
		if (amrex::ParallelDescriptor::NProcs() > nboxes) {
			amrex::Print() << "\n[Warning] [Performance] Too many resources / too little work!\n"
				       << "  It looks like you requested more compute resources than "
				       << "  the number of boxes of cells available on level " << ilev << " (" << nboxes << "). "
				       << "You started with (" << amrex::ParallelDescriptor::NProcs() << ") MPI ranks, so ("
				       << amrex::ParallelDescriptor::NProcs() - nboxes << ") rank(s) will have no work on this level.\n"
#ifdef AMREX_USE_GPU
				       << "  On GPUs, consider using 1-8 boxes per GPU per level that "
					  "together fill each GPU's memory sufficiently.\n"
#endif
				       << "\n";
		}
	}

	// check that blocking_factor and max_grid_size are set to reasonable values
#ifdef AMREX_USE_GPU
	const int recommended_blocking_factor = 32;
	const int recommended_max_grid_size = 128;
#else
	const int recommended_blocking_factor = 16;
	const int recommended_max_grid_size = 64;
#endif
	int min_blocking_factor = INT_MAX;
	int min_max_grid_size = INT_MAX;
	for (int ilev = 0; ilev <= finestLevel(); ++ilev) {
		min_blocking_factor = std::min(min_blocking_factor, blocking_factor[ilev].min());
		min_max_grid_size = std::min(min_max_grid_size, max_grid_size[ilev].min());
	}
	if (min_blocking_factor < recommended_blocking_factor) {
		amrex::Print() << "\n[Warning] [Performance] The grid blocking factor (" << min_blocking_factor
			       << ") is too small for reasonable performance. It should be 32 (or "
				  "greater) when running on GPUs, and 16 (or greater) when running on "
				  "CPUs.\n";
	}
	if (min_max_grid_size < recommended_max_grid_size) {
		amrex::Print() << "\n[Warning] [Performance] The maximum grid size (" << min_max_grid_size
			       << ") is too small for reasonable performance. It should be "
				  "128 (or greater) when running on GPUs, and 64 (or "
				  "greater) when running on CPUs.\n";
	}

#ifdef QUOKKA_USE_OPENPMD
	// warning about particles and OpenPMD outputs
	amrex::Print() << "\n[Warning] [I/O] OpenPMD outputs currently do NOT include particles!"
		       << " Support for outputting particles for openPMD is not yet implemented.\n";
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::readParameters()
{
	BL_PROFILE("AMRSimulation::readParameters()"); // NOLINT(misc-const-correctness)

	// ParmParse reads inputs from the *.inputs file
	const amrex::ParmParse pp;

	// Default nsteps == INT_MAX
	pp.query("max_timesteps", maxTimesteps_);

	// Default CFL number == 0.3, set to whatever is in the file
	pp.query("cfl", cflNumber_);

	// Default CFL number for particles == 0.5, set to whatever is in the file
	pp.query("particle_cfl", particleCflNumber_);

	// Default AMR interpolation method == lincc_interp
	pp.query("amr_interpolation_method", amrInterpMethod_);

	// Default stopping time
	pp.query("stop_time", stopTime_);

	// Default timestep cutoff (safety feature)
	pp.query("dt_cutoff", dtCutoff_);

	// Default ascent render interval
	pp.query("ascent_interval", ascentInterval_);

	// Default output interval
	pp.query("plotfile_interval", plotfileInterval_);

	// Default projection interval
	pp.query("projection_interval", projectionInterval_);

	// Default output interval
	pp.query("particle_csv_interval", particleInterval_);

	// Default statistics interval
	pp.query("statistics_interval", statisticsInterval_);

	// Default Time interval
	pp.query("plottime_interval", plotTimeInterval_);

	// Skip initial plotfile
	pp.query("skip_initial_plotfile", skipInitialPlotfile_);

	// Default Time interval
	pp.query("checkpointtime_interval", checkpointTimeInterval_);

	// Default checkpoint interval
	pp.query("checkpoint_interval", checkpointInterval_);

	// Default plotfile prefix
	pp.query("plotfile_prefix", plot_file);

	// Default checkpoint prefix
	pp.query("checkpoint_prefix", chk_file);

	// Default do_reflux = 1
	pp.query("do_reflux", do_reflux);

	// Default do_subcycle = 1
	pp.query("do_subcycle", do_subcycle);

	// Default poisson_supercycle_interval = 1
	pp.query("poisson_supercycle_interval", poissonSupercycleInterval_);

	// Default Poisson solver tolerances
	pp.query("poisson_reltol", reltolPoisson_);
	pp.query("poisson_abstol", abstolPoisson_);

	// Default do_tracers = 0 (turns on/off tracer particles)
	pp.query("do_tracers", do_tracers);

	// Default suppress_output = 0
	pp.query("suppress_output", suppress_output);

	// Default print_cycle_timing = 0
	pp.query("print_cycle_timing", printCycleTiming_);

	// specify this on the command-line in order to restart from a checkpoint
	// file
	pp.query("restartfile", restart_chkfile);

	// Specify derived variables to save to plotfiles
	pp.queryarr("derived_vars", derivedNames_);

	// re-grid interval
	pp.query("regrid_interval", regrid_int);

	// read density floor in g cm^-3
	pp.query("density_floor", densityFloor_);

	// read temperature floor in K
	pp.query("temperature_floor", tempFloor_);

	// specify maximum walltime in HH:MM:SS format
	std::string maxWalltimeInput;
	pp.query("max_walltime", maxWalltimeInput);
	// convert to seconds
	int hours = 0;
	int minutes = 0;
	int seconds = 0;
	int nargs = std::sscanf(maxWalltimeInput.c_str(), "%d:%d:%d", &hours, &minutes, &seconds); // NOLINT
	if (nargs == 3) {
		maxWalltime_ = 3600 * hours + 60 * minutes + seconds;
		amrex::Print() << fmt::format("Setting walltime limit to {} hours, {} minutes, {} seconds.\n", hours, minutes, seconds);
	}

	// IO settings (following the AMReX convention for the Amr class)
	// (Since we use AmrCore instead of Amr, we have to reimplement these.)
	amrex::ParmParse const pp_amr("amr");

	// Default max number of binary files per multifab when writing plotfiles
	pp_amr.query("plot_nfiles", plot_nfiles);

	// Default max number of binary files per multifab when writing checkpoints
	pp_amr.query("checkpoint_nfiles", checkpoint_nfiles);
}

template <typename problem_t> void AMRSimulation<problem_t>::rereadRuntimeParameters()
{
	// Re-read runtime parameters to ensure they override any compile-time settings
	// This is called at the beginning of evolve() to ensure user input takes precedence
	readParameters();
}

template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditions()
{
	BL_PROFILE("AMRSimulation::setInitialConditions()"); // NOLINT(misc-const-correctness)

	if (restart_chkfile.empty()) {
		// start simulation from the beginning
		const amrex::Real time = 0.0;
		InitFromScratch(time);
		AverageDown();

		if (do_tracers != 0) {
			InitParticles();
		}

		InitPhyParticles();

		if (checkpointInterval_ > 0) {
			WriteCheckpointFile();
		}
	} else {
		// restart from a checkpoint
		ReadCheckpointFile();
	}

	calculateGpotAllLevels();

	// abort if amrex.async_out=1, it is currently broken
	if (amrex::AsyncOut::UseAsyncOut()) {
		amrex::Print() << "[ERROR] [FATAL] AsyncOut is currently broken! If you want to "
				  "run with AsyncOut anyway (THIS MAY CAUSE DATA CORRUPTION), comment "
				  "out this line in src/simulation.hpp. Aborting."
			       << std::endl; // NOLINT(performance-avoid-endl)
		amrex::Abort();
	}

#ifdef AMREX_USE_ASCENT
	if (ascentInterval_ > 0) {
		RenderAscent();
	}
#endif

	if ((plotfileInterval_ > 0 || plotTimeInterval_ > 0) && !skipInitialPlotfile_) {
		WritePlotFile();
	}


	if (particleInterval_ > 0) {
		WriteParticleFile();
	}

	if (statisticsInterval_ > 0) {
		WriteStatisticsFile();
	}

	// initialize diagnostics
	createDiagnostics();
	// output diagnostics
	doDiagnostics();

	// ensure that there are enough boxes per MPI rank
	PerformanceHints();
}

template <typename problem_t> auto AMRSimulation<problem_t>::computeTimestepAtLevel(int lev) -> amrex::ValLocPair<amrex::Real, amrex::IntVect>
{
	// compute CFL timestep on level 'lev'
	BL_PROFILE("AMRSimulation::computeTimestepAtLevel()"); // NOLINT(misc-const-correctness)

	using dtloc_t = amrex::ValLocPair<amrex::Real, amrex::IntVect>;

	// compute hydro timestep on level 'lev'
	computeMaxSignalLocal(lev);
	const amrex::Real domain_signal_max = max_signal_speed_[lev].norminf();
	const amrex::IntVect domain_signal_maxloc = max_signal_speed_[lev].maxIndex(0);
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = geom[lev].CellSizeArray();
	const amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	dtloc_t hydro_dt{.value = cflNumber_ * (dx_min / domain_signal_max), .index = domain_signal_maxloc};

	if (verbose) {
		amrex::Print() << fmt::format("...[level {}] estimated hydro timestep: {:e}\n", lev, hydro_dt.value);
		amrex::Print() << fmt::format("...[level {}] \thydro timestep limited at cell {} with signal speed = {:e}\n", lev, hydro_dt.index,
					      domain_signal_max);
		printCellProperties(lev, hydro_dt.index);
	}

	// compute maximum particle speed on level 'lev'
	amrex::ValLocPair<amrex::Real, amrex::IntVect> particle_dt{.value = std::numeric_limits<amrex::Real>::max(),
								   .index = amrex::IntVect{AMREX_D_DECL(-1, -1, -1)}};
#if AMREX_SPACEDIM == 3
	if (particleRegister_.HasMassiveParticles()) {
		const amrex::ValLocPair<amrex::Real, amrex::RealVect> max_particle_speed = particleRegister_.computeMaxParticleSpeed(lev);
		AMREX_ALWAYS_ASSERT(!std::isnan(max_particle_speed.value));
		AMREX_ALWAYS_ASSERT(std::isfinite(max_particle_speed.value));
		// avoid division by zero by only computing dt if max_particle_speed is not too small
		if (max_particle_speed.value > 1e-5 * (dx_min / hydro_dt.value)) {
			particle_dt.value = particleCflNumber_ * (dx_min / max_particle_speed.value);
			// compute IntVect from RealVect and geom[lev]
			amrex::IntVect cell_idx;
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxinv = geom[lev].InvCellSizeArray();
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo = geom[lev].ProbLoArray();
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				cell_idx[i] = static_cast<int>(dxinv[i] * (max_particle_speed.index[i] - prob_lo[i]));
			}
			particle_dt.index = cell_idx;
		}
		if (verbose) {
			amrex::Print() << fmt::format("...[level {}] estimated particle timestep: {:e}\n", lev, particle_dt.value);
			amrex::Print() << fmt::format("...[level {}] \tmax particle velocity: {:e}\n", lev, max_particle_speed.value);
			amrex::Print() << fmt::format("...[level {}] \tparticle timestep limited at position {::e}\n", lev, max_particle_speed.index);
		}
	}
#endif

	// compute minimum timestep
	std::vector<dtloc_t *> dts = {&hydro_dt, &particle_dt};
	auto *const dt_min_ptr = *std::min_element(dts.begin(), dts.end(), [](dtloc_t *const p1, dtloc_t *const p2) { return p1->value < p2->value; });

	if (verbose) {
		// print the physics that limits the timestep
		if (dt_min_ptr == &hydro_dt) {
			amrex::Print() << fmt::format("...[level {}] timestep limited by HYDRO\n", lev);
		} else if (dt_min_ptr == &particle_dt) {
			amrex::Print() << fmt::format("...[level {}] timestep limited by PARTICLES\n", lev);
		}
	}

	return *dt_min_ptr;
}

template <typename problem_t> void AMRSimulation<problem_t>::computeTimestep()
{
	BL_PROFILE("AMRSimulation::computeTimestep()"); // NOLINT(misc-const-correctness)

	// compute candidate timestep dt_tmp on each level
	amrex::Vector<amrex::Real> dt_tmp(finest_level + 1);
	for (int level = 0; level <= finest_level; ++level) {
		dt_tmp[level] = computeTimestepAtLevel(level).value;
	}

	// limit change in timestep on each level
	constexpr amrex::Real change_max = 1.1;

	for (int level = 0; level <= finest_level; ++level) {
		dt_tmp[level] = std::min(dt_tmp[level], change_max * dt_[level]);
	}

	// set default subcycling pattern
	if (do_subcycle == 1) {
		for (int lev = 1; lev <= max_level; ++lev) {
			nsubsteps[lev] = MaxRefRatio(lev - 1);
		}
	}

	// compute root level timestep given nsubsteps
	amrex::Real dt_0 = dt_tmp[0];
	int level_that_sets_dt_0 = 0; // keep track of which level sets the min dt
	amrex::Long n_factor = 1;

	for (int level = 0; level <= finest_level; ++level) {
		n_factor *= nsubsteps[level];
		const amrex::Real dt_0_old = dt_0; // save old dt_0
		dt_0 = std::min(dt_0, static_cast<amrex::Real>(n_factor) * dt_tmp[level]);
		if (dt_0 < dt_0_old) {
			// level 'level' has now set the timestep
			level_that_sets_dt_0 = level;
		}

		dt_0 = std::min(dt_0, maxDt_); // limit to maxDt_

		if (tNew_[level] == 0.0) { // first timestep
			dt_0 = std::min(dt_0, initDt_);
		}
		if (constantDt_ > 0.0) { // use constant timestep if set
			dt_0 = constantDt_;
		}
	}

	if (verbose) {
		amrex::Print() << "...coarse timestep set by level " << level_that_sets_dt_0 << "\n";
	}

	// Limit dt to avoid overshooting stop_time
	const amrex::Real eps = 1.e-3 * dt_0;

	if (tNew_[0] + dt_0 > stopTime_ - eps) {
		dt_0 = stopTime_ - tNew_[0];
	}

	// assign timesteps on each level
	dt_[0] = dt_0;

	for (int level = 1; level <= finest_level; ++level) {
		dt_[level] = dt_[level - 1] / nsubsteps[level];
	}
}

template <typename problem_t> auto AMRSimulation<problem_t>::getWalltime() -> amrex::Real
{
	const static amrex::Real start_time = amrex::ParallelDescriptor::second(); // initialized on first call
	const amrex::Real time = amrex::ParallelDescriptor::second();
	return time - start_time;
}

template <typename problem_t> auto AMRSimulation<problem_t>::getCycleWalltime() -> amrex::Real
{
	static amrex::Real start_time = amrex::ParallelDescriptor::second(); // initialized on first call
	const amrex::Real current_time = amrex::ParallelDescriptor::second();
	const amrex::Real elapsed_time = current_time - start_time;
	start_time = current_time;
	return elapsed_time;
}

template <typename problem_t> void AMRSimulation<problem_t>::evolve()
{
	BL_PROFILE("AMRSimulation::evolve()"); // NOLINT(misc-const-correctness)

	AMREX_ALWAYS_ASSERT(areInitialConditionsDefined_);

	// Re-read runtime parameters to ensure they override any compile-time settings
	// set in problem_main(). This ensures user input always takes precedence.
	rereadRuntimeParameters();

	amrex::Real cur_time = tNew_[0];
#ifdef AMREX_USE_ASCENT
	int last_ascent_step = 0;
#endif
	int last_projection_step = 0;
	int last_particle_step = 0;
	int last_statistics_step = 0;
	int last_plot_file_step = 0;
	int last_chk_file_step = 0;

	double next_plot_file_time = 0;
	if (plotTimeInterval_ > 0) {
		// We have one plotfile at the start of the simulation, so we set next_plot_file_time to plotTimeInterval_
		next_plot_file_time = plotTimeInterval_;
		while (next_plot_file_time < cur_time) {
			// advance next_plot_file_time until it is >= cur_time
			// this is needed for restarts
			next_plot_file_time += plotTimeInterval_;
		}
	}
	double next_chk_file_time = 0;
	if (checkpointTimeInterval_ > 0) {
		while (next_chk_file_time < cur_time) {
			// advance next_chk_file_time until it is >= cur_time
			// this is needed for restarts
			next_chk_file_time += checkpointTimeInterval_;
		}
	}

	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Vector<amrex::Real> init_sum_cons(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		const int lev = 0;
		init_sum_cons[n] = state_new_cc_[lev].sum(n) * vol;
	}

	getWalltime(); // initialize start_time

	// Main time loop
	int step = istep[0];
	for (; step < maxTimesteps_ && cur_time < stopTime_; ++step) {

		if (suppress_output == 0) {
			amrex::Print() << "\nCoarse STEP " << step + 1 << " at t = " << cur_time << " (" << (cur_time / stopTime_) * 100. << "%) starts ";
		}

		amrex::ParallelDescriptor::Barrier(); // synchronize all MPI ranks

		// output per-cycle timing
		if (printCycleTiming_ == 1) {
			amrex::Real elapsed_sec = getCycleWalltime();
			amrex::Print() << "(cycle time: " << elapsed_sec << " s) ...\n";
		} else {
			amrex::Print() << "...\n";
		}

		computeTimestep();

		// Check for timestep drop (safety feature)
		if (dtCutoff_ > 0.0) {
			const amrex::Real dt_min_allowed = dtCutoff_ * cur_time;
			if ((cur_time > 0.0) && (dt_[0] < dt_min_allowed)) {
				amrex::Print() << "ERROR: Timestep has dropped below cutoff threshold!\n";
				amrex::Print() << "Current time: " << cur_time << "\n";
				amrex::Print() << "Current timestep: " << dt_[0] << "\n";
				amrex::Print() << "Minimum allowed timestep: " << dt_min_allowed << "\n";
				amrex::Print() << "dt_cutoff parameter: " << dtCutoff_ << "\n";
				amrex::Print() << "This may indicate a stability issue in the simulation.\n";
				amrex::Abort("Timestep drop detected - aborting simulation for safety");
			}
		}

		// do user-specified calculations before the level update
		computeBeforeTimestep();

#if AMREX_SPACEDIM == 3
		if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
			// do particle leapfrog (first kick at time t)
			if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
				kickParticlesAllLevels(dt_[0]);
			}
		}
#endif

		// hyperbolic advance over all levels
		// (N.B. when AMR is enabled, regridding may happen during this function!)
		// Particle redistribution is done here.
		int lev = 0;		 // coarsest level
		const int iteration = 1; // this is the first call to advance level 'lev'
		timeStepWithSubcycling(lev, cur_time, iteration);

#if AMREX_SPACEDIM == 3
		if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
			if (particleRegister_.HasMassiveParticles()) {
				// drift particles from t to (t + dt)
				// N.B.: MUST be done *before* Poisson solve at new time!
				particleRegister_.driftParticlesAllLevels(dt_[0], finest_level);
			}
		}
#endif

		// elliptic solve over entire AMR grid (post-timestep)
		ellipticSolveAllLevels(dt_[0]);

		// do particle leapfrog (second kick at t + dt)
#if AMREX_SPACEDIM == 3
		if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
			if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
				kickParticlesAllLevels(dt_[0]);
			}

			// Stellar evolution and SN deposition; only apply to star particles
			// Update particle properties (e.g., luminosity) before particle-mesh interaction
			particleRegister_.updateParticleProperties(cur_time);

			// TODO(cch): Need to take care of AMR subcycling
			particleMeshInteraction(cur_time, dt_[0]);

			// Use the new type-aware particle destruction method
			// TODO(cch): Need to take care of AMR subcycling
			particleRegister_.destroyParticles(0, cur_time, dt_[0]);
		}
#endif

		cur_time += dt_[0];
		++cycleCount_;
		computeAfterTimestep();

		// sync up time (to avoid roundoff error)
		for (lev = 0; lev <= finest_level; ++lev) {
			AMREX_ALWAYS_ASSERT(std::abs((tNew_[lev] - cur_time) / cur_time) < 1e-10);
			tNew_[lev] = cur_time;
		}

#ifdef AMREX_USE_ASCENT
		if (ascentInterval_ > 0 && (step + 1) % ascentInterval_ == 0) {
			last_ascent_step = step + 1;
			RenderAscent();
		}
#endif

		if (statisticsInterval_ > 0 && (step + 1) % statisticsInterval_ == 0) {
			last_statistics_step = step + 1;
			WriteStatisticsFile();
		}

		if (plotfileInterval_ > 0 && (step + 1) % plotfileInterval_ == 0) {
			last_plot_file_step = step + 1;
			WritePlotFile();
		}


		if (particleInterval_ > 0 && (step + 1) % particleInterval_ == 0) {
			last_particle_step = step + 1;
			WriteParticleFile();
		}

		// print particle statistics
		if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
			if (quokka::particle_verbose > 0) {
				particleRegister_.printParticleStatistics();
			}
		}

		// write diagnostics
		doDiagnostics();

		// Writing Plot files at time intervals
		if (last_plot_file_step != step + 1 && plotTimeInterval_ > 0 && next_plot_file_time <= cur_time) {
			next_plot_file_time += plotTimeInterval_;
			WritePlotFile();
		}

		// IMPORTANT: this MUST be written *after* the plotfile to avoid corruption:
		// 	https://github.com/quokka-astro/quokka/issues/554
		if (checkpointTimeInterval_ > 0 && next_chk_file_time <= cur_time) {
			next_chk_file_time += checkpointTimeInterval_;
			WriteCheckpointFile();
		}

		// IMPORTANT: this MUST be written *after* the plotfile to avoid corruption:
		// 	https://github.com/quokka-astro/quokka/issues/554
		if (checkpointInterval_ > 0 && (step + 1) % checkpointInterval_ == 0) {
			last_chk_file_step = step + 1;
			WriteCheckpointFile();
		}

		if (cur_time >= stopTime_ - 1.e-6 * dt_[0]) {
			// we have reached stopTime_
			break;
		}

		if (maxWalltime_ > 0 && getWalltime() > std::max(0.9 * maxWalltime_, static_cast<double>(maxWalltime_ - 300))) {
			// we have exceeded the walltime limit
			break;
		}
	}

	if (step == 0) {
		amrex::Print() << "No cell updates performed!\n";
#ifdef AMREX_USE_ASCENT
		// close Ascent
		ascent_.close();
#endif
		return;
	}

	amrex::Real elapsed_sec = getWalltime();

	// compute reference solution (if it's a test problem)
	computeAfterEvolve(init_sum_cons);

	// compute conservation error
	for (int n = 0; n < ncomp_cc; ++n) {
		amrex::Real const final_sum = state_new_cc_[0].sum(n) * vol;
		amrex::Real const abs_err = (final_sum - init_sum_cons[n]);
		amrex::Print() << "Initial " << componentNames_cc_[n] << " = " << init_sum_cons[n] << '\n';
		amrex::Print() << "\tabsolute conservation error = " << abs_err << '\n';
		if (init_sum_cons[n] != 0.0) {
			amrex::Real const rel_err = abs_err / init_sum_cons[n];
			amrex::Print() << "\trelative conservation error = " << rel_err << '\n';
		}
		amrex::Print() << '\n';
	}

	// compute zone-cycles/sec
	const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
	amrex::ParallelDescriptor::ReduceRealMax(elapsed_sec, IOProc);
	const double microseconds_per_update = 1.0e6 * elapsed_sec / cellUpdates_;
	const double megaupdates_per_second = 1.0 / microseconds_per_update;
	amrex::Print() << "Performance figure-of-merit: " << microseconds_per_update << " μs/zone-update [" << megaupdates_per_second << " Mupdates/s]\n";
	for (int lev = 0; lev <= max_level; ++lev) {
		amrex::Print() << "Zone-updates on level " << lev << ": " << cellUpdatesEachLevel_[lev] << "\n";
	}
	amrex::Print() << '\n';

	// write final plotfile
	if ((plotfileInterval_ > 0 || plotTimeInterval_ > 0) && istep[0] > last_plot_file_step) {
		WritePlotFile();
	}


	// write final particle file
	if (particleInterval_ > 0 && istep[0] > last_particle_step) {
		WriteParticleFile();
	}

	// write final statistics
	if (statisticsInterval_ > 0 && istep[0] > last_statistics_step) {
		WriteStatisticsFile();
	}

	// write final checkpoint
	// IMPORTANT: this MUST be written *after* the plotfile to avoid corruption:
	// 	https://github.com/quokka-astro/quokka/issues/554
	if (checkpointInterval_ > 0 && istep[0] > last_chk_file_step) {
		WriteCheckpointFile();
	}

#ifdef AMREX_USE_ASCENT
	// close Ascent
	ascent_.close();
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::calculateGpotAllLevels()
{
#if AMREX_SPACEDIM == 3
	if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
		if (do_subcycle == 1) { // not supported
			amrex::Abort("Poisson solve is not support when AMR subcycling is enabled! You must set do_subcycle = 0.");
		}

		BL_PROFILE_REGION("GravitySolver"); // NOLINT(misc-const-correctness)

		phi.resize(finest_level + 1);
		amrex::Vector<amrex::MultiFab> rhs(finest_level + 1);
		constexpr int nghost_phi = 1;
		constexpr int nghost_deposit = 1; // CIC deposition requires 1 ghost cell
		constexpr int nghost_drift = 1;	  // particle can drift up to 1 cell
		constexpr int nghost_rhs = nghost_deposit + nghost_drift;
		constexpr int ncomp = 1;
		amrex::Real rhs_min = std::numeric_limits<amrex::Real>::max();

		for (int lev = 0; lev <= finest_level; ++lev) {
			phi[lev].define(grids[lev], dmap[lev], ncomp, nghost_phi);
			rhs[lev].define(grids[lev], dmap[lev], ncomp, nghost_rhs);
			phi[lev].setVal(0); // set initial guess to zero
			rhs[lev].setVal(0);
		}

		for (int lev = 0; lev <= finest_level; ++lev) {
			fillPoissonRhsAtLevel(rhs[lev], lev);
			AMREX_ALWAYS_ASSERT(!rhs[lev].contains_nan());
			rhs_min = std::min(rhs_min, rhs[lev].min(0));
		}

		// deposit particle mass from all particles that have mass into rhs by accumulation
		if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
			if (particleRegister_.HasMassiveParticles()) {
				// create temporary buffer for mass deposition
				// The buffer needs an extra component for the count (last component)
				amrex::Vector<amrex::MultiFab> rhs_buffer(finest_level + 1);
				for (int lev = 0; lev <= finest_level; ++lev) {
					rhs_buffer[lev].define(grids[lev], dmap[lev], ncomp + 1, nghost_rhs);
					rhs_buffer[lev].setVal(0);
				}

				// deposit mass into temporary buffer
				particleRegister_.depositMass(amrex::GetVecOfPtrs(rhs_buffer), finest_level, Gconst_);

				// apply roundoff to buffer before adding to rhs
				for (int lev = 0; lev <= finest_level; ++lev) {
					quokka::ParticleUtils::roundoffMultiFab(rhs_buffer[lev]);
				}

				// add buffer to rhs
				for (int lev = 0; lev <= finest_level; ++lev) {
					amrex::MultiFab::Add(rhs[lev], rhs_buffer[lev], 0, 0, ncomp, 0);
				}
			}
		}

		// check for NaN
		for (int lev = 0; lev <= finest_level; ++lev) {
			AMREX_ALWAYS_ASSERT(!rhs[lev].contains_nan());
		}

		// Analyze boundary conditions for each dimension
		amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_lo;
		amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_hi;
		int num_periodic_dims = 0;
		std::string bc_description = "Gravity BCs: ";

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			std::string dim_name;
			if (idim == 0) {
				dim_name = "x";
			} else if (idim == 1) {
				dim_name = "y";
			} else {
				dim_name = "z";
			}

			if (geom[0].isPeriodic(idim)) {
				bc_lo[idim] = amrex::LinOpBCType::Periodic;
				bc_hi[idim] = amrex::LinOpBCType::Periodic;
				num_periodic_dims++;
				bc_description += dim_name + ":periodic ";
			} else {
				// Use homogeneous Dirichlet (phi = 0) for non-periodic dimensions
				bc_lo[idim] = amrex::LinOpBCType::Dirichlet;
				bc_hi[idim] = amrex::LinOpBCType::Dirichlet;
				bc_description += dim_name + ":Dirichlet ";
			}
		}

		if (verbose) {
			amrex::Print() << bc_description << "\n";
		}

		// Determine solver type: use MLMG if any dimension is periodic, otherwise OpenBCSolver
		const bool use_mlmg_solver = (num_periodic_dims > 0);

		if (use_mlmg_solver) {
			// Use MLMG solver with mixed/periodic boundary conditions
			if (verbose) {
				if (num_periodic_dims == 3) {
					amrex::Print() << "Using MLMG solver with fully periodic boundaries...\n\n";
				} else if (num_periodic_dims == 2) {
					amrex::Print() << "Using MLMG solver with mixed periodic/Dirichlet boundaries...\n\n";
				}
			}

			// Create MLPoisson linear operator with proper LPInfo for AMR
			amrex::LPInfo info;
			// For AMR problems, we need to ensure proper coarsening
			if (finest_level > 0) {
				info.setAgglomeration(false); // Disable agglomeration for AMR
				info.setConsolidation(false); // Disable consolidation for AMR
			}

			amrex::MLPoisson mlpoisson(Geom(0, finest_level), boxArray(0, finest_level), DistributionMap(0, finest_level), info);

			// Set the mixed boundary conditions (already computed above)
			mlpoisson.setDomainBC(bc_lo, bc_hi);

			// Set level boundary conditions for each AMR level
			for (int lev = 0; lev <= finest_level; ++lev) {
				// For gravity problems with mixed BCs, we don't need to specify Dirichlet values
				// MLMG will automatically handle the gauge freedom and find a solution
				// The gravitational force F = -∇φ is gauge invariant (independent of φ + constant)
				mlpoisson.setLevelBC(lev, nullptr);
			}

			// Create MLMG solver
			amrex::MLMG mlmg(mlpoisson);
			if (verbose) {
				mlmg.setVerbose(1);
				mlmg.setBottomVerbose(0);
			}

			// Set solver parameters optimized for gravity problems
			// mlmg.setMaxIter(200);
			// mlmg.setBottomMaxIter(200);

			// MLMG automatically handles solvability constraints for singular problems
			// (periodic boundaries with no Dirichlet conditions). The linear operator
			// automatically detects singular problems and applies the necessary corrections
			// through the getSolvabilityOffset() and fixSolvabilityByOffset() methods.
			// No manual RHS mean subtraction is needed.

			// Solve the system
			amrex::Real abstol = abstolPoisson_ * std::abs(rhs_min);
			amrex::Real final_resnorm = mlmg.solve(amrex::GetVecOfPtrs(phi), amrex::GetVecOfConstPtrs(rhs), reltolPoisson_, abstol);

			// Check convergence
			if (verbose) {
				amrex::Print() << "MLMG converged with final residual norm: " << final_resnorm << "\n";
			}
		} else {
			// Use OpenBC solver for open boundary conditions
			if (verbose) {
				amrex::Print() << "Doing Poisson solve with open boundaries using OpenBCSolver...\n\n";
			}

			amrex::OpenBCSolver poissonSolver(Geom(0, finest_level), boxArray(0, finest_level), DistributionMap(0, finest_level));
			if (verbose) {
				poissonSolver.setVerbose(1);
				poissonSolver.setBottomVerbose(0);
			}

			amrex::Real abstol = abstolPoisson_ * rhs_min;
			poissonSolver.solve(amrex::GetVecOfPtrs(phi), amrex::GetVecOfConstPtrs(rhs), reltolPoisson_, abstol);
		}

		if (verbose) {
			amrex::Print() << "\n";
		}

		// check for NaN
		for (int lev = 0; lev <= finest_level; ++lev) {
			// NOTE: this fails when multiple levels are fully refined when open boundary condition is used.
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!phi[lev].contains_nan(), fmt::format("NaN detected in phi at level {} after Poisson solve", lev));
		}
	}
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::gravAccelAllLevels(const amrex::Real dt)
{
#if AMREX_SPACEDIM == 3
	if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {

		BL_PROFILE_REGION("GravitySolver"); // NOLINT(misc-const-correctness)

		// add gravitational acceleration to hydro state (using operator splitting)
		for (int lev = 0; lev <= finest_level; ++lev) {
			applyPoissonGravityAtLevel(phi[lev], lev, dt);
		}
	}
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::ellipticSolveAllLevels(const amrex::Real dt)
{
#if AMREX_SPACEDIM == 3
	if constexpr (Physics_Traits<problem_t>::is_self_gravity_enabled) {
		if (poissonSupercycleInterval_ > 1) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(regrid_int <= 0, "Poisson supercycling is only allowed for static meshes!");
		}
		if (istep[0] % poissonSupercycleInterval_ == 0) {
			// do Poisson solve every poissonSupercycleInterval_ coarse steps
			calculateGpotAllLevels();
		}
		// this must be done every step
		gravAccelAllLevels(dt);
	}
#endif
}

struct setFunctorParticleAccel {
	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
					 amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp,
					 const int &orig_comp) const
	{
		amrex::ignore_unused(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
	}
};

#if AMREX_SPACEDIM == 3
template <typename problem_t> void AMRSimulation<problem_t>::kickParticlesAllLevels(const amrex::Real dt)
{
	const BL_PROFILE("AMRSimulation::kickParticlesAllLevels()");
	// kick particles (do: vel[i] += 0.5 * dt * accel[i])

	// skip if there are no massive particles
	if (!particleRegister_.HasMassiveParticles()) {
		return;
	}

	// Compute accelerations and kick particles
	for (int lev = 0; lev <= finest_level; ++lev) {
		// NOTE: CIC interpolation requires 1, but particles may have drifted
		// 	into 1 ghost cell since last particle redistribute.
		const int nghost_acc = 2;
		const int nghost_phi = nghost_acc + 1; // Need extra ghost cell for centered difference

		// Create potential MultiFab with sufficient ghost cells for gradient computation
		amrex::MultiFab phi_extended(boxArray(lev), DistributionMap(lev), 1, nghost_phi);

		// Fill extended potential from existing phi using FillPatch
		// This handles coarse-fine boundaries without InterpFromCoarseLevel
		if (lev == 0) {
			// Base level: just copy and fill boundaries
			amrex::MultiFab::Copy(phi_extended, phi[lev], 0, 0, 1, 0);
			phi_extended.FillBoundary(geom[lev].periodicity());

			// Apply physical boundary conditions to phi
			amrex::Vector<amrex::BCRec> phiBC(1);
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				phiBC[0].setLo(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].lo(i));
				phiBC[0].setHi(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].hi(i));
			}

			amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});
			amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiBdryFunct(geom[lev], phiBC, boundaryFunctor);
			phiBdryFunct(phi_extended, 0, 1, phi_extended.nGrowVect(), 0., 0);
		} else {
			// Fine level: use FillPatchTwoLevels to properly handle coarse-fine boundaries
			amrex::Vector<amrex::BCRec> phiBC(1);
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				phiBC[0].setLo(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].lo(i));
				phiBC[0].setHi(i, BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].hi(i));
			}

			amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});
			amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiBdryFunct(geom[lev], phiBC, boundaryFunctor);
			amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> phiCoarseBdryFunct(geom[lev - 1], phiBC, boundaryFunctor);

			amrex::FillPatchTwoLevels(phi_extended, 0., {&phi[lev - 1]}, {0.}, {&phi[lev]}, {0.}, 0, 0, 1, geom[lev - 1], geom[lev],
						  phiCoarseBdryFunct, 0, phiBdryFunct, 0, refRatio(lev - 1), &amrex::quadratic_interp, phiBC, 0);
		}

		// Create cell-centered acceleration MultiFab
		amrex::MultiFab accel_cc(boxArray(lev), DistributionMap(lev), AMREX_SPACEDIM, nghost_acc);

		// Compute acceleration directly from potential gradient at cell centers
		const auto &phi_arr = phi_extended.const_arrays();
		const auto dx_inv = geom[lev].InvCellSizeArray();
		auto accel_arr = accel_cc.arrays();

		amrex::ParallelFor(accel_cc, amrex::IntVect{nghost_acc}, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			// Compute cell-centered acceleration using central differences of potential
			// accel = -grad(phi)
			accel_arr[bx](i, j, k, 0) = -0.5 * dx_inv[0] * (phi_arr[bx](i + 1, j, k) - phi_arr[bx](i - 1, j, k));
#if AMREX_SPACEDIM >= 2
			accel_arr[bx](i, j, k, 1) = -0.5 * dx_inv[1] * (phi_arr[bx](i, j + 1, k) - phi_arr[bx](i, j - 1, k));
#endif
#if AMREX_SPACEDIM == 3 // NOLINT(readability-redundant-preprocessor)
			accel_arr[bx](i, j, k, 2) = -0.5 * dx_inv[2] * (phi_arr[bx](i, j, k + 1) - phi_arr[bx](i, j, k - 1));
#endif
		});
		amrex::Gpu::streamSynchronize();

		// check for NaN
		AMREX_ALWAYS_ASSERT(!accel_cc.contains_nan());

		// Kick particles using the acceleration field
		particleRegister_.kickParticlesAtLevel(lev, dt, accel_cc);
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::particleMeshInteraction(amrex::Real time, amrex::Real dt)
{
	const BL_PROFILE("AMRSimulation::particleMeshInteraction()");
	// Need 4 ghost cells for SN deposition with a SNR radius of 3 dx.
	const int nghost = 4;

	// Assume all SN progenitors are at the finest level
	const int lev = finest_level;

	// Fill boundary before computing accretion rate. This is necessary because the computation of accretion rate uses 3 ghost cells, but the
	// boundaries are not filled at this point.
	// TODO(cch): fill the hydro variables but not radiation variables, since radiation variables are used in accretion
	state_new_cc_[lev].FillBoundary(geom[lev].periodicity());

	// Create a MultiFab to hold the change of states (density, 3 x momentum, internal energy, energy) during particle-mesh interaction
	// The extra component is for the particle counts in cells
	amrex::MultiFab accretion_rate_at_level(grids[lev], dmap[lev], Physics_NumVars::numHydroVars + 1, nghost);

	accretion_rate_at_level.setVal(0.0);

	// Sink accretion, stage 1: compute the accretion rate
	particleRegister_.computeSinkAccretion(state_new_cc_[lev], accretion_rate_at_level, lev, time, dt);

	// Apply roundoff to accretion_rate_at_level
	// TODO(cch): compute accumulative errors and pass it to roundoffMultiFab
	quokka::ParticleUtils::roundoffMultiFab(accretion_rate_at_level);

	// Sink accretion, stage 2: update the particle states -- compute scale_down, apply to particle, apply to cells
	particleRegister_.applySinkAccretion(state_new_cc_[lev], accretion_rate_at_level, geom[lev], lev, time, dt);

	// We allow particle formation at the finest level only to avoid duplicate particle creation from multiple levels at the same location.
	particleRegister_.createParticlesFromState(state_new_cc_[lev], accretion_rate_at_level, lev, time, dt);

	// Deposit the SN particles into the MultiFab
	const amrex::Real max_velocity = particleRegister_.depositSN(state_new_cc_[lev], lev, time, dt);

	// Check if the maximum velocity is greater than the threshold
	constexpr amrex::Real v_over_c_threshold = 0.03;
	if (max_velocity > v_over_c_threshold * C::c_light) {
		amrex::Print() << "WARNING: SN remnant net velocity (" << max_velocity / C::c_light << " c) greater than " << v_over_c_threshold
			       << " c threshold!" << "\n";
	}
}
#endif // AMREX_SPACEDIM == 3

// N.B.: This function actually works for subcycled or not subcycled, as long as
// nsubsteps[lev] is set correctly.
template <typename problem_t> void AMRSimulation<problem_t>::timeStepWithSubcycling(int lev, amrex::Real time, int iteration)
{
	BL_PROFILE("AMRSimulation::timeStepWithSubcycling()"); // NOLINT(misc-const-correctness)

	// perform regrid if needed
	if (regrid_int > 0) {
		// help keep track of whether a level was already regridded
		// from a coarser level call to regrid
		static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

		// regrid changes level "lev+1" so we don't regrid on max_level
		// also make sure we don't regrid fine levels again if
		// it was taken care of during a coarser regrid
		if (lev < max_level && istep[lev] > last_regrid_step[lev]) {
			if (istep[lev] % regrid_int == 0) {
				// regrid could add newly refined levels (if finest_level < max_level)
				// so we save the previous finest level index
				int old_finest = finest_level;
				regrid(lev, time);

				// mark that we have regridded this level already
				for (int k = lev; k <= finest_level; ++k) {
					last_regrid_step[k] = istep[k];
				}

				// if there are newly created levels, set the time step
				for (int k = old_finest + 1; k <= finest_level; ++k) {
					if (do_subcycle != 0) {
						dt_[k] = dt_[k - 1] / nsubsteps[k];
					} else {
						dt_[k] = dt_[k - 1];
					}
				}

				// redistribute particles
				if (do_tracers != 0) {
					TracerPC->Redistribute(lev);
				}

				// redistribute all particles in particleRegister_
				particleRegister_.redistribute(lev);

				// do fix-up on all levels that have been re-gridded
				for (int k = lev; k <= finest_level; ++k) {
					FixupState(k);
				}
			}
		}
	}

	if (Verbose()) {
		amrex::Print() << "[Level " << lev << " step " << istep[lev] + 1 << "] ";
		amrex::Print() << "ADVANCE with time = " << std::scientific << tNew_[lev] << " dt = " << std::scientific << dt_[lev] << '\n';
	}

	// Advance a single level for a single time step, and update flux registers
	tOld_[lev] = tNew_[lev];
	tNew_[lev] += dt_[lev]; // critical that this is done *before* advanceAtLevel

	// do hyperbolic advance over all levels
	advanceSingleTimestepAtLevel(lev, time, dt_[lev], nsubsteps[lev]);

	++istep[lev];
	cellUpdates_ += CountCells(lev); // keep track of total number of cell updates
	cellUpdatesEachLevel_[lev] += CountCells(lev);

	if (Verbose()) {
		amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
		amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
	}

	// advance finer levels

	if (lev < finest_level) {

		// recursive call for next-finer level
		for (int i = 1; i <= nsubsteps[lev + 1]; ++i) {
			if (lev < finest_level) { // this may change during a regrid!
				timeStepWithSubcycling(lev + 1, time + (i - 1) * dt_[lev + 1], i);
			}
		}

		// do post-timestep operations

		if (do_reflux != 0) {
			if (flux_reg_[lev + 1] != nullptr) {
				flux_reg_[lev + 1]->Reflux(state_new_cc_[lev], 1.0, 0, 0, state_new_cc_[lev].nComp(), geom[lev]);
			}
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				if (emf_reg_[lev + 1] != nullptr) {
					// NOLINTNEXTLINE(readability-container-data-pointer)
					emf_reg_[lev + 1]->Reflux({AMREX_D_DECL(&state_new_fc_[lev][0], &state_new_fc_[lev][1], &state_new_fc_[lev][2])});
				}
			}
		}

		AverageDownTo(lev); // average lev+1 down to lev
		FixupState(lev);    // fix any unphysical states created by reflux or averaging

		fillpatcher_[lev + 1].reset(); // because the data on lev have changed.
	}

	// redistribute tracer particles
	if (do_tracers != 0) {
		int redistribute_ngrow = 0;
		if ((iteration < nsubsteps[lev]) || (lev == 0)) {
			if (lev == 0) {
				redistribute_ngrow = 0;
			} else {
				redistribute_ngrow = iteration;
			}
			TracerPC->Redistribute(lev, TracerPC->finestLevel(), redistribute_ngrow);
		}
	}

	// redistribute all particles in particleRegister_
	int redistribute_ngrow = 0;
	if ((iteration < nsubsteps[lev]) || (lev == 0)) {
		if (lev == 0) {
			redistribute_ngrow = 0;
		} else {
			redistribute_ngrow = iteration;
		}
		// redistribute all particles in particleRegister_
		particleRegister_.redistribute(lev, redistribute_ngrow);
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::incrementFluxRegisters(amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays, int const lev, amrex::Real const dt_lev)
{
	BL_PROFILE("AMRSimulation::incrementFluxRegisters()"); // NOLINT(misc-const-correctness)

	if ((fr_as_crse == nullptr) && (fr_as_fine == nullptr)) {
		return;
	}

	const auto dx = geom[lev].CellSizeArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> face_area{};

	if constexpr (AMREX_SPACEDIM == 1) {
		face_area[0] = 1.0;
	} else if constexpr (AMREX_SPACEDIM == 2) {
		face_area[0] = dx[1];
		face_area[1] = dx[0];
	} else {
		face_area[0] = dx[1] * dx[2];
		face_area[1] = dx[0] * dx[2];
		face_area[2] = dx[0] * dx[1];
	}

	if (fr_as_crse != nullptr) {
		AMREX_ASSERT(lev < finestLevel());
		AMREX_ASSERT(fr_as_crse == flux_reg_[lev + 1].get());
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			int const ncomp = fluxArrays[dir].nComp();
			if (ncomp == 0) {
				continue;
			}
			fr_as_crse->CrseInit(fluxArrays[dir], dir, 0, 0, ncomp, -dt_lev * face_area[dir], amrex::FluxRegister::ADD);
		}
	}

	if (fr_as_fine != nullptr) {
		AMREX_ASSERT(lev > 0);
		AMREX_ASSERT(fr_as_fine == flux_reg_[lev].get());
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			int const ncomp = fluxArrays[dir].nComp();
			if (ncomp == 0) {
				continue;
			}
			fr_as_fine->FineAdd(fluxArrays[dir], dir, 0, 0, ncomp, dt_lev * face_area[dir]);
		}
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::incrementEMFRegisters(amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine,
						     std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_emf_components, int const lev, amrex::Real const dt_lev)
{
	BL_PROFILE("AMRSimulation::incrementEMFRegisters()"); // NOLINT(misc-const-correctness)

#if (AMREX_SPACEDIM == 3)
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		if (emf_as_crse != nullptr) {
			AMREX_ASSERT(lev < finestLevel());
			AMREX_ASSERT(emf_as_crse == emf_reg_[lev + 1].get());
			emf_as_crse->CrseAdd(mfi, {ec_emf_components[0].fabPtr(mfi), ec_emf_components[1].fabPtr(mfi), ec_emf_components[2].fabPtr(mfi)},
					     dt_lev);
		}

		if (emf_as_fine != nullptr) {
			AMREX_ASSERT(lev > 0);
			AMREX_ASSERT(emf_as_fine == emf_reg_[lev].get());
			emf_as_fine->FineAdd(mfi, {ec_emf_components[0].fabPtr(mfi), ec_emf_components[1].fabPtr(mfi), ec_emf_components[2].fabPtr(mfi)},
					     dt_lev);
		}
	}
#else
	amrex::ignore_unused(emf_as_crse, emf_as_fine, ec_emf_components, lev, dt_lev);
#endif
}

template <typename problem_t> auto AMRSimulation<problem_t>::getAmrInterpolaterCellCentered() -> amrex::MFInterpolater *
{
	amrex::MFInterpolater *mapper = nullptr;

	if (amrInterpMethod_ == 0) { // piecewise-constant interpolation
		mapper = &amrex::mf_pc_interp;
	} else if (amrInterpMethod_ == 1) { // slope-limited linear interpolation
		//  It has the following important properties:
		// 1. should NOT produce new extrema
		//    (will revert to piecewise constant if any component has a local min/max)
		// 2. should be conservative
		// 3. preserves linear combinations of variables in each cell
		mapper = &amrex::mf_linear_slope_minmax_interp;
	} else {
		amrex::Abort("Invalid AMR interpolation method specified!");
	}

	return mapper; // global object, so this is ok
}

template <typename problem_t> auto AMRSimulation<problem_t>::getAmrInterpolaterFaceCentered() -> amrex::Interpolater *
{
	// Use linear interpolation for generic face-centered operations (e.g., ghost fills).
	return &amrex::face_linear_interp;
}

// Make a new level using provided BoxArray and DistributionMapping and fill
// with interpolated coarse level data. Overrides the pure virtual function in
// AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromCoarse(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	BL_PROFILE("AMRSimulation::MakeNewLevelFromCoarse()"); // NOLINT(misc-const-correctness)

	// cell-centred
	const int ncomp_cc = state_new_cc_[level - 1].nComp();
	const int nghost_cc = state_new_cc_[level - 1].nGrow();
	state_new_cc_[level].define(ba, dm, ncomp_cc, nghost_cc);
	state_old_cc_[level].define(ba, dm, ncomp_cc, nghost_cc);
	FillCoarsePatch(level, time, state_new_cc_[level], 0, ncomp_cc, BCs_cc_, quokka::centering::cc, quokka::direction::na);
	FillCoarsePatch(level, time, state_old_cc_[level], 0, ncomp_cc, BCs_cc_, quokka::centering::cc, quokka::direction::na); // also necessary

	max_signal_speed_[level].define(ba, dm, 1, nghost_cc);
	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0 && (do_reflux != 0)) {
		flux_reg_[level] = std::make_unique<amrex::FluxRegister>(ba, dm, refRatio(level - 1), level, ncomp_cc);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			const int nemf_vars = 1;
			emf_reg_[level] = std::make_unique<amrex::EdgeFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
										    Geom(level - 1), nemf_vars);
		}
	}

	// face-centred
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		const int ncomp_per_dim_fc = state_new_fc_[level - 1][0].nComp();
		const int nghost_fc = state_new_fc_[level - 1][0].nGrow();
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			state_new_fc_[level][idim] =
			    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			state_old_fc_[level][idim] =
			    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
		}
		amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> new_mf_array;
		amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> old_mf_array;
		amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> BCs_array;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			new_mf_array[idim] = &state_new_fc_[level][idim];
			old_mf_array[idim] = &state_old_fc_[level][idim];
			BCs_array[idim] = BCs_fc_;
		}
		FillCoarsePatchFaceArray(level, time, new_mf_array, 0, ncomp_per_dim_fc, BCs_array);
		FillCoarsePatchFaceArray(level, time, old_mf_array, 0, ncomp_per_dim_fc, BCs_array);
	}
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data. Overrides the pure virtual function
// in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::RemakeLevel(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	BL_PROFILE("AMRSimulation::RemakeLevel()"); // NOLINT(misc-const-correctness)

	// cell-centred
	const int ncomp_cc = state_new_cc_[level].nComp();
	const int nghost_cc = state_new_cc_[level].nGrow();
	amrex::MultiFab int_state_new_cc(ba, dm, ncomp_cc, nghost_cc);
	amrex::MultiFab int_state_old_cc(ba, dm, ncomp_cc, nghost_cc);
	FillPatch(level, time, int_state_new_cc, 0, ncomp_cc, quokka::centering::cc, quokka::direction::na, FillPatchType::fillpatch_function);
	std::swap(int_state_new_cc, state_new_cc_[level]);
	std::swap(int_state_old_cc, state_old_cc_[level]);

	amrex::MultiFab max_signal_speed(ba, dm, 1, nghost_cc);
	std::swap(max_signal_speed, max_signal_speed_[level]);

	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0 && (do_reflux != 0)) {
		flux_reg_[level] = std::make_unique<amrex::FluxRegister>(ba, dm, refRatio(level - 1), level, ncomp_cc);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			const int nemf_vars = 1;
			emf_reg_[level] = std::make_unique<amrex::EdgeFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
										    Geom(level - 1), nemf_vars);
		}
	}

	// face-centred
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		const int ncomp_per_dim_fc = state_new_fc_[level][0].nComp();
		const int nghost_fc = state_new_fc_[level][0].nGrow();
		amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> int_state_new_fc;
		amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> int_state_old_fc;
		amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> BCs_array;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			int_state_new_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			int_state_old_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			BCs_array[idim] = BCs_fc_;
		}
		amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> int_state_new_fc_ptr;
		amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> int_state_old_fc_ptr;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			int_state_new_fc_ptr[idim] = &int_state_new_fc[idim];
			int_state_old_fc_ptr[idim] = &int_state_old_fc[idim];
		}
		FillCoarsePatchFaceArray(level, time, int_state_new_fc_ptr, 0, ncomp_per_dim_fc, BCs_array);
		FillCoarsePatchFaceArray(level, time, int_state_old_fc_ptr, 0, ncomp_per_dim_fc, BCs_array);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			std::swap(int_state_new_fc[idim], state_new_fc_[level][idim]);
			std::swap(int_state_old_fc[idim], state_old_fc_[level][idim]);
		}
	}
}

// Delete level data. Overrides the pure virtual function in AmrCore
template <typename problem_t> void AMRSimulation<problem_t>::ClearLevel(int level)
{
	BL_PROFILE("AMRSimulation::ClearLevel()"); // NOLINT(misc-const-correctness)

	state_new_cc_[level].clear();
	state_old_cc_[level].clear();
	max_signal_speed_[level].clear();

	flux_reg_[level].reset(nullptr);
	emf_reg_[level].reset(nullptr);
	fillpatcher_[level].reset(nullptr);

	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			state_new_fc_[level][idim].clear();
			state_old_fc_[level][idim].clear();
		}
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::InterpHookNone(amrex::MultiFab &mf, int scomp, int ncomp)
{
	// do nothing
}

template <typename problem_t> struct setBoundaryFunctor {
	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
					 amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp,
					 const int &orig_comp) const
	{
		AMRSimulation<problem_t>::setCustomBoundaryConditions(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
	}
};

template <typename problem_t> struct setBoundaryFunctorFaceVar {
	quokka::direction dir_;

	// Constructor to store the direction
	AMREX_GPU_HOST_DEVICE explicit setBoundaryFunctorFaceVar(quokka::direction dir) : dir_(dir) {}

	// Default constructor (if needed for compatibility)
	AMREX_GPU_HOST_DEVICE setBoundaryFunctorFaceVar() : dir_(quokka::direction::na) {}

	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
					 amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp,
					 const int &orig_comp) const
	{
		// Now pass the stored direction to the function
		if (dir_ == quokka::direction::x) {
			AMRSimulation<problem_t>::template setCustomBoundaryConditionsFaceVar<quokka::direction::x>(iv, dest, dcomp, numcomp, geom, time, bcr,
														    bcomp, orig_comp);
		} else if (dir_ == quokka::direction::y) {
			AMRSimulation<problem_t>::template setCustomBoundaryConditionsFaceVar<quokka::direction::y>(iv, dest, dcomp, numcomp, geom, time, bcr,
														    bcomp, orig_comp);
		} else if (dir_ == quokka::direction::z) {
			AMRSimulation<problem_t>::template setCustomBoundaryConditionsFaceVar<quokka::direction::z>(iv, dest, dcomp, numcomp, geom, time, bcr,
														    bcomp, orig_comp);
		}
	}
};

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest,
											       int dcomp, int numcomp, amrex::GeometryData const &geom,
											       const amrex::Real time, const amrex::BCRec *bcr, int bcomp,
											       int orig_comp)
{
	// user should implement if needed using template specialization
	// (This is only called when amrex::BCType::ext_dir is set for a given
	// boundary.)

	// set boundary condition for cell 'iv'
}

template <typename problem_t>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<problem_t>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
							     amrex::GeometryData const &geom, const amrex::Real time, const amrex::BCRec *bcr, int bcomp,
							     int orig_comp)
{
	// user should implement if needed using template specialization
	// (This is only called when amrex::BCType::ext_dir is set for a given
	// boundary.)

	// set boundary condition for cell 'iv'
}

// Compute a new multifab 'mf' by copying in state from valid region and filling
// ghost cells
// NOTE: This implementation is only used by AdvectionSimulation.
//  QuokkaSimulation provides its own implementation.
template <typename problem_t>
void AMRSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
					 FillPatchType fptype)
{
	BL_PROFILE("AMRSimulation::FillPatch()"); // NOLINT(misc-const-correctness)

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
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_cc_, cen, dir, fptype, InterpHookNone, InterpHookNone);
	} else if (cen == quokka::centering::fc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_fc_, cen, dir, fptype, InterpHookNone, InterpHookNone);
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditionsAtLevel_cc(int level, amrex::Real time)
{
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	const int nghost_cc = nghost_cc_;
	// iterate over the domain
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		quokka::grid grid_elem(state_new_cc_[level].array(iter), iter.validbox(), geom[level].CellSizeArray(), geom[level].ProbLoArray(),
				       geom[level].ProbHiArray(), quokka::centering::cc, quokka::direction::na);
		// set initial conditions defined by the user
		setInitialConditionsOnGrid(grid_elem);
	}
	// check that the valid state_new_cc_[level] is properly filled
	AMREX_ALWAYS_ASSERT(!state_new_cc_[level].contains_nan(0, ncomp_cc));
	// fill ghost zones
	fillBoundaryConditions(state_new_cc_[level], state_new_cc_[level], level, time, quokka::centering::cc, quokka::direction::na, InterpHookNone,
			       InterpHookNone, FillPatchType::fillpatch_function);
	// copy to state_old_cc_ (including ghost zones)
	state_old_cc_[level].ParallelCopy(state_new_cc_[level], 0, 0, ncomp_cc, nghost_cc, nghost_cc);
}

template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditionsAtLevel_fc(int level, amrex::Real time)
{
	const int ncomp_per_dim_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
	const int nghost_fc = nghost_fc_;
	// for each face-centering
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		// initialize to zero
		state_new_fc_[level][idim].setVal(0.);
		// iterate over the domain and re-initialise data
		for (amrex::MFIter iter(state_new_fc_[level][idim]); iter.isValid(); ++iter) {
			quokka::grid grid_elem(state_new_fc_[level][idim].array(iter), iter.validbox(), geom[level].CellSizeArray(), geom[level].ProbLoArray(),
					       geom[level].ProbHiArray(), quokka::centering::fc, static_cast<quokka::direction>(idim));
			// set initial conditions defined by the user
			setInitialConditionsOnGridFaceVars(grid_elem);
		}
		// check that the valid state_new_fc_[level][idim] data is filled properly
		AMREX_ALWAYS_ASSERT(!state_new_fc_[level][idim].contains_nan(0, ncomp_per_dim_fc));
		// fill ghost zones
		// N.B. for face-centered fields, we must use FillPatchType::fillpatch_function
		fillBoundaryConditions(state_new_fc_[level][idim], state_new_fc_[level][idim], level, time, quokka::centering::fc,
				       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
		state_old_fc_[level][idim].ParallelCopy(state_new_fc_[level][idim], 0, 0, ncomp_per_dim_fc, nghost_fc, nghost_fc);
	}
}

// Make a new level from scratch using provided BoxArray and
// DistributionMapping. Only used during initialization. Overrides the pure
// virtual function in AmrCore
template <typename problem_t>
void AMRSimulation<problem_t>::MakeNewLevelFromScratch(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	BL_PROFILE("AMRSimulation::MakeNewLevelFromScratch()"); // NOLINT(misc-const-correctness)

	// define empty MultiFab containers with the right number of components and ghost-zones

	// cell-centred
	const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
	const int nghost_cc = nghost_cc_;
	state_new_cc_[level].define(ba, dm, ncomp_cc, nghost_cc);
	state_old_cc_[level].define(ba, dm, ncomp_cc, nghost_cc);
	max_signal_speed_[level].define(ba, dm, 1, nghost_cc);

	tNew_[level] = time;
	tOld_[level] = time - 1.e200;

	if (level > 0 && (do_reflux != 0)) {
		flux_reg_[level] = std::make_unique<amrex::FluxRegister>(ba, dm, refRatio(level - 1), level, ncomp_cc);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			const int nemf_vars = 1;
			emf_reg_[level] = std::make_unique<amrex::EdgeFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
										    Geom(level - 1), nemf_vars);
		}
	}

	// face-centred
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		const int ncomp_per_dim_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
		const int nghost_fc = nghost_fc_;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			state_new_fc_[level][idim] =
			    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			state_old_fc_[level][idim] =
			    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
		}
	}

	// precalculate any required data (e.g., data table; as implemented by the
	// user) before initialising state variables
	preCalculateInitialConditions();

	// initial state variables
	setInitialConditionsAtLevel_cc(level, time);
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		setInitialConditionsAtLevel_fc(level, time);
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

template <typename problem_t>
template <typename PreInterpHook, typename PostInterpHook>
void AMRSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state, int const lev, amrex::Real const time,
						      quokka::centering cen, quokka::direction dir, PreInterpHook const &pre_interp,
						      PostInterpHook const &post_interp, FillPatchType fptype)
{
	BL_PROFILE("AMRSimulation::fillBoundaryConditions()"); // NOLINT(misc-const-correctness)

	// On a single level, any periodic boundaries are filled first
	// 	then built-in boundary conditions are filled (with amrex::FilccCell()),
	//	then user-defined Dirichlet boundary conditions are filled.
	// (N.B.: The user-defined boundary function is called for *all* ghost cells.)

	// [NOTE: If user-defined and periodic boundaries are both used
	//  (for different coordinate dimensions), the edge/corner cells *will* be
	//  filled by amrex::FilccCell(). Remember to fill *all* variables in the
	//  MultiFab, e.g., both hydro and radiation).

	if ((cen != quokka::centering::cc) && (cen != quokka::centering::fc)) {
		amrex::Print() << "Centering passed to fillBoundaryConditions(): " << static_cast<int>(cen) << "\n";
		throw std::runtime_error("Only cell-centred (cc) and face-centred (fc) variables are supported, thus far.");
	}

	amrex::Vector<amrex::BCRec> BCs;
	if (cen == quokka::centering::cc) {
		BCs = BCs_cc_;
	} else if (cen == quokka::centering::fc) {
		BCs = BCs_fc_;
	}

	if (lev > 0) { // refined level
		amrex::Vector<amrex::MultiFab *> fineData{&state};
		amrex::Vector<amrex::Real> fineTime = {time};
		amrex::Vector<amrex::MultiFab *> coarseData;
		amrex::Vector<amrex::Real> coarseTime;

		// returns old state, new state, or both depending on 'time'
		GetData(lev - 1, time, coarseData, coarseTime, cen, dir);
		AMREX_ASSERT(!state.contains_nan(0, state.nComp()));

		for (auto &i : coarseData) {
			AMREX_ASSERT(!i->contains_nan(0, state.nComp()));
			AMREX_ASSERT(!i->contains_nan()); // check ghost zones
			amrex::ignore_unused(i);
		}

		FillPatchWithData(lev, time, S_filled, coarseData, coarseTime, fineData, fineTime, 0, S_filled.nComp(), BCs, cen, dir, fptype, pre_interp,
				  post_interp);
	} else { // level 0
		// fill internal and periodic boundaries, ignoring corners (cross=true)
		// (there is no performance benefit for this in practice)
		// state.FillBoundary(geom[lev].periodicity(), true);
		state.FillBoundary(geom[lev].periodicity());

		if (!geom[lev].isAllPeriodic()) {
			if (cen == quokka::centering::cc) {
				// create cell-centered boundary functor
				amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor =
				    amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>{setBoundaryFunctor<problem_t>{}};
				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> physicalBoundaryFunctor(geom[lev], BCs,
																  boundaryFunctor);
				// fill physical boundaries
				physicalBoundaryFunctor(state, 0, state.nComp(), state.nGrowVect(), time, 0);
			} else if (cen == quokka::centering::fc) {
				// create face-centered boundary functor using the passed direction
				amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor =
				    amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>{setBoundaryFunctorFaceVar<problem_t>{dir}};
				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> physicalBoundaryFunctor(geom[lev], BCs,
																	 boundaryFunctor);
				// fill physical boundaries
				physicalBoundaryFunctor(state, 0, state.nComp(), state.nGrowVect(), time, 0);
			}
		}
	}

	// ensure that there are no NaNs (can happen when domain boundary filling is
	// unimplemented or malfunctioning)
	AMREX_ASSERT(!S_filled.contains_nan(0, S_filled.nComp()));
	AMREX_ASSERT(!S_filled.contains_nan()); // check ghost zones (usually this is caused by
						// forgetting to fill some components when
						// using custom Dirichlet BCs, e.g., radiation
						// variables in a hydro-only problem)
}

// Compute a new multifab 'mf' by copying in state from given data and filling
// ghost cells
template <typename problem_t>
template <typename PreInterpHook, typename PostInterpHook>
void AMRSimulation<problem_t>::FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf, amrex::Vector<amrex::MultiFab *> &coarseData,
						 amrex::Vector<amrex::Real> &coarseTime, amrex::Vector<amrex::MultiFab *> &fineData,
						 amrex::Vector<amrex::Real> &fineTime, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs,
						 quokka::centering &cen, quokka::direction dir, FillPatchType fptype, PreInterpHook const &pre_interp,
						 PostInterpHook const &post_interp)
{
	BL_PROFILE("AMRSimulation::FillPatchWithData()"); // NOLINT(misc-const-correctness)

	amrex::MFInterpolater *mapper_cc = getAmrInterpolaterCellCentered();

	if (fptype == FillPatchType::fillpatch_class) {
		if (fillpatcher_[lev] == nullptr) {
			fillpatcher_[lev] = std::make_unique<amrex::FillPatcher<amrex::MultiFab>>(
			    grids[lev], dmap[lev], geom[lev], grids[lev - 1], dmap[lev - 1], geom[lev - 1], mf.nGrowVect(), mf.nComp(), mapper_cc);
		}
	}

	// create functor to fill ghost zones at domain boundaries
	// (note that domain boundaries may be present at any refinement level)
	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor_cc(setBoundaryFunctor<problem_t>{});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor_cc(geom[lev], BCs, boundaryFunctor_cc);

	amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor_fc(setBoundaryFunctorFaceVar<problem_t>{dir});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> finePhysicalBoundaryFunctor_fc(geom[lev], BCs, boundaryFunctor_fc);

	if (lev == 0) { // NOTE: used by RemakeLevel
		// copies interior zones, fills ghost zones
		if (cen == quokka::centering::cc) {
			amrex::FillPatchSingleLevel(mf, time, fineData, fineTime, 0, icomp, ncomp, geom[lev], finePhysicalBoundaryFunctor_cc, 0);
		} else if (cen == quokka::centering::fc) {
			amrex::FillPatchSingleLevel(mf, time, fineData, fineTime, 0, icomp, ncomp, geom[lev], finePhysicalBoundaryFunctor_fc, 0);
		}
	} else {
		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor_cc(geom[lev - 1], BCs,
															   boundaryFunctor_cc);
		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> coarsePhysicalBoundaryFunctor_fc(geom[lev - 1], BCs,
																  boundaryFunctor_fc);

		// copies interior zones, fills ghost zones with space-time interpolated
		// data
		if (fptype == FillPatchType::fillpatch_class) {
			if (cen == quokka::centering::cc) {
				// N.B.: this only works for cell-centered data
				fillpatcher_[lev]->fill(mf, mf.nGrowVect(), time, coarseData, coarseTime, fineData, fineTime, 0, icomp, ncomp,
							coarsePhysicalBoundaryFunctor_cc, 0, finePhysicalBoundaryFunctor_cc, 0, BCs, 0, pre_interp,
							post_interp);
			} else {
				// AMReX only implements FillPatch class for cell-centered data
				// See extern/amrex/Src/AmrCore/AMReX_FillPatcher.H
				// AMReX PR with explanation: https://github.com/AMReX-Codes/amrex/pull/2972
				amrex::Abort("FillPatchType::fillpatch_class is not implemented for non-cell-centered data! Use "
					     "FillPatchType::fillpatch_function instead.");
			}
		} else {
			if (cen == quokka::centering::cc) {
				amrex::FillPatchTwoLevels(mf, time, coarseData, coarseTime, fineData, fineTime, 0, icomp, ncomp, geom[lev - 1], geom[lev],
							  coarsePhysicalBoundaryFunctor_cc, 0, finePhysicalBoundaryFunctor_cc, 0, refRatio(lev - 1),
							  getAmrInterpolaterCellCentered(), BCs, 0, pre_interp, post_interp);
			} else if (cen == quokka::centering::fc) {
				amrex::FillPatchTwoLevels(mf, time, coarseData, coarseTime, fineData, fineTime, 0, icomp, ncomp, geom[lev - 1], geom[lev],
							  coarsePhysicalBoundaryFunctor_fc, 0, finePhysicalBoundaryFunctor_fc, 0, refRatio(lev - 1),
							  getAmrInterpolaterFaceCentered(), BCs, 0, pre_interp, post_interp);
			} else {
				amrex::Abort("AMR interpolation is not implemented for this zone centering!");
			}
		}
	}
}

// Fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
template <typename problem_t>
void AMRSimulation<problem_t>::FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs,
					       quokka::centering cen, quokka::direction dir)
{							// here neco
	BL_PROFILE("AMRSimulation::FillCoarsePatch()"); // NOLINT(misc-const-correctness)

	AMREX_ASSERT(lev > 0);

	amrex::Vector<amrex::MultiFab *> cmf;
	amrex::Vector<amrex::Real> ctime;
	GetData(lev - 1, time, cmf, ctime, cen, dir);

	if (cmf.size() != 1) {
		amrex::Abort("FillCoarsePatch: how did this happen?");
	}

	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(setBoundaryFunctor<problem_t>{});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor(geom[lev], BCs, boundaryFunctor);
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor(geom[lev - 1], BCs, boundaryFunctor);

	if (cen == quokka::centering::cc) {
		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
					     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), getAmrInterpolaterCellCentered(), BCs, 0);
	} else if (cen == quokka::centering::fc) {
		amrex::Interpolater *face_mapper = &amrex::face_divfree_interp;
		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
					     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), face_mapper, BCs, 0);
	} else {
		amrex::Abort("AMR interpolation is not implemented for this zone centering!");
	}
}

// Fill face-centred data for all directions simultaneously using divergence-free interpolation
template <typename problem_t>
void AMRSimulation<problem_t>::FillCoarsePatchFaceArray(int lev, amrex::Real time, amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> &mf_array, int icomp,
							int ncomp, amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> &BCs_array)
{
	BL_PROFILE("AMRSimulation::FillCoarsePatchFaceArray()"); // NOLINT(misc-const-correctness)

	AMREX_ASSERT(lev > 0);

	amrex::Array<amrex::Vector<amrex::MultiFab *>, AMREX_SPACEDIM> cmf_array;
	amrex::Vector<amrex::Real> ctime;
	GetDataFaceArray(lev - 1, time, cmf_array, ctime);

	if (ctime.size() != 1) {
		amrex::Abort("FillCoarsePatchFaceArray: time interpolation not yet implemented for face arrays");
	}

	amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> cmf_ptrs;
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		if (cmf_array[idim].size() != 1) {
			amrex::Abort("FillCoarsePatchFaceArray: unexpected number of face-centred MultiFabs");
		}
		cmf_ptrs[idim] = cmf_array[idim][0];
	}

	using BndryFunc = amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>;
	amrex::Array<amrex::PhysBCFunct<BndryFunc>, AMREX_SPACEDIM> finePhysicalBoundaryFunctor;
	amrex::Array<amrex::PhysBCFunct<BndryFunc>, AMREX_SPACEDIM> coarsePhysicalBoundaryFunctor;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		const auto dir = static_cast<quokka::direction>(idim);
		BndryFunc boundaryFunctor(setBoundaryFunctorFaceVar<problem_t>{dir});
		finePhysicalBoundaryFunctor[idim] = amrex::PhysBCFunct<BndryFunc>(geom[lev], BCs_array[idim], boundaryFunctor);
		coarsePhysicalBoundaryFunctor[idim] = amrex::PhysBCFunct<BndryFunc>(geom[lev - 1], BCs_array[idim], boundaryFunctor);
	}

	amrex::InterpFromCoarseLevel(mf_array, time, cmf_ptrs, 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
				     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), &amrex::face_divfree_interp, BCs_array, 0);
}

// utility to copy in data from state_old_cc_[lev] and/or state_new_cc_[lev]
// into another multifab
template <typename problem_t>
void AMRSimulation<problem_t>::GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data, amrex::Vector<amrex::Real> &datatime,
				       quokka::centering cen, quokka::direction dir)
{
	BL_PROFILE("AMRSimulation::GetData()"); // NOLINT(misc-const-correctness)

	if ((cen != quokka::centering::cc) && (cen != quokka::centering::fc)) {
		amrex::Print() << "Centering passed to GetData(): " << static_cast<int>(cen) << "\n";
		throw std::runtime_error("Only cell-centred (cc) and face-centred (fc) variables are supported, thus far.");
	}

	data.clear();
	datatime.clear();

	const int dim = static_cast<int>(dir);

	if (amrex::almostEqual(time, tNew_[lev], 5)) { // if time == tNew_[lev] within roundoff
		datatime.push_back(tNew_[lev]);
		if (cen == quokka::centering::cc) {
			data.push_back(&state_new_cc_[lev]);
		} else if (cen == quokka::centering::fc) {
			data.push_back(&state_new_fc_[lev][dim]);
		}
	} else if (amrex::almostEqual(time, tOld_[lev], 5)) { // if time == tOld_[lev] within roundoff
		datatime.push_back(tOld_[lev]);
		if (cen == quokka::centering::cc) {
			data.push_back(&state_old_cc_[lev]);
		} else if (cen == quokka::centering::fc) {
			data.push_back(&state_old_fc_[lev][dim]);
		}
	} else { // otherwise return both old and new states for interpolation
		datatime.push_back(tOld_[lev]);
		datatime.push_back(tNew_[lev]);
		if (cen == quokka::centering::cc) {
			data.push_back(&state_old_cc_[lev]);
			data.push_back(&state_new_cc_[lev]);
		} else if (cen == quokka::centering::fc) {
			data.push_back(&state_old_fc_[lev][dim]);
			data.push_back(&state_new_fc_[lev][dim]);
		}
	}
}

// Retrieve face-centred data for all spatial directions at the requested time
template <typename problem_t>
void AMRSimulation<problem_t>::GetDataFaceArray(int lev, amrex::Real time, amrex::Array<amrex::Vector<amrex::MultiFab *>, AMREX_SPACEDIM> &data_array,
						amrex::Vector<amrex::Real> &datatime)
{
	BL_PROFILE("AMRSimulation::GetDataFaceArray()"); // NOLINT(misc-const-correctness)

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		data_array[idim].clear();
	}
	datatime.clear();

	if (amrex::almostEqual(time, tNew_[lev], 5)) {
		datatime.push_back(tNew_[lev]);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			data_array[idim].push_back(&state_new_fc_[lev][idim]);
		}
	} else if (amrex::almostEqual(time, tOld_[lev], 5)) {
		datatime.push_back(tOld_[lev]);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			data_array[idim].push_back(&state_old_fc_[lev][idim]);
		}
	} else {
		datatime.push_back(tOld_[lev]);
		datatime.push_back(tNew_[lev]);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			data_array[idim].push_back(&state_old_fc_[lev][idim]);
			data_array[idim].push_back(&state_new_fc_[lev][idim]);
		}
	}
}

// average down on all levels
template <typename problem_t> void AMRSimulation<problem_t>::AverageDown()
{
	BL_PROFILE("AMRSimulation::AverageDown()"); // NOLINT(misc-const-correctness)

	for (int lev = finest_level - 1; lev >= 0; --lev) {
		AverageDownTo(lev);
	}
}

// set covered coarse cells to be the average of overlying fine cells
template <typename problem_t> void AMRSimulation<problem_t>::AverageDownTo(int crse_lev)
{
	BL_PROFILE("AMRSimulation::AverageDownTo()"); // NOLINT(misc-const-correctness)

	// cell-centred
	amrex::average_down(state_new_cc_[crse_lev + 1], state_new_cc_[crse_lev], geom[crse_lev + 1], geom[crse_lev], 0, state_new_cc_[crse_lev].nComp(),
			    refRatio(crse_lev));

	// face-centred
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		// for each face-centering (number of dimensions)
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::average_down_faces(state_new_fc_[crse_lev + 1][idim], state_new_fc_[crse_lev][idim], refRatio(crse_lev), geom[crse_lev]);
		}
	}
}

template <typename problem_t> template <typename F> auto AMRSimulation<problem_t>::computeVolumeIntegral(F const &user_f) -> amrex::Real
{
	// compute integral of user_f(i, j, k, state) along the given axis.
	const BL_PROFILE("AMRSimulation::computeVolumeIntegral()");

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
	}

	// evaluate user_f on all levels
	// (note: it is not necessary to average down)
	for (int lev = 0; lev <= finest_level; ++lev) {
		auto const &state = state_new_cc_[lev].const_arrays();
		auto const &result = q[lev].arrays();
		amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) { result[bx](i, j, k) = user_f(i, j, k, state[bx]); });
	}
	amrex::Gpu::streamSynchronize();

	// call amrex::volumeWeightedSum
	const amrex::Real result = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(q), 0, geom, ref_ratio);
	return result;
}

template <typename problem_t> void AMRSimulation<problem_t>::InitParticles()
{
	const BL_PROFILE("AMRSimulation::InitParticles()");
	if (do_tracers != 0) {
		AMREX_ASSERT(TracerPC == nullptr);
		TracerPC = std::make_unique<amrex::AmrTracerParticleContainer>(this);

		const amrex::AmrTracerParticleContainer::ParticleInitData pdata = {{AMREX_D_DECL(0.0, 0.0, 0.0)}, {}, {}, {}};

		TracerPC->SetVerbose(0);
		TracerPC->InitOnePerCell(0.5, 0.5, 0.5, pdata);
		TracerPC->Redistribute();
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::InitPhyParticles()
{
	const BL_PROFILE("AMRSimulation::InitPhyParticles()");
	// Verify that particle_switch is of the correct type
	detail::verify_particle_switch_type<problem_t>();

	// Read particle parameters from input file
	quokka::particleParmParse();

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Rad) {
		AMREX_ASSERT(RadParticles == nullptr);

		// Create particle container
		RadParticles = std::make_unique<quokka::RadParticleContainer<problem_t>>(this);
		RadParticles->SetVerbose(0);

		// Register with particle register - Rad particles do not allow creation
		particleRegister_.registerParticleType(RadParticles.get(), quokka::ParticleType::Rad);

		// Initialize particles through user-defined function
		createInitialRadParticles();
	}

#if AMREX_SPACEDIM == 3
	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CIC) {
		AMREX_ASSERT(CICParticles == nullptr);

		// Create particle container
		CICParticles = std::make_unique<quokka::CICParticleContainer>(this);
		CICParticles->SetVerbose(0);

		// Register with particle register - CIC particles allow creation
		particleRegister_.registerParticleType(CICParticles.get(), quokka::ParticleType::CIC);

		// Initialize particles through user-defined function
		createInitialCICParticles();
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CICRad) {
		AMREX_ASSERT(CICRadParticles == nullptr);

		// Create particle container
		CICRadParticles = std::make_unique<quokka::CICRadParticleContainer<problem_t>>(this);
		CICRadParticles->SetVerbose(0);

		// Register with particle register - CICRad particles do not allow creation
		particleRegister_.registerParticleType(CICRadParticles.get(), quokka::ParticleType::CICRad);

		// Initialize particles through user-defined function
		createInitialCICRadParticles();
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::StochasticStellarPop) {
		AMREX_ASSERT(StochasticStellarPopParticles == nullptr);

		static_assert(Physics_Traits<problem_t>::unit_system == UnitSystem::CGS, "UnitSystem must be CGS for StochasticStellarPop particles");

		// Create particle container
		StochasticStellarPopParticles = std::make_unique<quokka::StochasticStellarPopParticleContainer<problem_t>>(this);
		StochasticStellarPopParticles->SetVerbose(0);

		// Register with particle register - StochasticStellarPop particles allow creation
		particleRegister_.registerParticleType(StochasticStellarPopParticles.get(), quokka::ParticleType::StochasticStellarPop);

		// Initialize particles through user-defined function
		createInitialStochasticStellarPopParticles();
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Sink) {
		AMREX_ASSERT(SinkParticles == nullptr);

		// Create particle container
		SinkParticles = std::make_unique<quokka::SinkParticleContainer>(this);
		SinkParticles->SetVerbose(0);

		// Register with particle register - Sink particles allow creation
		particleRegister_.registerParticleType(SinkParticles.get(), quokka::ParticleType::Sink);

		// Initialize particles through user-defined function
		createInitialSinkParticles();
	}
	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Test) {
		AMREX_ASSERT(TestParticles == nullptr);

		// Create particle container
		TestParticles = std::make_unique<quokka::TestParticleContainer<problem_t>>(this);
		TestParticles->SetVerbose(0);

		// Register with particle register - Test particles have all features enabled
		particleRegister_.registerParticleType(TestParticles.get(), quokka::ParticleType::Test);

		// Initialize particles through user-defined function
		createInitialTestParticles();
	}
#endif // AMREX_SPACEDIM == 3

	particleRegister_.redistribute(0);
}

// get plotfile name
template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileName(int lev) const -> std::string { return amrex::Concatenate(plot_file, lev, 5); }

// get plotfile name
template <typename problem_t> auto AMRSimulation<problem_t>::CustomPlotFileName(const char *base, int lev) const -> std::string
{
	const std::string base_str(base);
	return amrex::Concatenate(base_str, lev, 5);
}

template <typename problem_t>
void AMRSimulation<problem_t>::AverageFCToCC(amrex::MultiFab &mf_cc, const amrex::MultiFab &mf_fc, int idim, int dstcomp_start, int srccomp_start,
					     int srccomp_total) const
{
	int di = 0;
	int dj = 0;
	int dk = 0;
	if (idim == 0) {
		di = 1;
	} else if (idim == 1) {
		dj = 1;
	} else if (idim == 2) {
		dk = 1;
	}
	// iterate over the domain
	auto const &state_cc = mf_cc.arrays();
	auto const &state_fc = mf_fc.const_arrays();
	int const ng_cc = mf_cc.nGrow();
	int const ng_fc = mf_fc.nGrow();
	AMREX_ALWAYS_ASSERT(ng_cc <= ng_fc); // if this is false, we can't fill the ghost cells!
	amrex::ParallelFor(mf_cc, amrex::IntVect(AMREX_D_DECL(ng_cc, ng_cc, ng_cc)), [=] AMREX_GPU_DEVICE(int boxidx, int i, int j, int k) {
		for (int icomp = 0; icomp < srccomp_total; ++icomp) {
			state_cc[boxidx](i, j, k, dstcomp_start + icomp) =
			    0.5 * (state_fc[boxidx](i, j, k, srccomp_start + icomp) + state_fc[boxidx](i + di, j + dj, k + dk, srccomp_start + icomp));
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMFAtLevel_cc(const int lev, const int included_ghosts) -> amrex::MultiFab
{
	const int ncomp_plotMF = plotfileVarsToInclude_cc_.size();
	amrex::MultiFab plotMF(grids[lev], dmap[lev], ncomp_plotMF, included_ghosts);

	if (included_ghosts > 0) {
		// Fill ghost zones for state_new_cc_
		fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, tNew_[lev], quokka::centering::cc, quokka::direction::na, InterpHookNone,
				       InterpHookNone, FillPatchType::fillpatch_function);

		// Fill ghost zones for state_new_fc_
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, tNew_[lev], quokka::centering::fc,
						       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
			}
		}
	}

	// Process each variable in the configurable list
	int comp = 0;
	for (const std::string &varname : plotfileVarsToInclude_cc_) {
		// Check if it's a cell-centered variable
		auto cc_it = std::find(componentNames_cc_.begin(), componentNames_cc_.end(), varname);
		if (cc_it != componentNames_cc_.end()) {
			int cc_comp = std::distance(componentNames_cc_.begin(), cc_it);
			amrex::MultiFab::Copy(plotMF, state_new_cc_[lev], cc_comp, comp, 1, included_ghosts);
			comp++;
			continue;
		}

		// Check if it's a face-centered variable
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			auto fc_it = std::find(componentNames_fc_flat_.begin(), componentNames_fc_flat_.end(), varname);
			if (fc_it != componentNames_fc_flat_.end()) {
				const int fc_comp_flat = std::distance(componentNames_fc_flat_.begin(), fc_it);
				// componentNames_fc_flat_ is organized as: all dims for var0, then all dims for var1, etc.
				// So for nvarPerDim_fc variables and AMREX_SPACEDIM dimensions:
				// [x-var0, y-var0, z-var0, x-var1, y-var1, z-var1, ...]
				const int var_idx = fc_comp_flat / AMREX_SPACEDIM; // which variable type
				const int idim = fc_comp_flat % AMREX_SPACEDIM;	   // which dimension
				const int fc_comp = var_idx;			   // component index within that dimension's MultiFab
				AverageFCToCC(plotMF, state_new_fc_[lev][idim], idim, comp, fc_comp, 1);
				comp++;
				continue;
			}
		}

		// Check if it's a derived variable
		auto deriv_it = std::find(derivedNames_.begin(), derivedNames_.end(), varname);
		if (deriv_it != derivedNames_.end()) {
			ComputeDerivedVar(lev, varname, plotMF, comp);
			comp++;
			continue;
		}

		// Variable not found in any category
		amrex::Abort("PlotFileMFAtLevel_cc: Variable '" + varname + "' not found in any component list");
	}

	return plotMF;
}

template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMFAtLevel_fc(const int lev, int idim, const int nghost_fc_) -> amrex::MultiFab
{
	int comp = 0;
	int nvar_dim_tot_fc = 0;
	if constexpr (Physics_Indices<problem_t>::nvarPerDim_fc > 0) {
		nvar_dim_tot_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
	}
	const int ncomp_plotMF_fc = nvar_dim_tot_fc;

	amrex::MultiFab plotMF_fc(amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)), dmap[lev], ncomp_plotMF_fc, nghost_fc_);
	// Fill ghost zones for state_new_fc_
	if constexpr (Physics_Indices<problem_t>::nvarPerDim_fc > 0) {
		fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, tNew_[lev], quokka::centering::fc,
				       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
	}

	// copy data from face-centred state variables
	for (int i = 0; i < ncomp_plotMF_fc; i++) {
		amrex::MultiFab::Copy(plotMF_fc, state_new_fc_[lev][idim], i, comp, 1, nghost_fc_);
		comp++;
	}
	return plotMF_fc;
}

// put together an array of multifabs for writing
template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMF_cc(const int included_ghosts) -> amrex::Vector<amrex::MultiFab>
{
	amrex::Vector<amrex::MultiFab> r;
	for (int i = 0; i <= finest_level; ++i) {
		r.push_back(PlotFileMFAtLevel_cc(i, included_ghosts));
	}
	return r;
}

template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMF_fc(const int nghost_fc_) -> std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM>
{
	std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM> r_fc;
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		for (int i = 0; i <= finest_level; ++i) {
			r_fc[idim].push_back(PlotFileMFAtLevel_fc(i, idim, nghost_fc_));
		}
	}

	return r_fc;
}

template <typename problem_t> void AMRSimulation<problem_t>::createDiagnostics()
{
	std::string const code_prefix = "quokka";
	amrex::ParmParse const pp(code_prefix);
	amrex::Vector<std::string> diags;

	int const n_diags = pp.countval("diagnostics");
	if (n_diags > 0) {
		m_diagnostics.resize(n_diags);
		diags.resize(n_diags);
	}

	for (int n = 0; n < n_diags; ++n) {
		pp.get("diagnostics", diags[n], n);
		std::string const diag_prefix = code_prefix + "." + diags[n];
		amrex::ParmParse const ppd(diag_prefix);
		std::string diag_type;
		ppd.get("type", diag_type);
		m_diagnostics[n] = DiagBase::create(diag_type);
		m_diagnostics[n]->init(diag_prefix, diags[n]);
		m_diagnostics[n]->addVars(m_diagVars);
	}

	// Remove duplicates from m_diagVars and check that all the variables exist
	std::sort(m_diagVars.begin(), m_diagVars.end());
	auto last = std::unique(m_diagVars.begin(), m_diagVars.end());
	m_diagVars.erase(last, m_diagVars.end());

	auto isVarName = [this](std::string const &v) {
		auto const varnames = GetPlotfileVarNames();
		return std::any_of(varnames.cbegin(), varnames.cend(), [v](std::string const &var) { return v == var; });
	};

	for (auto &v : m_diagVars) {
		if (!isVarName(v)) {
			amrex::Abort("[Diagnostics] Field " + v + " is not available!");
		}
	}

	// Check if any diagnostic is a PlotFile type and disable regular plotfile output
	for (const auto &diag : m_diagnostics) {
		if (dynamic_cast<DiagPlotfile *>(diag.get()) != nullptr) {
			if (plotfileInterval_ > 0) {
				amrex::Print() << "DiagPlotfile detected: disabling regular plotfile output (plotfileInterval_ set to -1)\n";
				plotfileInterval_ = -1;
			}
			break;
		}
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::updateDiagnostics()
{
	// Might need to update some internal data as the grid changes
	for (const auto &m_diagnostic : m_diagnostics) {
		if (m_diagnostic->needUpdate()) {
			m_diagnostic->prepare(finestLevel() + 1, Geom(0, finestLevel()), boxArray(0, finestLevel()), dmap, m_diagVars);
		}
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::doDiagnostics()
{
	// Assemble a vector of MF containing the requested data
	const BL_PROFILE("AMRSimulation::doDiagnostics()");
	updateDiagnostics();

	bool const computeVars =
	    std::any_of(m_diagnostics.cbegin(), m_diagnostics.cend(), [this](const auto &diag) { return diag->doDiag(tNew_[0], istep[0]); });

	amrex::Vector<std::unique_ptr<amrex::MultiFab>> diagMFVec(finestLevel() + 1);
	if (computeVars) {
		for (int lev{0}; lev <= finestLevel(); ++lev) {
			diagMFVec[lev] = std::make_unique<amrex::MultiFab>(grids[lev], dmap[lev], m_diagVars.size(), 1);
			amrex::MultiFab const mf = PlotFileMFAtLevel_cc(lev, std::min(nghost_cc_, nghost_fc_));
			auto const varnames = GetPlotfileVarNames();

			for (int v{0}; v < m_diagVars.size(); ++v) {
				// get component index for the 'mf' multifab
				int mf_idx = -1;
				for (int i = 0; i < varnames.size(); ++i) {
					if (m_diagVars[v] == varnames[i]) {
						mf_idx = i;
					}
				}
				AMREX_ALWAYS_ASSERT(mf_idx != -1);
				amrex::MultiFab::Copy(*diagMFVec[lev], mf, mf_idx, v, 1, 1);
			}
		}
	}

	for (const auto &diag : m_diagnostics) {
		if (diag->doDiag(tNew_[0], istep[0])) {
			// Check if this is a DiagPlotfile - if so, call its special writePlotfile method
			auto *plotfileDiag = dynamic_cast<DiagPlotfile *>(diag.get());
			if (plotfileDiag != nullptr) {
				// Prepare full plotfile data
				int const included_ghosts = std::min(nghost_cc_, nghost_fc_);
				amrex::Vector<amrex::MultiFab> mf_cc = PlotFileMF_cc(included_ghosts);
				amrex::Vector<const amrex::MultiFab *> mf_cc_ptr = amrex::GetVecOfConstPtrs(mf_cc);
				auto const varnames = GetPlotfileVarNames();

				// Prepare face-centered data
				std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM> mf_fc = PlotFileMF_fc(nghost_fc_);
				std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> mf_fc_ptr;
				for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
					mf_fc_ptr[idim] = amrex::GetVecOfConstPtrs(mf_fc[idim]);
				}
				auto const varnames_fc = GetPlotfileVarNames_fc();

				// Write cell-centered data and particles
				plotfileDiag->writePlotfile(istep[0], tNew_[0], finestLevel(), mf_cc_ptr, varnames, Geom(0, finestLevel()), istep, refRatio(),
							    particleRegister_, do_tracers, TracerPC.get(), simulationMetadata_);

				// Write face-centered data if present
				plotfileDiag->writePlotfileFC<problem_t>(amrex::Concatenate(plotfileDiag->getDiagFileName(), istep[0], 5), finestLevel(),
									 mf_fc_ptr, varnames_fc, Geom(0, finestLevel()), tNew_[0], istep, refRatio(),
									 simulationMetadata_);
			} else {
				// Check if this is a DiagProjectionPlot - if so, call its special writeProjection method
				auto *projectionDiag = dynamic_cast<DiagProjectionPlot *>(diag.get());
				if (projectionDiag != nullptr) {
					// Compute and write projections for each direction
					for (auto const &dir : projectionDiag->getProjectionDirs()) {
						std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj = ComputeProjections(dir);
						projectionDiag->writeProjection<problem_t>(dir, proj, tNew_[0], istep[0], particleRegister_, simulationMetadata_);
					}
				} else {
					// Regular diagnostic
					diag->processDiag(istep[0], tNew_[0], GetVecOfConstPtrs(diagMFVec), m_diagVars, simulationMetadata_);
				}
			}
		}
	}
}

// do in-situ rendering with Ascent
#ifdef AMREX_USE_ASCENT
template <typename problem_t> void AMRSimulation<problem_t>::AscentCustomActions(conduit::Node const &blueprintMesh)
{
	BL_PROFILE("AMRSimulation::AscentCustomActions()"); // NOLINT(misc-const-correctness)

	// add a scene with a pseudocolor plot
	Node scenes;
	scenes["s1/plots/p1/type"] = "pseudocolor";
	scenes["s1/plots/p1/field"] = "gasDensity";

	// set the output file name (ascent will add ".png")
	scenes["s1/renders/r1/image_prefix"] = "render_density%05d";

	// set camera position
	amrex::Array<double, 3> position = {-0.6, -0.6, -0.8};
	scenes["s1/renders/r1/camera/position"].set_float64_ptr(position.data(), 3);

	// setup actions
	Node actions;
	Node &add_plots = actions.append();
	add_plots["action"] = "add_scenes";
	add_plots["scenes"] = scenes;
	actions.append()["action"] = "execute";
	actions.append()["action"] = "reset";

	// send AMR mesh to ascent, do render
	ascent_.publish(blueprintMesh);
	ascent_.execute(actions); // will be replaced by ascent_actions.yml if present
}

// do Ascent render
template <typename problem_t> void AMRSimulation<problem_t>::RenderAscent()
{
	BL_PROFILE("AMRSimulation::RenderAscent()"); // NOLINT(misc-const-correctness)

	// combine multifabs
	const int included_ghosts = std::min(nghost_cc_, nghost_fc_);
	amrex::Vector<amrex::MultiFab> mf_overlapping = PlotFileMF_cc(included_ghosts);
	amrex::Vector<const amrex::MultiFab *> mf_overlapping_ptr = amrex::GetVecOfConstPtrs(mf_overlapping);
	amrex::Vector<std::string> varnames;
	varnames.insert(varnames.end(), componentNames_cc_.begin(), componentNames_cc_.end());
	varnames.insert(varnames.end(), derivedNames_.begin(), derivedNames_.end());

	// convexify multifabs
	// see: https://github.com/AMReX-Codes/amrex/pull/4013
	amrex::Vector<amrex::MultiFab> mf_convex = amrex::convexify(mf_overlapping_ptr, refRatio());
	amrex::Vector<const amrex::MultiFab *> mf_convex_ptr = amrex::GetVecOfConstPtrs(mf_convex);

	// rescale geometry
	// (Ascent fails to render if you use parsec-size boxes in units of cm...)
	amrex::Vector<amrex::Geometry> rescaledGeom = Geom();
	const amrex::Real length = geom[0].ProbLength(0);
	for (int i = 0; i < rescaledGeom.size(); ++i) {
		auto const &dlo = rescaledGeom[i].ProbLoArray();
		auto const &dhi = rescaledGeom[i].ProbHiArray();
		std::array<amrex::Real, AMREX_SPACEDIM> new_dlo{};
		std::array<amrex::Real, AMREX_SPACEDIM> new_dhi{};
		for (int k = 0; k < AMREX_SPACEDIM; ++k) {
			new_dlo[k] = dlo[k] / length;
			new_dhi[k] = dhi[k] / length;
		}
		amrex::RealBox rescaledRealBox(new_dlo, new_dhi);
		rescaledGeom[i].ProbDomain(rescaledRealBox);
	}

	// wrap MultiFabs into a Blueprint mesh
	conduit::Node blueprintMesh;
	amrex::MultiLevelToBlueprint(finest_level + 1, mf_convex_ptr, varnames, rescaledGeom, tNew_[0], istep, refRatio(), blueprintMesh);

	// copy to host mem (needed for DataBinning)
	conduit::Node bpMeshHost;
	bpMeshHost.set(blueprintMesh);

	// pass Blueprint mesh to Ascent, run actions
	AscentCustomActions(bpMeshHost);
}
#endif // AMREX_USE_ASCENT

template <typename problem_t> auto AMRSimulation<problem_t>::GetPlotfileVarNames() const -> amrex::Vector<std::string> { return plotfileVarsToInclude_cc_; }

template <typename problem_t> auto AMRSimulation<problem_t>::GetPlotfileVarNames_fc() const -> std::array<amrex::Vector<std::string>, AMREX_SPACEDIM>
{
	std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> varnames_fc; // nvarTotal or perDim?
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			for (int icomp = 0; icomp < Physics_Indices<problem_t>::nvarPerDim_fc; ++icomp) {
				varnames_fc[idim].push_back(componentNames_fc_[idim][icomp]);
			}
		}
	}
	return varnames_fc;
}

// write plotfile to disk
template <typename problem_t> void AMRSimulation<problem_t>::WritePlotFile()
{
	BL_PROFILE("AMRSimulation::WritePlotFile()"); // NOLINT(misc-const-correctness)

	if (amrex::AsyncOut::UseAsyncOut()) {
		// ensure that we flush any plotfiles that are currently being written
		amrex::AsyncOut::Finish();
	}

	// now construct output and submit to async write queue
#ifdef QUOKKA_USE_OPENPMD
	int included_ghosts = 0;
#else
	int included_ghosts = std::min(nghost_cc_, nghost_fc_);
#endif
	amrex::Vector<amrex::MultiFab> mf_cc = PlotFileMF_cc(included_ghosts);
	amrex::Vector<const amrex::MultiFab *> mf_cc_ptr = amrex::GetVecOfConstPtrs(mf_cc); // NOLINT(misc-const-correctness)

	const std::string &plotfilename = PlotFileName(istep[0]);
	auto varnames = GetPlotfileVarNames();
	amrex::Print() << "Writing plotfile " << plotfilename << "\n";

#ifdef QUOKKA_USE_OPENPMD
	// TODO(bwibking): write particles using openPMD
	quokka::OpenPMDOutput::WriteFile(varnames, finest_level + 1, mf_cc_ptr, Geom(), plot_file, tNew_[0], istep[0]);
	WriteMetadataFile(plotfilename + ".yaml");
#else
	// sets the maximum number of binary files per MultiFab
	quokka::ScopedVisMFNOutFiles scoped_nfiles(plot_nfiles);

	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf_cc_ptr, varnames, Geom(), tNew_[0], istep, refRatio());
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		// Create fc_vars directory if it doesn't exist
		const std::string fc_vars_dir = plotfilename + "/fc_vars";
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::UtilCreateDirectory(fc_vars_dir, 0755);
		}
		amrex::ParallelDescriptor::Barrier();

		std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM> mf_fc = PlotFileMF_fc(nghost_fc_);
		std::vector<std::string> dimNames = {"x", "y", "z"};
		auto varnames_fc = GetPlotfileVarNames_fc();
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			const amrex::Vector<const amrex::MultiFab *> mf_fc_ptr = amrex::GetVecOfConstPtrs(mf_fc[idim]);
			auto plotfilename_base = plotfilename + "/fc_vars/" + dimNames[idim];
			const std::string &plotfilename_fc = CustomPlotFileName(plotfilename_base.c_str(), istep[0]);
			auto varnames_fc_dim = varnames_fc[idim];
			amrex::WriteMultiLevelPlotfile(plotfilename_fc, finest_level + 1, mf_fc_ptr, varnames_fc_dim, Geom(), tNew_[0], istep, refRatio());
			WriteMetadataFile(plotfilename_fc + "/metadata.yaml");
		}
	}
	WriteMetadataFile(plotfilename + "/metadata.yaml");

	// write particles
	if (do_tracers != 0) {
		TracerPC->WritePlotFile(plotfilename, "tracer_particles");
	}

	// write all particles in particleRegister_ to plotfile
	particleRegister_.writePlotFile(plotfilename);
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::WriteParticleFile()
{
	const BL_PROFILE("AMRSimulation::WriteParticleFile()");

	// Create particle file name using the same pattern as PlotFileName
	const std::string partfilename = amrex::Concatenate("part", istep[0], 5);

	amrex::Print() << "Writing particle file " << partfilename << "\n";

	// Create directory, renaming existing one if it exists (following AMReX pattern)
	amrex::UtilCreateCleanDirectory(partfilename, true);

	// Save particle data to CSV files inside the created directory
	// Only save if particle count <= 1000 for each type
	particleRegister_.saveParticleDataToFileConditional(partfilename, 1000);
}

template <typename problem_t> void AMRSimulation<problem_t>::WriteMetadataFile(std::string const &MetadataFileName) const
{
	// write metadata file
	// (this is written for both checkpoints and plotfiles)

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::ofstream MetadataFile;
		MetadataFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		MetadataFile.open(MetadataFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!MetadataFile.good()) {
			amrex::FileOpenFailed(MetadataFileName);
		}

		// write YAML to MetadataFile
		MetadataFile << simulationMetadata_ << '\n';
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::ReadMetadataFile(std::string const &chkfilename)
{
	fenv_t orig_feenv;
	feholdexcept(&orig_feenv); // disable FPE for YAML reading

	// read metadata file in on all ranks (needed when restarting from checkpoint)
	const std::string MetadataFileName(chkfilename + "/metadata.yaml");

	// read YAML file into simulationMetadata_ std::map
	const YAML::Node metadata = YAML::LoadFile(MetadataFileName);
	amrex::Print() << "Reading " << MetadataFileName << "...\n";

	for (YAML::const_iterator it = metadata.begin(); it != metadata.end(); ++it) {
		const auto key = it->first.as<std::string>();
		const std::optional<amrex::Real> value_real = YAML::as_if<amrex::Real, std::optional<amrex::Real>>(it->second)();
		const std::optional<std::string> value_string = YAML::as_if<std::string, std::optional<std::string>>(it->second)();

		if (value_real) {
			simulationMetadata_[key] = value_real.value();
			amrex::Print() << fmt::format("\t{} = {}\n", key, value_real.value());
		} else if (value_string) {
			simulationMetadata_[key] = value_string.value();
			amrex::Print() << fmt::format("\t{} = {}\n", key, value_string.value());
		} else {
			amrex::Print() << fmt::format("\t{} has unknown type! skipping this entry.\n", key);
		}
	}

	fesetenv(&orig_feenv); // restore FPE
}

template <typename problem_t> void AMRSimulation<problem_t>::WriteStatisticsFile()
{
	// append to statistics file
	static bool isHeaderWritten = false;

	// compute statistics
	// IMPORTANT: the user is responsible for performing any necessary MPI reductions inside ComputeStatistics
	std::map<std::string, amrex::Real> statistics = ComputeStatistics();

	// write to file
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::ofstream StatisticsFile;
		StatisticsFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		StatisticsFile.open(stats_file.c_str(), std::ofstream::out | std::ofstream::app);
		if (!StatisticsFile.good()) {
			amrex::FileOpenFailed(stats_file);
		}

		// write header
		if (!isHeaderWritten) {
			const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			const std::tm now = *std::localtime(&t); // NOLINT(concurrency-mt-unsafe)
			StatisticsFile << "## Simulation restarted at: " << std::put_time(&now, "%c %Z") << "\n";
			StatisticsFile << "# cycle time ";
			for (auto const &[key, value] : statistics) {
				StatisticsFile << key << " ";
			}
			StatisticsFile << "\n";
			isHeaderWritten = true;
		}

		// save statistics to file
		StatisticsFile << istep[0] << " "; // cycle
		StatisticsFile << tNew_[0] << " "; // time
		for (auto const &[key, value] : statistics) {
			StatisticsFile << value << " ";
		}
		StatisticsFile << "\n";

		// file closed automatically by destructor
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::SetLastCheckpointSymlink(std::string const &checkpointname) const
{
	// creates a symlink pointing to the most recent checkpoint

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const std::string lastSymlinkName = "last_chk";

		// remove previous symlink, if it exists
		if (std::filesystem::is_symlink(lastSymlinkName)) {
			std::filesystem::remove(lastSymlinkName);
		}
		// create symlink
		std::filesystem::create_directory_symlink(checkpointname, lastSymlinkName);
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::WriteCheckpointFile() const
{
	BL_PROFILE("AMRSimulation::WriteCheckpointFile()"); // NOLINT(misc-const-correctness)

	// chk00010            write a checkpoint file with this root directory
	// chk00010/Header     this contains information you need to save (e.g.,
	// finest_level, t_new, etc.) and also
	//                     the BoxArrays at each level
	// chk00010/Level_0/
	// chk00010/Level_1/
	// etc.                these subdirectories will hold the MultiFab data at
	// each level of refinement

	// checkpoint file name, e.g., chk00010
	const std::string &checkpointname = amrex::Concatenate(chk_file, istep[0]);

	amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

	const int nlevels = finest_level + 1;

	// ---- prebuild a hierarchy of directories
	// ---- dirName is built first.  if dirName exists, it is renamed.  then build
	// ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
	// ---- if callBarrier is true, call ParallelDescriptor::Barrier()
	// ---- after all directories are built
	// ---- ParallelDescriptor::IOProcessor() creates the directories
	amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

	// write Header file
	if (amrex::ParallelDescriptor::IOProcessor()) {

		std::string HeaderFileName(checkpointname + "/Header");
		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::ofstream HeaderFile;
		HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!HeaderFile.good()) {
			amrex::FileOpenFailed(HeaderFileName);
		}

		HeaderFile.precision(17);

		// write out title line
		HeaderFile << "Checkpoint file for QuokkaCode\n";

		// write out finest_level
		HeaderFile << finest_level << "\n";

		// write out array of istep
		for (int const i : istep) {
			HeaderFile << i << " ";
		}
		HeaderFile << "\n";

		// write out array of dt
		for (double const i : dt_) {
			HeaderFile << i << " ";
		}
		HeaderFile << "\n";

		// write out array of t_new
		for (double const i : tNew_) {
			HeaderFile << i << " ";
		}
		HeaderFile << "\n";

		// write the BoxArray at each level
		for (int lev = 0; lev <= finest_level; ++lev) {
			boxArray(lev).writeOn(HeaderFile);
			HeaderFile << '\n';
		}
	}

	// write Metadata file
	WriteMetadataFile(checkpointname + "/metadata.yaml");

	// set the maximum number of binary files per MultiFab
	quokka::ScopedVisMFNOutFiles scoped_nfiles(checkpoint_nfiles);

	// write the cell-centred MultiFab data to, e.g., chk00010/Level_0/
	for (int lev = 0; lev <= finest_level; ++lev) {
		amrex::VisMF::Write(state_new_cc_[lev], amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "Cell"));
		amrex::ParallelDescriptor::Barrier(); // needed to avoid overwhelming Lustre I/O on Frontier
	}

	// write the face-centred MultiFab data to, e.g., chk00010/Level_0/
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			for (int lev = 0; lev <= finest_level; ++lev) {
				amrex::VisMF::Write(state_new_fc_[lev][idim], amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_",
													    std::string("Face_") + quokka::face_dir_str[idim]));
				amrex::ParallelDescriptor::Barrier(); // needed to avoid overwhelming Lustre I/O on Frontier
			}
		}
	}

	// write particle data
	if (do_tracers != 0) {
		TracerPC->Checkpoint(checkpointname, "tracer_particles", true);
	}

	// write all particles in particleRegister_ to checkpoint file
	particleRegister_.writeCheckpoint(checkpointname, true);

	// create symlink and point it at this checkpoint dir
	SetLastCheckpointSymlink(checkpointname);
}

// utility to skip to next line in Header
inline void GotoNextLine(std::istream &is)
{
	constexpr std::streamsize bl_ignore_max{100000};
	is.ignore(bl_ignore_max, '\n');
}

template <typename problem_t>
auto AMRSimulation<problem_t>::detectRefinementContext(const amrex::BoxArray &restart_ba, const amrex::Geometry &current_geom) ->
    typename AMRSimulation<problem_t>::RefinementContext
{
	RefinementContext context;

	amrex::Box const reDom = restart_ba.minimalBox();
	amrex::Box const inDom = current_geom.Domain();
	const amrex::IntVect restartGrid{
	    AMREX_D_DECL(reDom.bigEnd(0) - reDom.smallEnd(0) + 1, reDom.bigEnd(1) - reDom.smallEnd(1) + 1, reDom.bigEnd(2) - reDom.smallEnd(2) + 1)};
	const amrex::IntVect inputGrid{
	    AMREX_D_DECL(inDom.bigEnd(0) - inDom.smallEnd(0) + 1, inDom.bigEnd(1) - inDom.smallEnd(1) + 1, inDom.bigEnd(2) - inDom.smallEnd(2) + 1)};

	if (restartGrid != inputGrid) {
		amrex::Print() << "Input grid dimensions on level 0: " << inputGrid << "\n";
		amrex::Print() << "Restart file grid dimensions on level 0: " << restartGrid << "\n";
		// compute rescale factor
		const int rescaleFac = inputGrid[0] / restartGrid[0];
		amrex::Print() << "Rescaling MultiFabs in restart file by a factor of " << rescaleFac << "...\n";
		// make sure the grid size differs by the same integer factor for all dimensions
		if (amrex::ParallelDescriptor::IOProcessor()) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(restartGrid[idim] * rescaleFac == inputGrid[idim],
								 "Simulation has been restarted with a grid size that is not an integer "
								 "multiple of the grid written to disk!");
			}
		}
		// set refinement factor and create coarse level 0 geometry
		context.refinement_factor = rescaleFac;
		amrex::IntVect is_per = current_geom.periodicity().intVect();
		const amrex::Array<int, AMREX_SPACEDIM> is_per_arr{AMREX_D_DECL(is_per[0], is_per[1], is_per[2])};
		context.coarse_level0_geom = amrex::Geometry(reDom, current_geom.ProbDomain(), amrex::CoordSys::cartesian, is_per_arr);
	}

	return context;
}

template <typename problem_t> auto AMRSimulation<problem_t>::readCheckpointHeader(const std::string &restart_file) -> amrex::Vector<amrex::BoxArray>
{
	// Header
	std::string const File(restart_file + "/Header");

	const amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

	amrex::Vector<char> fileCharPtr;
	amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
	const std::string fileCharPtrString(fileCharPtr.dataPtr());
	std::istringstream is(fileCharPtrString, std::istringstream::in);

	std::string line;
	std::string word;

	// read in title line
	std::getline(is, line);

	// read in finest_level
	is >> finest_level;
	GotoNextLine(is);

	// read in array of istep
	std::getline(is, line);
	{
		std::istringstream lis(line);
		int i = 0;
		while (lis >> word) {
			istep[i++] = std::stoi(word);
		}
	}

	// read in array of dt
	std::getline(is, line);
	{
		std::istringstream lis(line);
		int i = 0;
		while (lis >> word) {
			dt_[i++] = std::stod(word);
		}
	}

	// read in array of t_new
	std::getline(is, line);
	{
		std::istringstream lis(line);
		int i = 0;
		while (lis >> word) {
			tNew_[i++] = std::stod(word);
		}
	}

	// read in BoxArrays for all levels
	amrex::Vector<amrex::BoxArray> ba(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		ba[lev].readFrom(is);
		GotoNextLine(is);
	}
	return ba;
}

template <typename problem_t>
void AMRSimulation<problem_t>::interpolateMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source, const RefinementContext &context,
							      const amrex::Geometry &coarse_geom, const amrex::Geometry &fine_geom,
							      const amrex::Vector<amrex::BCRec> &bcs)
{
	if (!context.needs_refinement()) {
		// if not refining, ParallelCopy
		target.ParallelCopy(source, 0, 0, source.nComp(), target.nGrowVect(), source.nGrowVect());
	} else {
		// if refining, InterpFromCoarseLevel
		amrex::IntVect restart_ref_ratio{AMREX_D_DECL(context.refinement_factor, context.refinement_factor, context.refinement_factor)};
		using BndryFunc = amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>;
		BndryFunc boundaryFunctor(setBoundaryFunctor<problem_t>{});
		amrex::PhysBCFunct<BndryFunc> fineBdryFunct(fine_geom, bcs, boundaryFunctor);
		amrex::PhysBCFunct<BndryFunc> coarseBdryFunct(coarse_geom, bcs, boundaryFunctor);
		// CRITICALLY IMPORTANT: the refined multifabs are NOT properly nested
		//   with respect to the multifabs in the checkpoints! this means we can
		//   only do piecewise **constant** refinement.
		amrex::MFInterpolater *mapper = &amrex::mf_pc_interp;
		amrex::InterpFromCoarseLevel(target, 0., source, 0, 0, source.nComp(), coarse_geom, fine_geom, coarseBdryFunct, 0, fineBdryFunct, 0,
					     restart_ref_ratio, mapper, bcs, 0);
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::interpolateFaceCenteredMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source,
									  const RefinementContext &context, const amrex::Geometry &coarse_geom,
									  const amrex::Geometry &fine_geom, quokka::direction dir)
{
	if (!context.needs_refinement()) {
		// if not refining, ParallelCopy
		target.ParallelCopy(source, 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_, nghost_fc_);
	} else {
		// if refining, InterpFromCoarseLevel
		amrex::IntVect restart_ref_ratio{AMREX_D_DECL(context.refinement_factor, context.refinement_factor, context.refinement_factor)};
		using BndryFunc = amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>;
		BndryFunc boundaryFunctor(setBoundaryFunctorFaceVar<problem_t>{dir});
		amrex::PhysBCFunct<BndryFunc> fineBdryFunct(fine_geom, BCs_fc_, boundaryFunctor);
		amrex::PhysBCFunct<BndryFunc> coarseBdryFunct(coarse_geom, BCs_fc_, boundaryFunctor);
		amrex::Interpolater *face_mapper = &amrex::face_divfree_interp;
		amrex::InterpFromCoarseLevel(target, 0., source, 0, 0, source.nComp(), coarse_geom, fine_geom, coarseBdryFunct, 0, fineBdryFunct, 0,
					     restart_ref_ratio, face_mapper, BCs_fc_, 0);
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::loadMultiFabData(const RefinementContext &context)
{
	for (int lev = 0; lev <= finest_level; ++lev) {
		amrex::Geometry coarse_geom;
		if (lev == 0) {
			coarse_geom = context.coarse_level0_geom;
		} else {
			coarse_geom = geom[lev - 1];
		}

		// cell-centred data
		amrex::MultiFab tmp;
		amrex::VisMF::Read(tmp, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Cell"));
		interpolateMultiFabFromRestart(state_new_cc_[lev], tmp, context, coarse_geom, geom[lev], BCs_cc_);
		AMREX_ALWAYS_ASSERT(!state_new_cc_[lev].contains_nan(0, state_new_cc_[lev].nComp())); // check valid cells

		// face-centred data
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::MultiFab tmp_fc;
				amrex::VisMF::Read(
				    tmp_fc, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", std::string("Face_") + quokka::face_dir_str[idim]));
#if AMREX_SPACEDIM == 1
				constexpr std::array<quokka::direction, 1> directions = {quokka::direction::x};
#elif AMREX_SPACEDIM == 2
				constexpr std::array<quokka::direction, 2> directions = {quokka::direction::x, quokka::direction::y};
#elif AMREX_SPACEDIM == 3
				constexpr std::array<quokka::direction, 3> directions = {quokka::direction::x, quokka::direction::y, quokka::direction::z};
#endif
				interpolateFaceCenteredMultiFabFromRestart(state_new_fc_[lev][idim], tmp_fc, context, coarse_geom, geom[lev], directions[idim]);
				AMREX_ALWAYS_ASSERT(!state_new_fc_[lev][idim].contains_nan(0, state_new_fc_[lev][idim].nComp())); // check valid faces
			}
		}
	}
}

template <typename problem_t> auto AMRSimulation<problem_t>::loadBalanceOnRestart(const amrex::BoxArray &input_ba, int lev) -> amrex::BoxArray
{
	if (lev == 0) {
		// For level 0, create optimally-sized boxes from the domain
		amrex::IntVect fac(2);
		const amrex::Box dom = geom[lev].Domain();
		const amrex::Box dom2 = amrex::refine(amrex::coarsen(dom, 2), 2);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (dom.length(idim) != dom2.length(idim)) {
				fac[idim] = 1;
			}
		}
		amrex::BoxArray ba_lev(amrex::coarsen(dom, fac));
		ba_lev.maxSize(max_grid_size[lev] / fac);
		ba_lev.refine(fac);
		// Boxes in ba have even number of cells in each direction
		// unless the domain has odd number of cells in that direction.
		ChopGrids(lev, ba_lev, amrex::ParallelDescriptor::NProcs());
		return ba_lev;
	}

	// For higher levels, use the input BoxArray and chop grids for better load balancing
	amrex::BoxArray ba_lev = input_ba;
	ChopGrids(lev, ba_lev, amrex::ParallelDescriptor::NProcs());
	return ba_lev;
}

template <typename problem_t> void AMRSimulation<problem_t>::ReadCheckpointFile()
{
	BL_PROFILE("AMRSimulation::ReadCheckpointFile()"); // NOLINT(misc-const-correctness)

	amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

	// 1. Read header information
	auto header_box_arrays = readCheckpointHeader(restart_chkfile);

	// 3. Detect refinement context
	auto refinement_context = detectRefinementContext(header_box_arrays[0], geom[0]);
	restartRefineFactor_ = refinement_context.refinement_factor;

	// 4. Set up grid structure from header
	for (int lev = 0; lev <= finest_level; ++lev) {
		amrex::BoxArray ba(header_box_arrays[lev]);
		if (restartRefineFactor_ > 1) {
			// refine boxes by restartRefineFactor
			ba.refine(restartRefineFactor_);
		}

		// load balancing: create new BoxArray for a possibly different number of MPI ranks
		ba = loadBalanceOnRestart(ba, lev);

		// create a distribution mapping
		amrex::DistributionMapping dm{ba, amrex::ParallelDescriptor::NProcs()};

		// set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
		SetBoxArray(lev, ba);
		SetDistributionMap(lev, dm);

		// build MultiFab and FluxRegister data
		const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
		const int nghost_cc = nghost_cc_;
		state_old_cc_[lev].define(grids[lev], dmap[lev], ncomp_cc, nghost_cc);
		state_new_cc_[lev].define(grids[lev], dmap[lev], ncomp_cc, nghost_cc);
		max_signal_speed_[lev].define(ba, dm, 1, nghost_cc);

		if (lev > 0 && (do_reflux != 0)) {
			flux_reg_[lev] = std::make_unique<amrex::FluxRegister>(ba, dm, refRatio(lev - 1), lev, ncomp_cc);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				const int nemf_vars = 1;
				emf_reg_[lev] = std::make_unique<amrex::EdgeFluxRegister>(ba, boxArray(lev - 1), dm, DistributionMap(lev - 1), Geom(lev),
											  Geom(lev - 1), nemf_vars);
			}
		}

		const int ncomp_per_dim_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
		const int nghost_fc = nghost_fc_;
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				state_new_fc_[lev][idim] =
				    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
				state_old_fc_[lev][idim] =
				    amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			}
		}
	}

	ReadMetadataFile(restart_chkfile);

	// 5. Load MultiFab data with refinement handling
	loadMultiFabData(refinement_context);

	// read particle data
	if (do_tracers != 0) {
		AMREX_ASSERT(TracerPC == nullptr);
		TracerPC = std::make_unique<amrex::AmrTracerParticleContainer>(this);
		restartParticleContainerWithRefinement(TracerPC, restart_chkfile, "tracer_particles", header_box_arrays);
	}

	// 6. Initialize and register particle containers from checkpoint file

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Rad) {
		initializeParticleContainerFromCheckpoint(RadParticles, quokka::ParticleType::Rad, header_box_arrays);
	}

#if AMREX_SPACEDIM == 3
	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CIC) {
		initializeParticleContainerFromCheckpoint(CICParticles, quokka::ParticleType::CIC, header_box_arrays);
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CICRad) {
		initializeParticleContainerFromCheckpoint(CICRadParticles, quokka::ParticleType::CICRad, header_box_arrays);
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::StochasticStellarPop) {
		initializeParticleContainerFromCheckpoint(StochasticStellarPopParticles, quokka::ParticleType::StochasticStellarPop, header_box_arrays);
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Sink) {
		initializeParticleContainerFromCheckpoint(SinkParticles, quokka::ParticleType::Sink, header_box_arrays);
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Test) {
		initializeParticleContainerFromCheckpoint(TestParticles, quokka::ParticleType::Test, header_box_arrays);
	}
#endif // AMREX_SPACEDIM == 3

	areInitialConditionsDefined_ = true;
}

template <typename problem_t>
template <typename ParticleContainer>
void AMRSimulation<problem_t>::restartParticleContainerWithRefinement(std::unique_ptr<ParticleContainer> &particles, std::string const &restart_chkfile,
								      std::string const &particle_type_name,
								      amrex::Vector<amrex::BoxArray> const &header_box_arrays)
{
	// Check whether there are any particles to read
	std::string const pc_path = restart_chkfile + "/" + particle_type_name;

	// Check if the particle checkpoint directory exists
	if (!amrex::FileSystem::Exists(pc_path)) {
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "WARNING: Particle checkpoint directory not found: " << pc_path << "\n";
		}
		return;
	}

	// Check for subdirectories with "Level_" prefix
	int has_level_dirs = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Check for Level_N directories from 0 to finestLevel
		const int finest_level = finestLevel();
		for (int lev = 0; lev <= finest_level; ++lev) {
			std::string const level_path = pc_path + "/Level_" + std::to_string(lev);
			if (amrex::FileSystem::Exists(level_path)) {
				has_level_dirs = 1;
				break;
			}
		}
	}

	// Broadcast the result to all processors
	// (NOTE: ParallelDescriptor does not support Bcast for bool types, so we use int instead.)
	amrex::ParallelDescriptor::Bcast(&has_level_dirs, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	// If no Level_ directories found, there are no particles to read
	if (!has_level_dirs) {
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "WARNING: No particle data found in checkpoint (no Level_* directories) in " << pc_path << "\n";
		}
		return;
	}

	if (restartRefineFactor_ > 1) {
		// Save current geometry for all levels
		amrex::Vector<amrex::Geometry> current_geom(finest_level + 1);
		amrex::Vector<amrex::BoxArray> current_ba(finest_level + 1);
		amrex::Vector<amrex::DistributionMapping> current_dm(finest_level + 1);

		for (int lev = 0; lev <= finest_level; ++lev) {
			current_geom[lev] = particles->Geom(lev);
			current_ba[lev] = particles->ParticleBoxArray(lev);
			current_dm[lev] = particles->ParticleDistributionMap(lev);
		}

		// Set up original (coarse) geometry and boxArrays for particle reading
		for (int lev = 0; lev <= finest_level; ++lev) {
			// Get coarse BoxArray from header box arrays
			amrex::BoxArray coarse_ba = header_box_arrays[lev];
			amrex::DistributionMapping const coarse_dm{coarse_ba, amrex::ParallelDescriptor::NProcs()};
			amrex::Geometry coarse_geom_lev = amrex::coarsen(current_geom[lev], restartRefineFactor_);
			particles->SetParticleGeometry(lev, coarse_geom_lev);
			particles->SetParticleBoxArray(lev, coarse_ba);
			particles->SetParticleDistributionMap(lev, coarse_dm);
		}

		// Read particles with coarse grid structure
		particles->resizeData(); // Ensure particle container internal structures are properly sized

		// WORKAROUND for AMReX bug: If the particle container has more levels than the
		// checkpoint file, Restart() will crash due to out-of-bounds access in the
		// old_dms vector. We need to temporarily reduce the number of levels.
		const int pc_finest_level = particles->finestLevel();
		if (pc_finest_level > finest_level) {
			amrex::Abort("ERROR: Particle container has more levels than checkpoint. "
				     "This is not currently supported due to an AMReX limitation.");
		}

		particles->Restart(restart_chkfile, particle_type_name);

		// Restore refined geometry for all levels
		for (int lev = 0; lev <= finest_level; ++lev) {
			particles->SetParticleGeometry(lev, current_geom[lev]);
			particles->SetParticleBoxArray(lev, current_ba[lev]);
			particles->SetParticleDistributionMap(lev, current_dm[lev]);
		}

		// Redistribute particles to refined grid
		particles->Redistribute();
	} else {
		// Normal restart without refinement
		particles->Restart(restart_chkfile, particle_type_name);
	}
}

template <typename problem_t>
template <typename ContainerType>
void AMRSimulation<problem_t>::initializeParticleContainerFromCheckpoint(std::unique_ptr<ContainerType> &container, quokka::ParticleType particle_type,
									 amrex::Vector<amrex::BoxArray> const &header_box_arrays)
{
	AMREX_ASSERT(container == nullptr);
	container = std::make_unique<ContainerType>(this);

	// Register container
	particleRegister_.registerParticleType(container.get(), particle_type);

	// Read particles
	restartParticleContainerWithRefinement(container, restart_chkfile, particleRegister_.getParticleTypeName(particle_type), header_box_arrays);

	// Split particles
#if AMREX_SPACEDIM == 3
	if (restartRefineFactor_ > 1) {
		const int split_factor = gcem::pow(restartRefineFactor_, AMREX_SPACEDIM);
		amrex::Print() << fmt::format("Splitting {} using split_factor = {}\n", particleRegister_.getParticleTypeName(particle_type), split_factor);
		auto descriptor = particleRegister_.getParticleDescriptor(particle_type);
		for (int lev = 0; lev <= finestLevel(); ++lev) {
			descriptor->splitParticles(lev, split_factor);
		}
	}
#endif
}

template <typename problem_t>
void AMRSimulation<problem_t>::writeFaceVelocitiesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArrays, int lev, int timestep)
{
	// Create directory for face velocity outputs if it doesn't exist
	std::string dirname = fmt::format("facevel_lev{}_step{}", lev, timestep);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::UtilCreateDirectory(dirname, 0755);
	}
	amrex::ParallelDescriptor::Barrier();

	// Write each direction's face velocities
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		std::string dimname;
		if (idim == 0) {
			dimname = "x";
		} else if (idim == 1) {
			dimname = "y";
		} else {
			dimname = "z";
		}

		// Write each FAB in the MultiFab
		for (amrex::MFIter mfi(faceVelArrays[idim]); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.fabbox(); // This includes ghost cells
			const amrex::FArrayBox &fab = faceVelArrays[idim][mfi];

			// Create filename for this FAB
			const std::string filename = fmt::format("{}/facevel_{}_box_{}.fab", dirname, dimname, mfi.index());

			// Write FAB to disk in ASCII format
			std::ofstream ofs(filename, std::ios::out);
			if (ofs.is_open()) {
				// Write box information
				ofs << "# Face velocity FAB for direction " << dimname << "\n";
				ofs << "# Box: " << bx << "\n";
				ofs << "# Valid box: " << mfi.validbox() << "\n";
				ofs << "# MultiFab ghost cells: " << faceVelArrays[idim].nGrow() << "\n";
#if AMREX_SPACEDIM == 1
				ofs << "# Format: i value\n";
#elif AMREX_SPACEDIM == 2
				ofs << "# Format: i j value\n";
#elif AMREX_SPACEDIM == 3
				ofs << "# Format: i j k value\n";
#endif

				// Write data
				auto const &arr = fab.const_array();
				amrex::Loop(bx, [&](int i, int j, int k) {
#if AMREX_SPACEDIM == 1
					ofs << i << " " << arr(i, j, k, 0) << "\n";
#elif AMREX_SPACEDIM == 2
					ofs << i << " " << j << " " << arr(i,j,k,0) << "\n";
#elif AMREX_SPACEDIM == 3
					ofs << i << " " << j << " " << k << " " << arr(i,j,k,0) << "\n";
#endif
				});
				ofs.close();
			}
		}
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::writeReconstructedStatesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &leftState,
							      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &rightState, int lev, int timestep)
{
	// Create directory for reconstructed state outputs if it doesn't exist
	std::string dirname = fmt::format("reconst_lev{}_step{}", lev, timestep);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::UtilCreateDirectory(dirname, 0755);
	}
	amrex::ParallelDescriptor::Barrier();

	// Write each direction's reconstructed states
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		std::string dimname;
		if (idim == 0) {
			dimname = "x";
		} else if (idim == 1) {
			dimname = "y";
		} else {
			dimname = "z";
		}

		// Write left and right states for each FAB in the MultiFab
		for (amrex::MFIter mfi(leftState[idim]); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.fabbox(); // This includes ghost cells
			const amrex::FArrayBox &leftFab = leftState[idim][mfi];
			const amrex::FArrayBox &rightFab = rightState[idim][mfi];

			// Create filenames for this FAB's left and right states
			const std::string leftFilename = fmt::format("{}/reconst_left_{}_box_{}.fab", dirname, dimname, mfi.index());
			const std::string rightFilename = fmt::format("{}/reconst_right_{}_box_{}.fab", dirname, dimname, mfi.index());

			// Write left state FAB to disk
			std::ofstream leftOfs(leftFilename, std::ios::out);
			if (leftOfs.is_open()) {
				// Write box information
				leftOfs << "# Left reconstructed state FAB for direction " << dimname << "\n";
				leftOfs << "# Box: " << bx << "\n";
				leftOfs << "# Valid box: " << mfi.validbox() << "\n";
				leftOfs << "# MultiFab ghost cells: " << leftState[idim].nGrow() << "\n";
#if AMREX_SPACEDIM == 1
				leftOfs << "# Format: i density xmom ymom zmom energy intenergy\n";
#elif AMREX_SPACEDIM == 2
				leftOfs << "# Format: i j density xmom ymom zmom energy intenergy\n";
#elif AMREX_SPACEDIM == 3
				leftOfs << "# Format: i j k density xmom ymom zmom energy intenergy\n";
#endif

				// Write data (all hydro variables)
				auto const &leftArr = leftFab.const_array();
				amrex::Loop(bx, [&](int i, int j, int k) {
#if AMREX_SPACEDIM == 1
					leftOfs << i;
#elif AMREX_SPACEDIM == 2
					leftOfs << i << " " << j;
#elif AMREX_SPACEDIM == 3
					leftOfs << i << " " << j << " " << k;
#endif
					// Write all hydro variables (density, xmom, ymom, zmom, energy, intenergy)
					for (int ivar = 0; ivar < leftFab.nComp(); ++ivar) {
						leftOfs << " " << leftArr(i, j, k, ivar);
					}
					leftOfs << "\n";
				});
				leftOfs.close();
			}

			// Write right state FAB to disk
			std::ofstream rightOfs(rightFilename, std::ios::out);
			if (rightOfs.is_open()) {
				// Write box information
				rightOfs << "# Right reconstructed state FAB for direction " << dimname << "\n";
				rightOfs << "# Box: " << bx << "\n";
				rightOfs << "# Valid box: " << mfi.validbox() << "\n";
				rightOfs << "# MultiFab ghost cells: " << rightState[idim].nGrow() << "\n";
#if AMREX_SPACEDIM == 1
				rightOfs << "# Format: i density xmom ymom zmom energy intenergy\n";
#elif AMREX_SPACEDIM == 2
				rightOfs << "# Format: i j density xmom ymom zmom energy intenergy\n";
#elif AMREX_SPACEDIM == 3
				rightOfs << "# Format: i j k density xmom ymom zmom energy intenergy\n";
#endif

				// Write data (all hydro variables)
				auto const &rightArr = rightFab.const_array();
				amrex::Loop(bx, [&](int i, int j, int k) {
#if AMREX_SPACEDIM == 1
					rightOfs << i;
#elif AMREX_SPACEDIM == 2
					rightOfs << i << " " << j;
#elif AMREX_SPACEDIM == 3
					rightOfs << i << " " << j << " " << k;
#endif
					// Write all hydro variables (density, xmom, ymom, zmom, energy, intenergy)
					for (int ivar = 0; ivar < rightFab.nComp(); ++ivar) {
						rightOfs << " " << rightArr(i, j, k, ivar);
					}
					rightOfs << "\n";
				});
				rightOfs.close();
			}
		}
	}
}

#endif // SIMULATION_HPP_
