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
#include "AMReX_YAFluxRegister.H"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <yaml-cpp/yaml.h>

#ifdef AMREX_PARTICLES
#include "AMReX_AmrParticles.H"
#include "particles/PhysicsParticles.hpp"
#endif

#if AMREX_SPACEDIM == 3
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
#include "io/projection.hpp"
#include "physics_info.hpp"

#ifdef QUOKKA_USE_OPENPMD
#include "io/openPMD.hpp"
#endif

#define USE_YAFLUXREGISTER

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
	amrex::Long cycleCount_ = 0;
	int printCycleTiming_ = 0;				     // default: don't print
	amrex::Long maxTimesteps_ = std::numeric_limits<int>::max(); // default: no limit
	amrex::Long maxWalltime_ = 0;				     // default: no limit
	int ascentInterval_ = -1;				     // -1 == no in-situ renders with Ascent
	int plotfileInterval_ = -1;				     // -1 == no output
	int projectionInterval_ = -1;				     // -1 == no output
	int statisticsInterval_ = -1;				     // -1 == no output
	amrex::Real plotTimeInterval_ = -1.0;			     // time interval for plt file
	amrex::Real checkpointTimeInterval_ = -1.0;		     // time interval for checkpoints
	int checkpointInterval_ = -1;				     // -1 == no output
	int amrInterpMethod_ = 1;				     // 0 == piecewise constant, 1 == lincc_interp
	amrex::Real reltolPoisson_ = 1.0e-5;			     // default
	amrex::Real abstolPoisson_ = 1.0e-5;			     // default (scaled by minimum RHS value)
	int doPoissonSolve_ = 0;				     // 1 == self-gravity enabled, 0 == disabled
	int poissonSupercycleInterval_ = 1;			     // number of coarse steps between Poisson solves (default: 1)
	amrex::Vector<amrex::MultiFab> phi;

	amrex::Real densityFloor_ = 0.0; // default
	amrex::Real tempFloor_ = 0.0;	 // default

	YAML::Node simulationMetadata_;

	// constructor
	explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : BCs_cc_(BCs_cc), BCs_fc_(BCs_fc) { initialize(); }

	explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : BCs_cc_(BCs_cc), BCs_fc_(builtin_BCs_fc(BCs_cc)) { initialize(); }

	auto builtin_BCs_fc(amrex::Vector<amrex::BCRec> &BCs_cc) -> amrex::Vector<amrex::BCRec>
	{
		amrex::Vector<amrex::BCRec> BCs_fc(Physics_Indices<problem_t>::nvarPerDim_fc);

		if (Physics_Traits<problem_t>::is_hydro_enabled) {
			AMREX_ALWAYS_ASSERT(Physics_Indices<problem_t>::nvarPerDim_fc == 1);
			// set boundary conditions for face velocities (used ONLY for tracer particles)
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				// lower boundary
				if (BCs_cc[Physics_Indices<problem_t>::hydroFirstIndex].lo(i) == amrex::BCType::int_dir) {
					BCs_fc[Physics_Indices<problem_t>::velFirstIndex].setLo(i, amrex::BCType::int_dir);
				} else {
					BCs_fc[Physics_Indices<problem_t>::velFirstIndex].setLo(i, amrex::BCType::foextrap);
				}
				// upper boundary
				if (BCs_cc[Physics_Indices<problem_t>::hydroFirstIndex].hi(i) == amrex::BCType::int_dir) {
					BCs_fc[Physics_Indices<problem_t>::velFirstIndex].setHi(i, amrex::BCType::int_dir);
				} else {
					BCs_fc[Physics_Indices<problem_t>::velFirstIndex].setHi(i, amrex::BCType::foextrap);
				}
			}
		}
		return BCs_fc;
	}

	void initialize();
	void PerformanceHints();
	void readParameters();
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
			       int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering &cen, FillPatchType fptype,
			       PreInterpHook const &pre_interp, PostInterpHook const &post_interp);

	static void InterpHookNone(amrex::MultiFab &mf, int scomp, int ncomp);
	virtual void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
			       FillPatchType fptype);

	auto getAmrInterpolaterCellCentered() -> amrex::MFInterpolater *;
	auto getAmrInterpolaterFaceCentered() -> amrex::Interpolater *;
	void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering cen,
			     quokka::direction dir);
	void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data, amrex::Vector<amrex::Real> &datatime, quokka::centering cen,
		     quokka::direction dir);
	void AverageDown();
	void AverageDownTo(int crse_lev);
	void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);
	void calculateGpotAllLevels();
	void gravAccelAllLevels(amrex::Real dt);
	void ellipticSolveAllLevels(amrex::Real dt);

	void incrementFluxRegisters(amrex::MFIter &mfi, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
				    std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxArrays, int lev, amrex::Real dt_lev);

	void incrementFluxRegisters(amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
				    std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays, int lev, amrex::Real dt_lev);

	void particleMeshInteraction(amrex::Real time, amrex::Real dt);

	// boundary condition
	AMREX_GPU_DEVICE static void setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
								 amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr, int bcomp,
								 int orig_comp); // template specialized by problem generator

	// boundary condition
	AMREX_GPU_DEVICE static void setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp,
									int numcomp, amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr,
									int bcomp,
									int orig_comp); // template specialized by problem generator

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
	void WriteProjectionPlotfile() const;
	void WriteCheckpointFile() const;
	void SetLastCheckpointSymlink(std::string const &checkpointname) const;
	void ReadCheckpointFile();
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
	amrex::Vector<std::unique_ptr<amrex::YAFluxRegister>> flux_reg_;

	// This is for fillpatch during timestepping, but not for regridding.
	amrex::Vector<std::unique_ptr<amrex::FillPatcher<amrex::MultiFab>>> fillpatcher_;

	// Nghost = number of ghost cells for each array
	int nghost_cc_ = 4;						    // PPM needs nghost >= 3, PPM+flattening needs nghost >= 4
	int nghost_fc_ = Physics_Traits<problem_t>::is_mhd_enabled ? 4 : 2; // 4 needed for MHD, otherwise only 2 for tracer particles
	amrex::Vector<std::string> componentNames_cc_;
	amrex::Vector<std::string> componentNames_fc_flat_;
	std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> componentNames_fc_;
	amrex::Vector<std::string> derivedNames_;
	bool areInitialConditionsDefined_ = false;

	/// output parameters
	std::string plot_file{"plt"};	       // plotfile prefix
	std::string chk_file{"chk"};	       // checkpoint prefix
	std::string stats_file{"history.txt"}; // statistics filename
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
#ifdef AMREX_PARTICLES
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
#endif

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
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		state_new_fc_.resize(nlevs_max);
		state_old_fc_.resize(nlevs_max);
	}
	max_signal_speed_.resize(nlevs_max);
	flux_reg_.resize(nlevs_max + 1);
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

	// Default ascent render interval
	pp.query("ascent_interval", ascentInterval_);

	// Default output interval
	pp.query("plotfile_interval", plotfileInterval_);

	// Default projection interval
	pp.query("projection_interval", projectionInterval_);

	// Default statistics interval
	pp.query("statistics_interval", statisticsInterval_);

	// Default Time interval
	pp.query("plottime_interval", plotTimeInterval_);

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
}

template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditions()
{
	BL_PROFILE("AMRSimulation::setInitialConditions()"); // NOLINT(misc-const-correctness)

	if (restart_chkfile.empty()) {
		// start simulation from the beginning
		const amrex::Real time = 0.0;
		InitFromScratch(time);
		AverageDown();

#ifdef AMREX_PARTICLES
		if (do_tracers != 0) {
			InitParticles();
		}

		InitPhyParticles();
#endif

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

	if (plotfileInterval_ > 0 || plotTimeInterval_ > 0) {
		WritePlotFile();
	}

	if (projectionInterval_ > 0) {
		WriteProjectionPlotfile();
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

	amrex::Real cur_time = tNew_[0];
#ifdef AMREX_USE_ASCENT
	int last_ascent_step = 0;
#endif
	int last_projection_step = 0;
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

		// do user-specified calculations before the level update
		computeBeforeTimestep();

#if AMREX_SPACEDIM == 3
		// do particle leapfrog (first kick at time t)
		if (doPoissonSolve_ != 0) {
			kickParticlesAllLevels(dt_[0]);
		}
#endif

		// hyperbolic advance over all levels
		// (N.B. when AMR is enabled, regridding may happen during this function!)
		// Particle redistribution is done here.
		int lev = 0;		 // coarsest level
		const int iteration = 1; // this is the first call to advance level 'lev'
		timeStepWithSubcycling(lev, cur_time, iteration);

#if AMREX_SPACEDIM == 3
		// drift particles from t to (t + dt)
		// N.B.: MUST be done *before* Poisson solve at new time!
		particleRegister_.driftParticlesAllLevels(dt_[0], finest_level);
#endif

		// elliptic solve over entire AMR grid (post-timestep)
		ellipticSolveAllLevels(dt_[0]);

		// do particle leapfrog (second kick at t + dt)
#if AMREX_SPACEDIM == 3
		if (doPoissonSolve_ != 0) {
			kickParticlesAllLevels(dt_[0]);
		}

		// Stellar evolution and SN deposition; only apply to star particles
		if (particleRegister_.HasStarParticles()) {
			// TODO(cch): Need to take care of AMR subcycling
			particleMeshInteraction(cur_time, dt_[0]);
		}

		// Use the new type-aware particle destruction method
		// TODO(cch): Need to take care of AMR subcycling
		particleRegister_.destroyParticles(0, cur_time, dt_[0]);
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

		if (projectionInterval_ > 0 && (step + 1) % projectionInterval_ == 0) {
			last_projection_step = step + 1;
			WriteProjectionPlotfile();
		}

		// print particle statistics
		if (quokka::particle_verbose > 0) {
			particleRegister_.printParticleStatistics();
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

		if (maxWalltime_ > 0 && getWalltime() > 0.9 * maxWalltime_) {
			// we have exceeded 90% of maxWalltime_
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

	// write final projection
	if (projectionInterval_ > 0 && istep[0] > last_projection_step) {
		WriteProjectionPlotfile();
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
	if (doPoissonSolve_ != 0) {
		if (do_subcycle == 1) { // not supported
			amrex::Abort("Poisson solve is not support when AMR subcycling is enabled! You must set do_subcycle = 0.");
		}

		BL_PROFILE_REGION("GravitySolver"); // NOLINT(misc-const-correctness)

		// set up elliptic solve object
		amrex::OpenBCSolver poissonSolver(Geom(0, finest_level), boxArray(0, finest_level), DistributionMap(0, finest_level));
		if (verbose) {
			poissonSolver.setVerbose(1);
			poissonSolver.setBottomVerbose(0);
			amrex::Print() << "Doing Poisson solve...\n\n";
		}

		phi.resize(finest_level + 1);
		// solve Poisson equation with open b.c. using the method of James (1977)
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

#ifdef AMREX_PARTICLES
		// deposit particle mass from all particles that have mass into rhs by accumulation
		particleRegister_.depositMass(amrex::GetVecOfPtrs(rhs), finest_level, Gconst_);
#endif

		// check for NaN
		for (int lev = 0; lev <= finest_level; ++lev) {
			AMREX_ALWAYS_ASSERT(!rhs[lev].contains_nan());
		}

		amrex::Real abstol = abstolPoisson_ * rhs_min;
		poissonSolver.solve(amrex::GetVecOfPtrs(phi), amrex::GetVecOfConstPtrs(rhs), reltolPoisson_, abstol);
		if (verbose) {
			amrex::Print() << "\n";
		}

		// check for NaN
		for (int lev = 0; lev <= finest_level; ++lev) {
			AMREX_ALWAYS_ASSERT(!phi[lev].contains_nan()); // NOTE: this fails when multiple levels are fully refined
		}
	}
#endif
}

template <typename problem_t> void AMRSimulation<problem_t>::gravAccelAllLevels(const amrex::Real dt)
{
#if AMREX_SPACEDIM == 3
	if (doPoissonSolve_ != 0) {

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
	if (doPoissonSolve_ != 0) {
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
	// kick particles (do: vel[i] += 0.5 * dt * accel[i])

	// skip if there are no massive particles
	if (!particleRegister_.HasMassiveParticles()) {
		return;
	}

	// Create acceleration MultiFabs for each level
	amrex::Vector<amrex::Vector<amrex::MultiFab>> accel(finest_level + 1);

	// set boundary conditions for accelerations
	amrex::Vector<amrex::BCRec> accelBC(1);
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		// lower boundary
		if (BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].lo(i) == amrex::BCType::int_dir) {
			accelBC[0].setLo(i, amrex::BCType::int_dir);
		} else {
			accelBC[0].setLo(i, amrex::BCType::foextrap);
		}
		// upper boundary
		if (BCs_cc_[Physics_Indices<problem_t>::hydroFirstIndex].hi(i) == amrex::BCType::int_dir) {
			accelBC[0].setHi(i, amrex::BCType::int_dir);
		} else {
			accelBC[0].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Compute accelerations and kick particles
	for (int lev = 0; lev <= finest_level; ++lev) {
		// NOTE: CIC interpolation requires 1, but particles may have drifted
		// 	into 1 ghost cell since last particle redistribute.
		const int nghost_acc = 2;

		accel[lev].resize(AMREX_SPACEDIM);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_face = amrex::convert(boxArray(lev), amrex::IntVect::TheDimensionVector(idim));
			accel[lev][idim].define(ba_face, DistributionMap(lev), 1, nghost_acc);
			accel[lev][idim].setVal(0.);
		}

		// Fill ghosts at coarse-fine boundary
		// (This *also* fills valid cells, so we have to do it before computing the valid cell accelerations.)
		if (lev > 0) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});
				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> fineBdryFunct(geom[lev], accelBC, boundaryFunctor);
				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> coarseBdryFunct(geom[lev - 1], accelBC, boundaryFunctor);
				amrex::InterpFromCoarseLevel(accel[lev][idim], 0., accel[lev - 1][idim], 0, 0, 1, geom[lev - 1], geom[lev], coarseBdryFunct, 0,
							     fineBdryFunct, 0, refRatio(lev - 1), &amrex::face_linear_interp, accelBC, 0);
				// deallocate coarse MF
				accel[lev - 1][idim].clear();
			}
		}

		// Fill valid cells
		const auto &phi_arr = phi[lev].const_arrays();
		const auto dx_inv = geom[lev].InvCellSizeArray();
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto accel_arr = accel[lev][idim].arrays();
			amrex::ParallelFor(accel[lev][idim], amrex::IntVect{0}, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
				// compute face-centered acceleration -grad(phi)
				if (idim == 0) {
					accel_arr[bx](i, j, k) = -dx_inv[0] * (phi_arr[bx](i, j, k) - phi_arr[bx](i - 1, j, k));
				}
				if (idim == 1) {
					accel_arr[bx](i, j, k) = -dx_inv[1] * (phi_arr[bx](i, j, k) - phi_arr[bx](i, j - 1, k));
				}
				if (idim == 2) {
					accel_arr[bx](i, j, k) = -dx_inv[2] * (phi_arr[bx](i, j, k) - phi_arr[bx](i, j, k - 1));
				}
			});
		}
		amrex::Gpu::streamSynchronizeAll();

		// Fill remaining ghost cells
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			// internal boundaries
			accel[lev][idim].FillBoundary(geom[lev].periodicity());

			// physical boundaries
			amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});
			amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> fineBdryFunct(geom[lev], accelBC, boundaryFunctor);
			fineBdryFunct(accel[lev][idim], 0, accel[lev][idim].nComp(), accel[lev][idim].nGrowVect(), 0., 0);

			// check for NaN
			AMREX_ALWAYS_ASSERT(!accel[lev][idim].contains_nan());
		}

		// average to cell centers
		amrex::MultiFab accel_cc(boxArray(lev), DistributionMap(lev), AMREX_SPACEDIM, nghost_acc);
		amrex::average_face_to_cellcenter(accel_cc, 0, amrex::GetVecOfConstPtrs(accel[lev]), nghost_acc);

		// check for NaN
		AMREX_ALWAYS_ASSERT(!accel_cc.contains_nan());

		// Kick particles using the acceleration field
		particleRegister_.kickParticlesAtLevel(lev, dt, accel_cc);
	}
}

template <typename problem_t> void AMRSimulation<problem_t>::particleMeshInteraction(amrex::Real time, amrex::Real dt)
{
	// Need 4 ghost cells for SN deposition with a SNR radius of 3 dx.
	const int nghost = 4;

	// Assume all SN progenitors are at the finest level
	const int lev = finest_level;

	// Fill boundary before computing accretion rate. This is necessary because the computation of accretion rate uses 3 ghost cells, but the
	// boundaries are not filled at this point.
	// TODO(cch): we can fill the hydro variables only if no other variables are used in accretion rate computation
	state_new_cc_[lev].FillBoundary(geom[lev].periodicity());

	// Create a MultiFab to hold the change of states (density, 3 x momentum, internal energy, energy) during particle-mesh interaction
	amrex::MultiFab accretion_rate_at_level(grids[lev], dmap[lev], Physics_NumVars::numHydroVars, nghost);

	accretion_rate_at_level.setVal(0.0);

	// Sink accretion, stage 1: compute the accretion rate
	particleRegister_.computeSinkAccretion(state_new_cc_[lev], accretion_rate_at_level, lev, time, dt);

	// Sink accretion, stage 2: update the particle states
	particleRegister_.applySinkAccretion(state_new_cc_[lev], accretion_rate_at_level, geom[lev], lev, time, dt);

	// We allow particle formation at the finest level only to avoid duplicate particle creation from multiple levels at the same location. 
	// particleRegister_.createParticlesFromState(state_new_cc_[lev], accretion_rate_at_level, lev, time, dt);

	// Deposit the SN particles into the MultiFab
	// TODO(cch): put accretion_rate_at_level inside depositSN
	particleRegister_.depositSN(state_new_cc_[lev], accretion_rate_at_level, lev, time, dt);
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

#ifdef AMREX_PARTICLES
				// redistribute particles
				if (do_tracers != 0) {
					TracerPC->Redistribute(lev);
				}

				// redistribute all particles in particleRegister_
				particleRegister_.redistribute(lev);
#endif

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
			// update lev based on coarse-fine flux mismatch
			flux_reg_[lev + 1]->Reflux(state_new_cc_[lev]);
		}

		AverageDownTo(lev); // average lev+1 down to lev
		FixupState(lev);    // fix any unphysical states created by reflux or averaging

		fillpatcher_[lev + 1].reset(); // because the data on lev have changed.
	}

#ifdef AMREX_PARTICLES
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
#endif
}

template <typename problem_t>
void AMRSimulation<problem_t>::incrementFluxRegisters(amrex::MFIter &mfi, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
						      std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxArrays, int const lev, amrex::Real const dt_lev)
{
	BL_PROFILE("AMRSimulation::incrementFluxRegisters()"); // NOLINT(misc-const-correctness)

	if (fr_as_crse != nullptr) {
		AMREX_ASSERT(lev < finestLevel());
		AMREX_ASSERT(fr_as_crse == flux_reg_[lev + 1].get());
		fr_as_crse->CrseAdd(mfi, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])}, // NOLINT(readability-container-data-pointer)
				    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
	}

	if (fr_as_fine != nullptr) {
		AMREX_ASSERT(lev > 0);
		AMREX_ASSERT(fr_as_fine == flux_reg_[lev].get());
		fr_as_fine->FineAdd(mfi, {AMREX_D_DECL(&fluxArrays[0], &fluxArrays[1], &fluxArrays[2])}, // NOLINT(readability-container-data-pointer)
				    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
	}
}

template <typename problem_t>
void AMRSimulation<problem_t>::incrementFluxRegisters(amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays, int const lev, amrex::Real const dt_lev)
{
	BL_PROFILE("AMRSimulation::incrementFluxRegisters()"); // NOLINT(misc-const-correctness)

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		if (fr_as_crse != nullptr) {
			AMREX_ASSERT(lev < finestLevel());
			AMREX_ASSERT(fr_as_crse == flux_reg_[lev + 1].get());
			fr_as_crse->CrseAdd(mfi, {AMREX_D_DECL(fluxArrays[0].fabPtr(mfi), fluxArrays[1].fabPtr(mfi), fluxArrays[2].fabPtr(mfi))},
					    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
		}

		if (fr_as_fine != nullptr) {
			AMREX_ASSERT(lev > 0);
			AMREX_ASSERT(fr_as_fine == flux_reg_[lev].get());
			fr_as_fine->FineAdd(mfi, {AMREX_D_DECL(fluxArrays[0].fabPtr(mfi), fluxArrays[1].fabPtr(mfi), fluxArrays[2].fabPtr(mfi))},
					    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
		}
	}
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
	// TODO(bwibking): this must be changed to amrex::face_divfree_interp for magnetic fields!
	// TODO(neco): implement fc interpolator
	amrex::Interpolater *mapper = &amrex::face_linear_interp;
	return mapper; // global object, so this is ok
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
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
									   Geom(level - 1), refRatio(level - 1), level, ncomp_cc);
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
			FillCoarsePatch(level, time, state_new_fc_[level][idim], 0, ncomp_per_dim_fc, BCs_fc_, quokka::centering::fc,
					static_cast<quokka::direction>(idim));
			FillCoarsePatch(level, time, state_old_fc_[level][idim], 0, ncomp_per_dim_fc, BCs_fc_, quokka::centering::fc,
					static_cast<quokka::direction>(idim)); // also necessary
		}
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
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
									   Geom(level - 1), refRatio(level - 1), level, ncomp_cc);
	}

	// face-centred
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		const int ncomp_per_dim_fc = state_new_fc_[level][0].nComp();
		const int nghost_fc = state_new_fc_[level][0].nGrow();
		amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> int_state_new_fc;
		amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> int_state_old_fc;
		// define
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			int_state_new_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
			int_state_old_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
		}
		// TODO(neco): fillPatchFC
		// swap
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
	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
					 amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp,
					 const int &orig_comp) const
	{
		AMRSimulation<problem_t>::setCustomBoundaryConditionsFaceVar(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp);
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
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_cc_, cen, fptype, InterpHookNone, InterpHookNone);
	} else if (cen == quokka::centering::fc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_fc_, cen, fptype, InterpHookNone, InterpHookNone);
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
		flux_reg_[level] = std::make_unique<amrex::YAFluxRegister>(ba, boxArray(level - 1), dm, DistributionMap(level - 1), Geom(level),
									   Geom(level - 1), refRatio(level - 1), level, ncomp_cc);
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

		FillPatchWithData(lev, time, S_filled, coarseData, coarseTime, fineData, fineTime, 0, S_filled.nComp(), BCs, cen, fptype, pre_interp,
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
				// create face-centered boundary functor
				amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor =
				    amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>{setBoundaryFunctorFaceVar<problem_t>{}};
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
						 quokka::centering &cen, FillPatchType fptype, PreInterpHook const &pre_interp,
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

	amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor_fc(setBoundaryFunctorFaceVar<problem_t>{});
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
		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
					     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), getAmrInterpolaterFaceCentered(), BCs, 0);
	} else {
		amrex::Abort("AMR interpolation is not implemented for this zone centering!");
	}
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

#ifdef AMREX_PARTICLES
template <typename problem_t> void AMRSimulation<problem_t>::InitParticles()
{
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

		// Create particle container
		StochasticStellarPopParticles = std::make_unique<quokka::StochasticStellarPopParticleContainer<problem_t>>(this);
		StochasticStellarPopParticles->SetVerbose(0);

		// Register with particle register - StochasticStellarPop particles allow creation
		particleRegister_.registerStarParticleType(StochasticStellarPopParticles.get(), quokka::ParticleType::StochasticStellarPop);

		// Initialize particles through user-defined function
		createInitialStochasticStellarPopParticles();
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Sink) {
		AMREX_ASSERT(SinkParticles == nullptr);

		// Create particle container
		SinkParticles = std::make_unique<quokka::SinkParticleContainer>(this);
		SinkParticles->SetVerbose(0);

		// Register with particle register - Sink particles allow creation
		particleRegister_.registerStarParticleType(SinkParticles.get(), quokka::ParticleType::Sink);

		// Initialize particles through user-defined function
		createInitialSinkParticles();
	}
	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Test) {
		AMREX_ASSERT(TestParticles == nullptr);

		// Create particle container
		TestParticles = std::make_unique<quokka::TestParticleContainer<problem_t>>(this);
		TestParticles->SetVerbose(0);

		// Register with particle register - Test particles have all features enabled
		particleRegister_.registerStarParticleType(TestParticles.get(), quokka::ParticleType::Test);

		// Initialize particles through user-defined function
		createInitialTestParticles();
	}
#endif // AMREX_SPACEDIM == 3

	particleRegister_.redistribute(0);
}
#endif

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
	// Combine state_new_cc_[lev] and derived variables in a new MF
	const int ncomp_cc = state_new_cc_[lev].nComp();
	int comp = 0;
	int ncomp_per_dim_fc = 0;
	int ncomp_tot_fc = 0;
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		ncomp_per_dim_fc = Physics_Indices<problem_t>::nvarPerDim_fc;
		ncomp_tot_fc = Physics_Indices<problem_t>::nvarTotal_fc;
	}
	const int ncomp_deriv = derivedNames_.size();
	const int ncomp_plotMF = ncomp_cc + ncomp_tot_fc + ncomp_deriv;
	amrex::MultiFab plotMF(grids[lev], dmap[lev], ncomp_plotMF, included_ghosts);

	if (included_ghosts > 0) {
		// Fill ghost zones for state_new_cc_
		fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, tNew_[lev], quokka::centering::cc, quokka::direction::na, InterpHookNone,
				       InterpHookNone, FillPatchType::fillpatch_function);
	}

	// Fill ghost zones for state_new_fc_
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, tNew_[lev], quokka::centering::fc,
					       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
		}
	}

	// copy data from cell-centred state variables
	for (int i = 0; i < ncomp_cc; i++) {
		amrex::MultiFab::Copy(plotMF, state_new_cc_[lev], i, comp, 1, included_ghosts);
		comp++;
	}

	// compute cell-center averaged face-centred data
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			AverageFCToCC(plotMF, state_new_fc_[lev][idim], idim, comp, 0, ncomp_per_dim_fc);
			comp += ncomp_per_dim_fc;
		}
	}

	// compute derived vars
	for (auto const &dname : derivedNames_) {
		ComputeDerivedVar(lev, dname, plotMF, comp);
		comp++;
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

	amrex::MultiFab plotMF_fc(grids[lev], dmap[lev], ncomp_plotMF_fc, nghost_fc_);

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
			diag->processDiag(istep[0], tNew_[0], GetVecOfConstPtrs(diagMFVec), m_diagVars, simulationMetadata_);
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

template <typename problem_t> auto AMRSimulation<problem_t>::GetPlotfileVarNames() const -> amrex::Vector<std::string>
{
	amrex::Vector<std::string> varnames;
	varnames.insert(varnames.end(), componentNames_cc_.begin(), componentNames_cc_.end());
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		for (int icomp = 0; icomp < Physics_Indices<problem_t>::nvarTotal_fc; ++icomp) {
			varnames.push_back(componentNames_fc_flat_[icomp]);
		}
	}
	varnames.insert(varnames.end(), derivedNames_.begin(), derivedNames_.end());
	return varnames;
}

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
	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf_cc_ptr, varnames, Geom(), tNew_[0], istep, refRatio());
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
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

#ifdef AMREX_PARTICLES
	// write particles
	if (do_tracers != 0) {
		TracerPC->WritePlotFile(plotfilename, "tracer_particles");
	}

	// write all particles in particleRegister_ to plotfile
	particleRegister_.writePlotFile(plotfilename);
#endif // AMREX_PARTICLES
#endif
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

template <typename problem_t> void AMRSimulation<problem_t>::WriteProjectionPlotfile() const
{
	std::vector<std::string> dirs{};
	const amrex::ParmParse pp;
	pp.queryarr("projection.dirs", dirs);

	auto dir_from_string = [=](const std::string &dir_str) {
		if (dir_str == "x") {
			return amrex::Direction::x;
		}
#if AMREX_SPACEDIM >= 2
		if (dir_str == "y") {
			return amrex::Direction::y;
		}
#endif
#if AMREX_SPACEDIM == 3
		if (dir_str == "z") {
			return amrex::Direction::z;
		}
#endif
		amrex::Error("invalid direction for projection!");
		return amrex::Direction::x;
	};

	for (auto &dir_str : dirs) {
		// compute projections along axis 'dir'
		amrex::Direction dir = dir_from_string(dir_str);
		std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj = ComputeProjections(dir);
		quokka::diagnostics::WriteProjection(dir, proj, tNew_[0], istep[0]);
	}
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
#ifdef AMREX_PARTICLES
	if (do_tracers != 0) {
		TracerPC->Checkpoint(checkpointname, "tracer_particles", true);
	}

	// write all particles in particleRegister_ to checkpoint file
	particleRegister_.writeCheckpoint(checkpointname, true);
#endif

	// create symlink and point it at this checkpoint dir
	SetLastCheckpointSymlink(checkpointname);
}

// utility to skip to next line in Header
inline void GotoNextLine(std::istream &is)
{
	constexpr std::streamsize bl_ignore_max{100000};
	is.ignore(bl_ignore_max, '\n');
}

template <typename problem_t> void AMRSimulation<problem_t>::ReadCheckpointFile()
{
	BL_PROFILE("AMRSimulation::ReadCheckpointFile()"); // NOLINT(misc-const-correctness)

	amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

	// Header
	std::string File(restart_chkfile + "/Header");

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

	for (int lev = 0; lev <= finest_level; ++lev) {
		// read in level 'lev' BoxArray from Header
		amrex::BoxArray ba;
		ba.readFrom(is);
		GotoNextLine(is);

		/*Create New BoxArray at Level 0 for optimum load distribution*/
		if (lev == 0) {
			amrex::IntVect fac(2);
			const amrex::IntVect domlo{AMREX_D_DECL(0, 0, 0)};
			const amrex::IntVect domhi{AMREX_D_DECL(ba[ba.size() - 1].bigEnd(0), ba[ba.size() - 1].bigEnd(1), ba[ba.size() - 1].bigEnd(2))};
			const amrex::Box dom(domlo, domhi);
			const amrex::Box dom2 = amrex::refine(amrex::coarsen(dom, 2), 2);
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (dom.length(idim) != dom2.length(idim)) {
					fac[idim] = 1;
				}
			}
			amrex::BoxArray ba_lev0(amrex::coarsen(dom, fac));
			ba_lev0.maxSize(max_grid_size[0] / fac);
			ba_lev0.refine(fac);
			// Boxes in ba have even number of cells in each direction
			// unless the domain has odd number of cells in that direction.
			ChopGrids(0, ba_lev0, amrex::ParallelDescriptor::NProcs());
			ba = ba_lev0;
		}

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
			flux_reg_[lev] = std::make_unique<amrex::YAFluxRegister>(ba, boxArray(lev - 1), dm, DistributionMap(lev - 1), Geom(lev), Geom(lev - 1),
										 refRatio(lev - 1), lev, ncomp_cc);
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

	// read in the MultiFab data
	for (int lev = 0; lev <= finest_level; ++lev) {
		// cell-centred
		if (lev == 0) {
			amrex::MultiFab tmp;
			amrex::VisMF::Read(tmp, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Cell"));
			state_new_cc_[0].ParallelCopy(tmp, 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_, nghost_cc_);
		} else {
			amrex::VisMF::Read(state_new_cc_[lev], amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Cell"));
		}
		// face-centred
		if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (lev == 0) {
					amrex::MultiFab tmp;
					amrex::VisMF::Read(tmp, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_",
											      std::string("Face_") + quokka::face_dir_str[idim]));
					state_new_fc_[0][idim].ParallelCopy(tmp, 0, 0, Physics_Indices<problem_t>::nvarPerDim_fc, nghost_fc_, nghost_fc_);
				} else {
					amrex::VisMF::Read(
					    state_new_fc_[lev][idim],
					    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", std::string("Face_") + quokka::face_dir_str[idim]));
				}
			}
		}
	}

#ifdef AMREX_PARTICLES
	// read particle data
	if (do_tracers != 0) {
		AMREX_ASSERT(TracerPC == nullptr);
		TracerPC = std::make_unique<amrex::AmrTracerParticleContainer>(this);
		TracerPC->Restart(restart_chkfile, "tracer_particles");
	}

	// Initialize and register particle containers from checkpoint file

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Rad) {
		AMREX_ASSERT(RadParticles == nullptr);
		RadParticles = std::make_unique<quokka::RadParticleContainer<problem_t>>(this);
		particleRegister_.registerParticleType(RadParticles.get(), quokka::ParticleType::Rad);
		RadParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::Rad));
	}

#if AMREX_SPACEDIM == 3
	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CIC) {
		AMREX_ASSERT(CICParticles == nullptr);
		CICParticles = std::make_unique<quokka::CICParticleContainer>(this);
		particleRegister_.registerParticleType(CICParticles.get(), quokka::ParticleType::CIC);
		CICParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::CIC));
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::CICRad) {
		AMREX_ASSERT(CICRadParticles == nullptr);
		CICRadParticles = std::make_unique<quokka::CICRadParticleContainer<problem_t>>(this);
		particleRegister_.registerParticleType(CICRadParticles.get(), quokka::ParticleType::CICRad);
		CICRadParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::CICRad));
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::StochasticStellarPop) {
		AMREX_ASSERT(StochasticStellarPopParticles == nullptr);
		StochasticStellarPopParticles = std::make_unique<quokka::StochasticStellarPopParticleContainer<problem_t>>(this);
		particleRegister_.registerStarParticleType(StochasticStellarPopParticles.get(), quokka::ParticleType::StochasticStellarPop);
		StochasticStellarPopParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::StochasticStellarPop));
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Sink) {
		AMREX_ASSERT(SinkParticles == nullptr);
		SinkParticles = std::make_unique<quokka::SinkParticleContainer>(this);
		particleRegister_.registerStarParticleType(SinkParticles.get(), quokka::ParticleType::Sink);
		SinkParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::Sink));
	}

	if constexpr (Particle_Traits<problem_t>::particle_switch & ParticleSwitch::Test) {
		AMREX_ASSERT(TestParticles == nullptr);
		TestParticles = std::make_unique<quokka::TestParticleContainer<problem_t>>(this);
		particleRegister_.registerStarParticleType(TestParticles.get(), quokka::ParticleType::Test);
		TestParticles->Restart(restart_chkfile, particleRegister_.getParticleTypeName(quokka::ParticleType::Test));
	}
#endif // AMREX_SPACEDIM == 3
#endif

	areInitialConditionsDefined_ = true;
}

#endif // SIMULATION_HPP_
