#ifndef DIAGPROJECTIONPLOT_H
#define DIAGPROJECTIONPLOT_H

#include <string>
#include <unordered_map>
#include <vector>

#include "AMReX_BaseFab.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "AMReX_Vector.H"
#include "DiagBase.H"
#include "projection.hpp"

// Forward declaration
namespace quokka
{
template <typename problem_t> class PhysicsParticleRegister;
} // namespace quokka

class DiagProjectionPlot : public DiagBase::Register<DiagProjectionPlot>
{
      public:
	static auto identifier() -> std::string { return "ProjectionPlot"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			 const amrex::Vector<std::string> &a_varNames, int finest_level, const amrex::Vector<amrex::Geometry> &a_geoms,
			 const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio, int do_tracers, void *tracerPC_ptr,
			 const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> *a_state_fc,
			 const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> *a_varNames_fc, const ParticleWriterFunc &particleWriter,
			 const YAML::Node &simulationMetadata, const ProjectionData *projectionData, amrex::Direction projectionDir) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Get the list of particle types to include (empty means none)
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

	// Check if any particles should be included
	[[nodiscard]] auto includeParticles() const -> bool { return !m_particleTypes.empty(); }

	// Get the diagnostic file name
	[[nodiscard]] auto getDiagFileName() const -> const std::string & { return m_diagfile; }

	// Get projection directions
	[[nodiscard]] auto getProjectionDirs() const -> const std::vector<amrex::Direction> & { return m_projectionDirs; }

      private:
	std::vector<std::string> m_particleTypes;	// List of particle types to include (default: empty)
	std::vector<amrex::Direction> m_projectionDirs; // Projection directions (x, y, z)
};

#endif // DIAGPROJECTIONPLOT_H
