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

	void processDiag(int a_nstep, const amrex::Real &a_time, const ProjectionData *projectionData = nullptr,
			 amrex::Direction projectionDir = amrex::Direction::x) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Template method that does the actual work - called from AMRSimulation::doDiagnostics()
	template <typename problem_t>
	void processDiagImpl(AMRSimulation<problem_t> *sim, int a_nstep, const amrex::Real &a_time, const ProjectionData *projectionData,
			     amrex::Direction projectionDir);

	// Get the list of particle types to include (empty means none)
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

	// Get projection directions
	[[nodiscard]] auto getProjectionDirs() const -> const std::vector<amrex::Direction> & { return m_projectionDirs; }

      private:
	std::vector<std::string> m_particleTypes;	// List of particle types to include (default: empty)
	std::vector<amrex::Direction> m_projectionDirs; // Projection directions (x, y, z)
};

// Template implementation
template <typename problem_t>
void DiagProjectionPlot::processDiagImpl(AMRSimulation<problem_t> *sim, int a_nstep, const amrex::Real &a_time, const ProjectionData *projectionData,
					 amrex::Direction projectionDir)
{
	BL_PROFILE("DiagProjectionPlot::processDiagImpl()");

	// If no projection data is provided, skip
	if (projectionData == nullptr || projectionData->empty()) {
		return;
	}

	// Construct the base filename: proj_<direction>_<file>
	std::string basename = "proj_";
	basename += quokka::diagnostics::detail::direction_to_string(projectionDir);
	basename += "_";
	basename += m_diagfile;

	// Write projection data
	quokka::diagnostics::WriteProjection(projectionDir, *projectionData, a_time, a_nstep, basename, sim->simulationMetadata_);

	// If particles are requested, write them
	if (!m_particleTypes.empty()) {
		const std::string filename = amrex::Concatenate(basename, a_nstep, 5);
		sim->particleRegister_.writePlotFileFiltered(filename, m_particleTypes);
		amrex::Print() << "  Wrote particles to projection " << filename << "\n";
	}
}

#endif // DIAGPROJECTIONPLOT_H
