#ifndef DIAGPROJECTIONPLOT_H
#define DIAGPROJECTIONPLOT_H

#include <string>
#include <unordered_map>
#include <vector>

#include "AMReX_BaseFab.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "AMReX_Reduce.H"
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

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

	// Get the list of particle types to include (empty means none)
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

      private:
	std::vector<std::string> m_particleTypes;	// List of particle types to include (default: empty)
	std::vector<amrex::Direction> m_projectionDirs; // Projection directions (x, y, z)
	std::vector<std::string> m_fieldNames;		// Projection field names
};

// Template implementation
template <typename problem_t> void DiagProjectionPlot::processDiag(int a_nstep, const amrex::Real &a_time)
{
	BL_PROFILE("DiagProjectionPlot::processDiag()");

	// Get the simulation pointer
	auto *sim = getSim<problem_t>();

	// Loop through all requested projection directions
	for (auto const &dir : m_projectionDirs) {
		// Compute projections for this direction
		std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> proj;
		proj.reserve(m_fieldNames.size());
		for (const auto &field : m_fieldNames) {
			const int comp = getFieldIndex(field, *m_diagVars);
			proj[field] = quokka::diagnostics::ComputePlaneProjectionFromMultiFab<amrex::ReduceOpSum>(*m_diagMF, sim->finestLevel(), *m_geoms,
														  *m_refRatio, dir, comp);
		}

		// Skip if no projection data
		if (proj.empty()) {
			continue;
		}

		// Construct the base filename (same behavior as DiagFramePlane).
		std::string const basename = m_diagfile;

		// Write projection data
		quokka::diagnostics::WriteProjection(dir, proj, sim->Geom(), sim->ref_ratio, a_time, a_nstep, basename, sim->simulationMetadata_);

		// If particles are requested, write them
		if (!m_particleTypes.empty()) {
			const std::string filename = amrex::Concatenate(basename, a_nstep, 7);
			sim->particleRegister_.writePlotFileFiltered(filename, m_particleTypes);
			amrex::Print() << "  Wrote particles to projection " << filename << "\n";
		}
	}
}

#endif // DIAGPROJECTIONPLOT_H
