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
			 const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio, void *particleRegister_ptr,
			 int do_tracers, void *tracerPC_ptr,
			 const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> *a_state_fc,
			 const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> *a_varNames_fc, const YAML::Node &simulationMetadata) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Special method for writing projections with optional particles
	// This is called directly from the simulation
	template <typename problem_t>
	void writeProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> const &proj, amrex::Real time, int istep,
			     quokka::PhysicsParticleRegister<problem_t> &particleRegister, const YAML::Node &simulationMetadata);

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

// Template implementation must be in header
template <typename problem_t>
void DiagProjectionPlot::writeProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> const &proj, amrex::Real time,
					 int istep, quokka::PhysicsParticleRegister<problem_t> &particleRegister, const YAML::Node &simulationMetadata)
{
	BL_PROFILE("DiagProjectionPlot::writeProjection()");

	// Construct the base filename: proj_<direction>_<file>
	// e.g., if m_diagfile is "proj_plt", this becomes "proj_x_proj_plt"
	std::string basename = "proj_";
	basename += quokka::diagnostics::detail::direction_to_string(dir);
	basename += "_";
	basename += m_diagfile;

	// Call the new WriteProjection function with particle support
	quokka::diagnostics::WriteProjection<problem_t>(dir, proj, time, istep, basename, particleRegister, m_particleTypes, simulationMetadata);
}

#endif // DIAGPROJECTIONPLOT_H
