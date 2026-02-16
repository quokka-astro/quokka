#ifndef DIAG_PARTICLE_DEPOSITION_H_
#define DIAG_PARTICLE_DEPOSITION_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Vector.H"
#include "io/DiagBase.H"

namespace quokka
{

/// Diagnostic class for depositing particle properties into grid fields
/// Supports per-species deposition of mass, momentum, and energy
class DiagParticleDeposition : public DiagBase::Register<DiagParticleDeposition>
{
      public:
	static auto identifier() -> std::string { return "DiagParticleDeposition"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;
	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);
	void addVars(amrex::Vector<std::string> &a_varList) override;
	void close() override {}

      private:
	/// Particle types to deposit (e.g., "CIC", "StochasticStellarPop")
	std::vector<std::string> m_particleTypes;

	/// Fields to deposit for each particle type (e.g., "mass", "momentum", "energy")
	std::vector<std::string> m_depositFields;

	/// Output format: "plotfile" or "ascii"
	std::string m_outputFormat;

	/// Temporary storage for deposition data
	amrex::Vector<amrex::MultiFab> m_depositionData;

	/// Number of components needed for deposition
	int m_nComponents;

	/// Variable names for output
	amrex::Vector<std::string> m_varNames;

	/// Initialize deposition data structures
	void initializeDepositionData(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
				      const amrex::Vector<amrex::DistributionMapping> &a_dmap);

	/// Deposit particle properties onto grid
	void depositParticleProperties(int a_nstep, const amrex::Real &a_time);

	/// Write output data
	void writeOutput(int a_nstep, const amrex::Real &a_time, const YAML::Node &simulationMetadata);

	/// Get number of components for a field
	auto getFieldComponents(const std::string &field) const -> int;

	/// Get field names for a particle type and field
	auto getFieldNames(const std::string &particleType, const std::string &field) const -> std::vector<std::string>;
};

template <typename problem_t> void DiagParticleDeposition::processDiag(int a_nstep, const amrex::Real &a_time)
{
	amrex::ignore_unused(a_nstep);
	amrex::ignore_unused(a_time);
	amrex::ignore_unused(getSim<problem_t>());

	// Clear deposition data
	for (auto &mf : m_depositionData) {
		mf.setVal(0.0);
	}

	// Deposit particle properties
	depositParticleProperties(a_nstep, a_time);

	// Write output
	writeOutput(a_nstep, a_time, *m_metadata);
}

} // namespace quokka

#endif // DIAG_PARTICLE_DEPOSITION_H_
