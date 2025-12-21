#ifndef DIAGPARTICLETXT_H
#define DIAGPARTICLETXT_H

#include <string>
#include <vector>

#include "DiagBase.H"
#include "particles/PhysicsParticles.hpp"
#include "particles/particle_IO.hpp"
#include <fstream>

class DiagParticleTxt : public DiagBase::Register<DiagParticleTxt>
{
      public:
	static auto identifier() -> std::string { return "ParticleTxt"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

      private:
	std::vector<std::string> m_particleTypes; // List of particle types to include (empty means all)
};

// Template implementation - processes diagnostic with full access to simulation
template <typename problem_t> void DiagParticleTxt::processDiag(int a_nstep, const amrex::Real &a_time)
{
	const BL_PROFILE("DiagParticleTxt::processDiag()");

	// Get the simulation pointer
	auto *sim = getSim<problem_t>();

	// Generate filename using step number
	const std::string plotfilename = amrex::Concatenate(m_diagfile, a_nstep, 5);

	// Create directory if it doesn't exist
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!amrex::UtilCreateDirectory(plotfilename, 0755)) {
			amrex::CreateDirectoryFailed(plotfilename);
		}
	}
	amrex::ParallelDescriptor::Barrier();

	// Get the particle registry
	auto &particleRegister = sim->GetParticleRegister();

	if (!m_particleTypes.empty()) {
		// Save only specified particle types
		particleRegister.saveParticleDataToTxtFileFiltered(plotfilename, m_particleTypes);
	}
}

#endif // DIAGPARTICLETXT_H
