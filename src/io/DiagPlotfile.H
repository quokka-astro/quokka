#ifndef DIAGPLOTFILE_H
#define DIAGPLOTFILE_H

#include <string>
#include <vector>

#include "AMReX_AmrParticles.H"
#include "AMReX_MultiFab.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_Vector.H"
#include "DiagBase.H"
#include "io/io_utils.hpp"
#include "particles/PhysicsParticles.hpp"
#include <fstream>

#ifdef QUOKKA_USE_OPENPMD
#include "io/openPMD.hpp"
#endif

// Forward declaration
namespace quokka
{
template <typename problem_t> class PhysicsParticleRegister;
enum class ParticleType;
} // namespace quokka

class DiagPlotfile : public DiagBase::Register<DiagPlotfile>
{
      public:
	static auto identifier() -> std::string { return "PlotFile"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			 const amrex::Vector<std::string> &a_varNames, const YAML::Node &simulationMetadata) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Special method for writing cell-centered data and particles
	// This is called directly from AMRSimulation::doDiagnostics()
	template <typename problem_t>
	void writePlotfile(int a_nstep, const amrex::Real &a_time, int finest_level, const amrex::Vector<const amrex::MultiFab *> &a_mf,
			   const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<int> &a_istep,
			   const amrex::Vector<amrex::IntVect> &a_refRatio, quokka::PhysicsParticleRegister<problem_t> &particleRegister, int do_tracers,
			   void *tracerPC_ptr, const YAML::Node &simulationMetadata);

	// Helper method for writing face-centered data
	template <typename problem_t>
	void writePlotfileFC(const std::string &plotfilename, int finest_level,
			     const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> &a_mf_fc,
			     const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> &a_varnames_fc, const amrex::Vector<amrex::Geometry> &a_geoms,
			     const amrex::Real &a_time, const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio,
			     const YAML::Node &simulationMetadata);

	// Check if this diagnostic should write at this time/step (not needed since we use doDiag directly)
	// [[nodiscard]] auto shouldWrite(const amrex::Real &a_time, int a_nstep) const -> bool { return doDiag(a_time, a_nstep); }

	// Get the list of particle types to include (empty means all)
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

	// Check if all particles should be included
	[[nodiscard]] auto includeAllParticles() const -> bool { return m_includeAllParticles; }

	// Get the diagnostic file name
	[[nodiscard]] auto getDiagFileName() const -> const std::string & { return m_diagfile; }

      private:
	std::vector<std::string> m_particleTypes; // List of particle types to include
	bool m_includeAllParticles{true};	  // If true, include all particle types
	int m_nfiles{-1};			  // Number of files for output (-1 means default)

	// Helper function to write metadata file
	static void WriteMetadataFile(const std::string &metadataFilename, const YAML::Node &simulationMetadata);
};

// Template implementation must be in header
template <typename problem_t>
void DiagPlotfile::writePlotfile(int a_nstep, const amrex::Real &a_time, int finest_level, const amrex::Vector<const amrex::MultiFab *> &a_mf,
				 const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<int> &a_istep,
				 const amrex::Vector<amrex::IntVect> &a_refRatio, quokka::PhysicsParticleRegister<problem_t> &particleRegister, int do_tracers,
				 void *tracerPC_ptr, const YAML::Node &simulationMetadata)
{
	BL_PROFILE("DiagPlotfile::writePlotfile()");

	const std::string plotfilename = amrex::Concatenate(m_diagfile, a_nstep, 5);
	amrex::Print() << "DiagPlotfile: Writing plotfile " << plotfilename << "\n";

#ifdef QUOKKA_USE_OPENPMD
	// Write using OpenPMD format
	quokka::OpenPMDOutput::WriteFile(a_varnames, finest_level + 1, a_mf, a_geoms, m_diagfile, a_time, a_istep[0]);

	// Write metadata file (outside the plotfile directory for OpenPMD)
	WriteMetadataFile(plotfilename + ".yaml", simulationMetadata);
#else
	// Set the number of output files if specified
	quokka::ScopedVisMFNOutFiles scoped_nfiles(m_nfiles);

	// Write the main plotfile data using standard AMReX format
	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, a_mf, a_varnames, a_geoms, a_time, a_istep, a_refRatio);

	// Write metadata file (inside the plotfile directory)
	WriteMetadataFile(plotfilename + "/metadata.yaml", simulationMetadata);

	// Write tracer particles if enabled
	if (do_tracers != 0) {
		auto *tracerPC = static_cast<amrex::AmrTracerParticleContainer *>(tracerPC_ptr);
		if (tracerPC != nullptr) {
			tracerPC->WritePlotFile(plotfilename, "tracer_particles");
		}
	}

	// Write physics particles based on configuration
	if (m_includeAllParticles) {
		// Write all particle types
		particleRegister.writePlotFile(plotfilename);
	} else {
		// Write only specified particle types using the filtered method
		particleRegister.writePlotFileFiltered(plotfilename, m_particleTypes);
	}
#endif
}

template <typename problem_t>
void DiagPlotfile::writePlotfileFC(const std::string &plotfilename, int finest_level,
				   const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> &a_mf_fc,
				   const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> &a_varnames_fc, const amrex::Vector<amrex::Geometry> &a_geoms,
				   const amrex::Real &a_time, const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio,
				   const YAML::Node &simulationMetadata)
{
	BL_PROFILE("DiagPlotfile::writePlotfileFC()");

#ifndef QUOKKA_USE_OPENPMD
	if constexpr (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		// Create fc_vars directory if it doesn't exist
		const std::string fc_vars_dir = plotfilename + "/fc_vars";
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::UtilCreateDirectory(fc_vars_dir, 0755);
		}
		amrex::ParallelDescriptor::Barrier();

		std::vector<std::string> dimNames = {"x", "y", "z"};
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto plotfilename_base = plotfilename + "/fc_vars/" + dimNames[idim];
			const std::string plotfilename_fc = amrex::Concatenate(plotfilename_base, a_istep[0], 5);
			amrex::WriteMultiLevelPlotfile(plotfilename_fc, finest_level + 1, a_mf_fc[idim], a_varnames_fc[idim], a_geoms, a_time, a_istep,
						       a_refRatio);
			WriteMetadataFile(plotfilename_fc + "/metadata.yaml", simulationMetadata);
		}
	}
#endif
}

// Helper function implementation
inline void DiagPlotfile::WriteMetadataFile(const std::string &metadataFilename, const YAML::Node &simulationMetadata)
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::ofstream MetadataFile;
		MetadataFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		MetadataFile.open(metadataFilename.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!MetadataFile.good()) {
			amrex::FileOpenFailed(metadataFilename);
		}
		MetadataFile << simulationMetadata << '\n';
		MetadataFile.close();
	}
}

#endif // DIAGPLOTFILE_H
