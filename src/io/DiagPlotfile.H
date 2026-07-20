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

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

	// Get the diagnostic file name
	[[nodiscard]] auto getDiagFileName() const -> const std::string & { return m_diagfile; }

	// Get particle types
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

      private:
	std::vector<std::string> m_particleTypes; // List of particle types to include (empty means all)
	std::vector<std::string> m_varNames;	  // Selected cc field names (empty = all)
	bool m_includeFcFields{true};		  // Whether to write fc_vars (default: true)
	int m_nfiles{-1};			  // Number of files for output (-1 means default)

	// Helper function to write metadata file
	static void WriteMetadataFile(const std::string &metadataFilename, const YAML::Node &simulationMetadata);
};

// Template implementation - processes diagnostic with full access to simulation
template <typename problem_t> void DiagPlotfile::processDiag(int a_nstep, const amrex::Real &a_time)
{
	const BL_PROFILE("DiagPlotfile::processDiag()");

	// Get the simulation pointer
	auto *sim = getSim<problem_t>();

	// Generate filename using step number (works for both step-based and time-based intervals)
	const std::string plotfilename = amrex::Concatenate(m_diagfile, a_nstep, 7);
	amrex::Print() << "DiagPlotfile: Writing plotfile " << plotfilename << "\n";

	// Prepare cell-centered data
#ifdef QUOKKA_USE_OPENPMD
	int const included_ghosts = 0;
#else
	int const included_ghosts = std::min(sim->nghost_cc_, sim->nghost_fc_);
#endif
	amrex::Vector<amrex::MultiFab> mf_cc = sim->PlotFileMF_cc(included_ghosts);
	amrex::Vector<const amrex::MultiFab *> mf_cc_ptr = amrex::GetVecOfConstPtrs(mf_cc);
	auto const varnames = sim->GetPlotfileVarNames();

	// Prepare face-centered data
	std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM> mf_fc = sim->PlotFileMF_fc(sim->nghost_fc_);
	auto const varnames_fc = sim->GetPlotfileVarNames_fc();

	auto const finest_level = sim->finestLevel();
	auto const geoms = sim->Geom(0, finest_level);
	auto const refRatio = sim->refRatio();

	// Filter variables if field_names specified
	amrex::Vector<std::string> varnames_out;
	amrex::Vector<amrex::MultiFab> mf_cc_filtered; // keep alive for scope
	amrex::Vector<const amrex::MultiFab *> mf_cc_out_ptr;

	if (m_varNames.empty()) {
		varnames_out = varnames;
		mf_cc_out_ptr = mf_cc_ptr;
	} else {
		amrex::Vector<int> selected_indices;
		for (const auto &name : m_varNames) {
			bool found = false;
			for (int n = 0; n < static_cast<int>(varnames.size()); ++n) {
				if (varnames[n] == name) {
					varnames_out.push_back(name);
					selected_indices.push_back(n);
					found = true;
					break;
				}
			}
			if (!found) {
				amrex::Abort("DiagPlotfile: Requested field in field_names '" + name + "' not found in available variables!");
			}
		}
		int const nSelected = static_cast<int>(selected_indices.size());
		for (int lev = 0; lev <= finest_level; ++lev) {
			mf_cc_filtered.emplace_back(mf_cc[lev].boxArray(), mf_cc[lev].DistributionMap(), nSelected, mf_cc[lev].nGrowVect());
			for (int n = 0; n < nSelected; ++n) {
				amrex::MultiFab::Copy(mf_cc_filtered.back(), mf_cc[lev], selected_indices[n], n, 1, mf_cc[lev].nGrowVect());
			}
		}
		mf_cc_out_ptr = amrex::GetVecOfConstPtrs(mf_cc_filtered);
	}

#ifdef QUOKKA_USE_OPENPMD
	// Write using OpenPMD format
	quokka::OpenPMDOutput::WriteFile(varnames, finest_level + 1, mf_cc_ptr, geoms, m_diagfile, a_time, sim->istep[0]);

	// Write metadata file (outside the plotfile directory for OpenPMD)
	WriteMetadataFile(plotfilename + ".yaml", sim->simulationMetadata_);
#else
	// Set the number of output files if specified
	quokka::ScopedVisMFNOutFiles scoped_nfiles(m_nfiles);

	// Write the main plotfile data using standard AMReX format
	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf_cc_out_ptr, varnames_out, geoms, a_time, sim->istep, refRatio);

	// Write metadata file (inside the plotfile directory)
	WriteMetadataFile(plotfilename + "/metadata.yaml", sim->simulationMetadata_);

	// Write tracer particles if enabled
	if (sim->TracerPC != nullptr) {
		sim->TracerPC->WritePlotFile(plotfilename, "tracer_particles");
	}

	// Write face-centered data whenever fc variables exist (MHD enabled) and include_fc_fields = 1
	if (m_includeFcFields && !varnames_fc[0].empty()) {
		// Create fc_vars directory if it doesn't exist
		const std::string fc_vars_dir = plotfilename + "/fc_vars";
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::UtilCreateDirectory(fc_vars_dir, 0755);
		}
		amrex::ParallelDescriptor::Barrier();

		std::vector<std::string> dimNames = {"x", "y", "z"};
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto plotfilename_base = plotfilename + "/fc_vars/" + dimNames[idim];
			const std::string plotfilename_fc = amrex::Concatenate(plotfilename_base, sim->istep[0], 7);
			amrex::Vector<const amrex::MultiFab *> mf_fc_ptr = amrex::GetVecOfConstPtrs(mf_fc[idim]);
			amrex::WriteMultiLevelPlotfile(plotfilename_fc, finest_level + 1, mf_fc_ptr, varnames_fc[idim], geoms, a_time, sim->istep, refRatio);
			WriteMetadataFile(plotfilename_fc + "/metadata.yaml", sim->simulationMetadata_);
		}
	}

	// Write physics particles
#if AMREX_SPACEDIM == 3
	if (m_particleTypes.empty()) {
		// Write all particle types
		sim->particleRegister_.redistribute(0, 0);
		sim->particleRegister_.writePlotFile(plotfilename);
	} else {
		// Write only specified particle types
		sim->particleRegister_.redistribute(0, 0);
		sim->particleRegister_.writePlotFileFiltered(plotfilename, m_particleTypes);
	}
#endif // AMREX_SPACEDIM == 3
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
