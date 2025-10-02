#include "DiagPlotfile.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

void DiagPlotfile::init(const std::string &a_prefix, std::string_view a_diagName)
{
	// Call base class init
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);

	// Check if file prefix ends with "plt" - if not, warn and append it
	if (m_diagfile.size() < 3 || m_diagfile.substr(m_diagfile.size() - 3) != "plt") {
		if (m_diagfile != std::string(a_diagName)) {
			// User specified a custom file prefix that doesn't end in plt
			amrex::Print() << "Warning: DiagPlotfile file prefix '" << m_diagfile << "' does not end with 'plt'. Appending 'plt'.\n";
			m_diagfile += "_plt";
		} else {
			// No custom prefix, use default
			m_diagfile = "plt";
		}
	}

	// Read particle types to include (optional)
	int const nParticleTypes = pp.countval("particles");
	if (nParticleTypes > 0) {
		m_particleTypes.resize(nParticleTypes);
		m_includeAllParticles = false;
		for (int n = 0; n < nParticleTypes; ++n) {
			pp.get("particles", m_particleTypes[n], n);
		}

		amrex::Print() << "DiagPlotfile: Including only particles: ";
		for (const auto &ptype : m_particleTypes) {
			amrex::Print() << ptype << " ";
		}
		amrex::Print() << "\n";
	} else {
		m_includeAllParticles = true;
		amrex::Print() << "DiagPlotfile: Including all particle types\n";
	}

	// Read number of output files (optional)
	pp.query("nfiles", m_nfiles);

	amrex::Print() << "DiagPlotfile initialized: file=" << m_diagfile << ", interval=" << m_interval << "\n";
}

void DiagPlotfile::prepare(int /*a_nlevels*/, const amrex::Vector<amrex::Geometry> & /*a_geoms*/, const amrex::Vector<amrex::BoxArray> & /*a_grids*/,
			   const amrex::Vector<amrex::DistributionMapping> & /*a_dmap*/, const amrex::Vector<std::string> & /*a_varNames*/)
{
	// DiagPlotfile doesn't need special preparation
	// The base class prepare() handles filter setup if needed
	DiagBase::prepare(0, {}, {}, {}, {});
}

void DiagPlotfile::processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			       const amrex::Vector<std::string> &a_varNames, int finest_level, const amrex::Vector<amrex::Geometry> &a_geoms,
			       const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio, int do_tracers, void *tracerPC_ptr,
			       const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> *a_state_fc,
			       const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> *a_varNames_fc, const ParticleWriterFunc &particleWriter,
			       const YAML::Node &simulationMetadata)
{
	const BL_PROFILE("DiagPlotfile::processDiag()");

	const std::string plotfilename = amrex::Concatenate(m_diagfile, a_nstep, 5);
	amrex::Print() << "DiagPlotfile: Writing plotfile " << plotfilename << "\n";

#ifdef QUOKKA_USE_OPENPMD
	// Write using OpenPMD format
	quokka::OpenPMDOutput::WriteFile(a_varNames, finest_level + 1, a_state, a_geoms, m_diagfile, a_time, a_istep[0]);

	// Write metadata file (outside the plotfile directory for OpenPMD)
	WriteMetadataFile(plotfilename + ".yaml", simulationMetadata);

	amrex::ignore_unused(do_tracers, a_refRatio, tracerPC_ptr, a_state_fc, a_varNames_fc, particleWriter);
#else
	// Set the number of output files if specified
	quokka::ScopedVisMFNOutFiles scoped_nfiles(m_nfiles);

	// Write the main plotfile data using standard AMReX format
	amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, a_state, a_varNames, a_geoms, a_time, a_istep, a_refRatio);

	// Write metadata file (inside the plotfile directory)
	WriteMetadataFile(plotfilename + "/metadata.yaml", simulationMetadata);

	// Write tracer particles if enabled
	if (do_tracers != 0) {
		auto *tracerPC = static_cast<amrex::AmrTracerParticleContainer *>(tracerPC_ptr);
		if (tracerPC != nullptr) {
			tracerPC->WritePlotFile(plotfilename, "tracer_particles");
		}
	}

	// Write face-centered data if provided
	if (a_state_fc != nullptr && a_varNames_fc != nullptr) {
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
			amrex::WriteMultiLevelPlotfile(plotfilename_fc, finest_level + 1, (*a_state_fc)[idim], (*a_varNames_fc)[idim], a_geoms, a_time, a_istep,
						       a_refRatio);
			WriteMetadataFile(plotfilename_fc + "/metadata.yaml", simulationMetadata);
		}
	}

	// Write physics particles using the provided callback
	if (particleWriter) {
		particleWriter(plotfilename);
	}
#endif
}

void DiagPlotfile::addVars(amrex::Vector<std::string> & /*a_varList*/)
{
	// DiagPlotfile doesn't use the standard diagnostic variable extraction system
	// It accesses the full plotfile data directly, so we don't add variables here
}
