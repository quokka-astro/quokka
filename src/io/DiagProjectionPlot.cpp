#include "DiagProjectionPlot.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

void DiagProjectionPlot::init(const std::string &a_prefix, std::string_view a_diagName)
{
	// Call base class init
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);

	// Check if file prefix ends with "plt" - if not, warn and append it
	if (m_diagfile.size() < 3 || m_diagfile.substr(m_diagfile.size() - 3) != "plt") {
		if (m_diagfile != std::string(a_diagName)) {
			// User specified a custom file prefix that doesn't end in plt
			amrex::Print() << "Warning: DiagProjectionPlot file prefix '" << m_diagfile << "' does not end with 'plt'. Appending '_plt'.\n";
			m_diagfile += "_plt";
		} else {
			// No custom prefix, use default
			m_diagfile = "proj_plt";
		}
	}

	// Read particle types to include (optional, default to empty = no particles)
	int const nParticleTypes = pp.countval("particles");
	if (nParticleTypes > 0) {
		m_particleTypes.resize(nParticleTypes);
		for (int n = 0; n < nParticleTypes; ++n) {
			pp.get("particles", m_particleTypes[n], n);
		}

		amrex::Print() << "DiagProjectionPlot: Including particles: ";
		for (const auto &ptype : m_particleTypes) {
			amrex::Print() << ptype << " ";
		}
		amrex::Print() << "\n";
	} else {
		amrex::Print() << "DiagProjectionPlot: No particles will be included\n";
	}

	// Read projection directions (optional, default to all directions)
	int const nDirs = pp.countval("dirs");
	if (nDirs > 0) {
		std::vector<std::string> dirStrings(nDirs);
		for (int n = 0; n < nDirs; ++n) {
			pp.get("dirs", dirStrings[n], n);
		}

		for (const auto &dirStr : dirStrings) {
			if (dirStr == "x") {
				m_projectionDirs.push_back(amrex::Direction::x);
			}
#if AMREX_SPACEDIM >= 2
			else if (dirStr == "y") {
				m_projectionDirs.push_back(amrex::Direction::y);
			}
#endif
#if AMREX_SPACEDIM == 3
			else if (dirStr == "z") {
				m_projectionDirs.push_back(amrex::Direction::z);
			}
#endif
			else {
				amrex::Print() << "Warning: Unknown projection direction '" << dirStr << "' ignored\n";
			}
		}
	} else {
		// Default to all directions
		m_projectionDirs.push_back(amrex::Direction::x);
#if AMREX_SPACEDIM >= 2
		m_projectionDirs.push_back(amrex::Direction::y);
#endif
#if AMREX_SPACEDIM == 3
		m_projectionDirs.push_back(amrex::Direction::z);
#endif
	}

	amrex::Print() << "DiagProjectionPlot initialized: file=" << m_diagfile << ", interval=" << m_interval << ", dirs=";
	for (const auto &dir : m_projectionDirs) {
		if (dir == amrex::Direction::x) {
			amrex::Print() << "x ";
		}
#if AMREX_SPACEDIM >= 2
		else if (dir == amrex::Direction::y) {
			amrex::Print() << "y ";
		}
#endif
#if AMREX_SPACEDIM == 3
		else if (dir == amrex::Direction::z) {
			amrex::Print() << "z ";
		}
#endif
	}
	amrex::Print() << "\n";
}

void DiagProjectionPlot::prepare(int /*a_nlevels*/, const amrex::Vector<amrex::Geometry> & /*a_geoms*/, const amrex::Vector<amrex::BoxArray> & /*a_grids*/,
				 const amrex::Vector<amrex::DistributionMapping> & /*a_dmap*/, const amrex::Vector<std::string> & /*a_varNames*/)
{
	// DiagProjectionPlot doesn't need special preparation
	// The base class prepare() handles filter setup if needed
	DiagBase::prepare(0, {}, {}, {}, {});
}

void DiagProjectionPlot::processDiag(int /*a_nstep*/, const amrex::Real & /*a_time*/, const amrex::Vector<const amrex::MultiFab *> & /*a_state*/,
				     const amrex::Vector<std::string> & /*a_varNames*/, const YAML::Node & /*simulationMetadata*/)
{
	// The actual work is done in writeProjection() which is called directly from the simulation
	// This method is just a placeholder to satisfy the DiagBase interface
}

void DiagProjectionPlot::addVars(amrex::Vector<std::string> & /*a_varList*/)
{
	// DiagProjectionPlot doesn't use the standard diagnostic variable extraction system
	// It accesses the full state data directly, so we don't add variables here
}
