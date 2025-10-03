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

	// Read particle types to include (optional, empty means all)
	int const nParticleTypes = pp.countval("particles");
	if (nParticleTypes > 0) {
		m_particleTypes.resize(nParticleTypes);
		for (int n = 0; n < nParticleTypes; ++n) {
			pp.get("particles", m_particleTypes[n], n);
		}

		amrex::Print() << "DiagPlotfile: Including only particles: ";
		for (const auto &ptype : m_particleTypes) {
			amrex::Print() << ptype << " ";
		}
		amrex::Print() << "\n";
	} else {
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

void DiagPlotfile::addVars(amrex::Vector<std::string> & /*a_varList*/)
{
	// DiagPlotfile doesn't use the standard diagnostic variable extraction system
	// It accesses the full plotfile data directly, so we don't add variables here
}
