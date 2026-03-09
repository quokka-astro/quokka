#include "DiagProjectionPlot.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

void DiagProjectionPlot::init(const std::string &a_prefix, std::string_view a_diagName)
{
	// Call base class init
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);

	if (!m_filters.empty()) {
		amrex::Print() << " Filters are not available on DiagProjectionPlot and will be discarded \n";
		m_filters.clear();
	}

	// Outputted variables
	pp.queryarr("field_names", m_fieldNames);
	int const nOutFields = static_cast<int>(m_fieldNames.size());
	AMREX_ALWAYS_ASSERT(nOutFields > 0);

	// DiagFramePlane does not enforce a "plt" suffix; match that behavior here.

	// Read particle types to include (optional, default to empty = no particles)
	pp.queryarr("particles", m_particleTypes);
	int const nParticleTypes = static_cast<int>(m_particleTypes.size());
	if (nParticleTypes > 0) {
		amrex::Print() << "DiagProjectionPlot: Including particles: ";
		for (const auto &ptype : m_particleTypes) {
			amrex::Print() << ptype << " ";
		}
		amrex::Print() << "\n";
	} else {
		amrex::Print() << "DiagProjectionPlot: No particles will be included\n";
	}

	// Read projection direction (optional, default to x).
	// Use "normal" to mirror DiagFramePlane.
	int const nNormals = pp.countval("normal");
	int const nDirs = pp.countval("dirs");
	if (nDirs > 0) {
		amrex::Abort("DiagProjectionPlot: 'dirs' is not supported. Use 'normal' (0==x,1==y,2==z).");
	}

	if (nNormals > 0) {
		int normal = 0;
		pp.get("normal", normal);
		if (normal < 0 || normal >= AMREX_SPACEDIM) {
			amrex::Abort("DiagProjectionPlot: 'normal' must be in [0, AMREX_SPACEDIM).");
		}
		if (normal == 0) {
			m_projectionDirs.push_back(amrex::Direction::x);
		}
#if AMREX_SPACEDIM >= 2
		else if (normal == 1) {
			m_projectionDirs.push_back(amrex::Direction::y);
		}
#endif
#if AMREX_SPACEDIM == 3
		else if (normal == 2) {
			m_projectionDirs.push_back(amrex::Direction::z);
		}
#endif
	} else {
		// Default to x direction
		m_projectionDirs.push_back(amrex::Direction::x);
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
				 const amrex::Vector<amrex::DistributionMapping> & /*a_dmap*/, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		for (auto const &field : m_fieldNames) {
			bool found = false;
			for (auto const &name : a_varNames) {
				if (name == field) {
					found = true;
					break;
				}
			}
			if (!found) {
				amrex::Abort("DiagProjectionPlot: Field '" + field + "' is not available!");
			}
		}
	}

	// DiagProjectionPlot doesn't need special preparation
	DiagBase::prepare(0, {}, {}, {}, a_varNames);
}

void DiagProjectionPlot::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	for (const auto &v : m_fieldNames) {
		a_varList.push_back(v);
	}
}
