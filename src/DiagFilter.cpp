#include "DiagFilter.H"
#include "AMReX_ParmParse.H"

void DiagFilter::init(const std::string &a_prefix)
{
	amrex::ParmParse pp(a_prefix);

	pp.query("field_name", m_filterVar);
	if (m_filterVar.empty()) {
		amrex::Abort("filter: " + a_prefix + " is missing a field_name !");
	}

	int definedRange = 0;
	if (pp.countval("value_greater") != 0) {
		pp.get("value_greater", m_fdata.m_low_val);
		m_fdata.m_high_val = AMREX_REAL_MAX;
		definedRange = 1;
	} else if (pp.countval("value_less") != 0) {
		pp.get("value_less", m_fdata.m_high_val);
		m_fdata.m_low_val = AMREX_REAL_LOWEST;
		definedRange = 1;
	} else if (pp.countval("value_inrange") != 0) {
		amrex::Vector<amrex::Real> range{0.0};
		pp.getarr("value_inrange", range, 0, 2);
		m_fdata.m_low_val = std::min(range[0], range[1]);
		m_fdata.m_high_val = std::max(range[0], range[1]);
		definedRange = 1;
	}
	if (definedRange == 0) {
		amrex::Abort("filter: " + a_prefix + " is missing a range definition !");
	}
}

void DiagFilter::setup(const amrex::Vector<std::string> &a_varNames)
{
	m_fdata.m_filterVarIdx = -1;
	for (int n{0}; n < a_varNames.size(); ++n) {
		if (a_varNames[n] == m_filterVar) {
			m_fdata.m_filterVarIdx = n;
		}
	}
	if (m_fdata.m_filterVarIdx < 0) {
		amrex::Abort("Couldn't find filter field: " + m_filterVar);
	}
}
