#include "DiagBase.H"
#include "AMReX_ParmParse.H"

void DiagBase::init(const std::string &a_prefix, std::string_view a_diagName)
{
	amrex::ParmParse pp(a_prefix);

	// IO
	pp.query("int", m_interval);
	pp.query("per", m_per);
	m_diagfile = a_diagName;
	pp.query("file", m_diagfile);
	AMREX_ASSERT(m_interval > 0 || m_per > 0.0);

	// Filters
	int nFilters = 0;
	nFilters = pp.countval("filters");
	amrex::Vector<std::string> filtersName;
	if (nFilters > 0) {
		m_filters.resize(nFilters);
		filtersName.resize(nFilters);
	}
	for (int n = 0; n < nFilters; ++n) {
		pp.get("filters", filtersName[n], n);
		const std::string filter_prefix = a_prefix + "." + filtersName[n];
		m_filters[n].init(filter_prefix);
	}
}

void DiagBase::prepare(int /*a_nlevels*/, const amrex::Vector<amrex::Geometry> & /*a_geoms*/, const amrex::Vector<amrex::BoxArray> & /*a_grids*/,
		       const amrex::Vector<amrex::DistributionMapping> & /*a_dmap*/, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		int nFilters = static_cast<int>(m_filters.size());
		// Move the filter data to the device
		for (int n = 0; n < nFilters; ++n) {
			m_filters[n].setup(a_varNames);
		}
		amrex::Vector<DiagFilterData> hostFilterData;
		for (int n = 0; n < nFilters; ++n) {
			hostFilterData.push_back(m_filters[n].m_fdata);
		}
		m_filterData.resize(nFilters);
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostFilterData.begin(), hostFilterData.end(), m_filterData.begin());
	}
}

auto DiagBase::doDiag(const amrex::Real &a_time, int a_nstep) -> bool
{
	bool willDo = false;
	if (m_interval > 0 && (a_nstep % m_interval == 0)) {
		willDo = true;
	}

	// TODO: using a_time
	amrex::ignore_unused(a_time);

	return willDo;
}

void DiagBase::addVars(amrex::Vector<std::string> &a_varList)
{
	int nFilters = static_cast<int>(m_filters.size());
	for (int n = 0; n < nFilters; ++n) {
		a_varList.push_back(m_filters[n].m_filterVar);
	}
}

auto DiagBase::getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int
{
	int index = -1;
	for (int n{0}; n < a_varList.size(); ++n) {
		if (a_varList[n] == a_field) {
			index = n;
			break;
		}
	}
	if (index < 0) {
		amrex::Abort("Field : " + a_field + " wasn't found in available fields");
	}
	return index;
}
