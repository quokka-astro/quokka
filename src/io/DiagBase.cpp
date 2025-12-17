#include "DiagBase.H"
#include "AMReX_ParmParse.H"

void DiagBase::init(const std::string &a_prefix, std::string_view a_diagName)
{
	amrex::ParmParse const pp(a_prefix);

	// IO
	pp.query("int", m_interval);
	pp.query("per", m_per);
	pp.query("time_int", m_time_interval); // time_int takes precedence over per
	m_diagfile = a_diagName;
	pp.query("file", m_diagfile);
	AMREX_ASSERT(m_interval > 0 || m_per > 0.0 || m_time_interval > 0.0);

	// Initialize next output time for time-based diagnostics
	if (m_time_interval > 0.0) {
		m_next_output_time = m_time_interval;
	}

	// Filters
	int const nFilters = pp.countval("filters");
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
		int const nFilters = static_cast<int>(m_filters.size());
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
	// Reset flag if we're on a new step
	if (m_last_output_step != a_nstep) {
		m_did_output_this_step = false;
	}
	
	bool willDo = false;
	
	// Check step-based output condition
	if (m_interval > 0 && (a_nstep % m_interval == 0)) {
		willDo = true;
	}

	// Check time-based output condition
	// Use the cached decision if we've already decided for this step
	if (m_time_interval > 0.0) {
		if (m_did_output_this_step) {
			// We already decided to output at this step, return true
			willDo = true;
		} else if (m_next_output_time <= a_time) {
			// First check for this step - decide whether to output
			willDo = true;
			m_did_output_this_step = true;
			// Update next output time
			m_next_output_time += m_time_interval;
			// Handle case where multiple intervals have passed
			while (m_next_output_time < a_time) {
				m_next_output_time += m_time_interval;
			}
			// Record this output step
			m_last_output_step = a_nstep;
		}
	}

	return willDo;
}

void DiagBase::addVars(amrex::Vector<std::string> &a_varList)
{
	int const nFilters = static_cast<int>(m_filters.size());
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

auto DiagBase::getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int>
{
	amrex::Vector<int> indexVec(a_field.size());
	for (amrex::Long n = 0; n < indexVec.size(); ++n) {
		indexVec[n] = getFieldIndex(a_field[n], a_varList);
	}
	return indexVec;
}
