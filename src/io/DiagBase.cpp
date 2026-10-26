// This file is derived from PelePhysics.
// Original license:
//
// PelePhysics Copyright (c) 2022, Alliance for Sustainable Energy, LLC.,
// through National Renewable Energy Laboratory, the Regents
// of the University of California, through Lawrence Berkeley
// National Laboratory, National Technology and Engineering Solutions
// of Sandia, LLC, and UT-Battelle, LLC (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence
//     Berkeley National Laboratory, Alliance for Sustainable Energy, LLC.,
//     National Renewable Energy Laboratory, National Technology and Engineering
//     Solutions of Sandia LLC, UT-Battelle LLC, U.S. Dept. of Energy, nor the
//     names of its contributors may be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// You are under no obligation whatsoever to provide any bug fixes,
// patches, or upgrades to the features, functionality or performance of
// the source code ("Enhancements") to anyone; however, if you choose to
// make your Enhancements available either publicly, or directly to
// National Renewable Energy Laboratory, Lawrence Berkeley National
// Laboratory, National Technology and Engineering Solutions
// of Sandia LLC, UT-Battelle LLC, or U.S. Dept. of Energy without
// imposing a separate written license agreement for
// such Enhancements, then you hereby grant the following license: a
// non-exclusive, royalty-free perpetual license to install, use, modify,
// prepare derivative works, incorporate into other computer software,
// distribute, and sublicense such enhancements or derivative works
// thereof, in binary and source code form.
//
// Additional modifications SPDX-FileCopyrightText: © The Quokka Authors

#include "DiagBase.H"
#include "AMReX_ParmParse.H"

void DiagBase::init(const std::string &a_prefix, std::string_view a_diagName)
{
	amrex::ParmParse const pp(a_prefix);

	// IO
	pp.query("int", m_interval);
	pp.query("per", m_per);
	pp.queryWithParser("time_int", m_time_interval); // time_int takes precedence over per; supports unit expressions (e.g. "1.0*Myr")
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
	// Always output initial snapshot at step 0 when any interval is active,
	// matching the behaviour of plotfile_interval / plottime_interval.
	// Do NOT update m_next_output_time so the first time-based output fires at t = m_time_interval.
	if (a_nstep == 0) {
		return (m_interval > 0 || m_time_interval > 0.0);
	}

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
