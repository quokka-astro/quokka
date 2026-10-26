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

#ifndef DIAGBASE_H
#define DIAGBASE_H

#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "DiagFilter.H"
#include "Factory.H"
#include "yaml-cpp/yaml.h"
#include <functional>
#include <unordered_map>

// Forward declaration
template <typename problem_t> class AMRSimulation;

class DiagBase : public quokka::Factory<DiagBase>
{
      public:
	~DiagBase() override = default;

	static auto base_identifier() -> std::string { return "DiagBase"; }

	virtual void init(const std::string &a_prefix, std::string_view a_diagName);

	virtual void close() = 0;

	[[nodiscard]] auto needUpdate() const -> bool { return need_update; }

	virtual auto doDiag(const amrex::Real &a_time, int a_nstep) -> bool;

	virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames);

	// Type for projection data (used by DiagProjectionPlot)
	using ProjectionData = std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>;

	// processDiag is implemented as a template method in derived classes
	// Each diagnostic type implements: template<typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time, ...)
	// This allows diagnostics to access problem-specific functionality without virtual function overhead

	virtual void addVars(amrex::Vector<std::string> &a_varList);

	static auto getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int;

	static auto getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int>;

	// Set diagnostic data - called by AMRSimulation::doDiagnostics()
	template <typename problem_t>
	void setDiagData(AMRSimulation<problem_t> *sim, const amrex::Vector<const amrex::MultiFab *> *diagMF, const amrex::Vector<std::string> *diagVars,
			 const amrex::Vector<amrex::Geometry> *geoms, const amrex::Vector<amrex::IntVect> *refRatio, const YAML::Node *metadata)
	{
		m_simulation = sim;
		m_diagMF = diagMF;
		m_diagVars = diagVars;
		m_geoms = geoms;
		m_refRatio = refRatio;
		m_metadata = metadata;
	}

      protected:
	std::string m_diagfile;
	int m_verbose{0};
	amrex::Real m_per{-1.0};
	amrex::Real m_time_interval{-1.0};
	int m_interval{-1};
	amrex::Real m_next_output_time{0.0}; // Track when next time-based output should occur
	int m_last_output_step{-1};	     // Track the last step we decided to output
	bool m_did_output_this_step{false};  // Flag to ensure consistent return value for multiple calls per step
	bool need_update{true};
	bool first_time{true};
	amrex::Vector<DiagFilter> m_filters{};
	amrex::Gpu::DeviceVector<DiagFilterData> m_filterData;

	// Pointer to the AMRSimulation instance (set by setDiagData)
	void *m_simulation{nullptr};

	// Pointers to commonly used diagnostic data (set by setDiagData)
	const amrex::Vector<const amrex::MultiFab *> *m_diagMF{nullptr};
	const amrex::Vector<std::string> *m_diagVars{nullptr};
	const amrex::Vector<amrex::Geometry> *m_geoms{nullptr};
	const amrex::Vector<amrex::IntVect> *m_refRatio{nullptr};
	const YAML::Node *m_metadata{nullptr};

	// Helper to get simulation pointer with correct type
	template <typename problem_t> auto getSim() const -> AMRSimulation<problem_t> * { return static_cast<AMRSimulation<problem_t> *>(m_simulation); }

	// Make AMRSimulation a friend so derived diagnostic classes can access its members
	template <typename problem_t> friend class AMRSimulation;
};

#endif
