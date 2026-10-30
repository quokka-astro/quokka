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

#ifndef DIAGPDF_H
#define DIAGPDF_H

#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFabUtil.H"
#include "DiagBase.H"

class DiagPDF : public DiagBase::Register<DiagPDF>
{
      public:
	static auto identifier() -> std::string { return "DiagPDF"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

	void addVars(amrex::Vector<std::string> &a_varList) override;

	static auto MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;
	static auto MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;
	void writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf);

	void close() override {}

      private:
	// PDF properties
	std::string m_weightType{"volume"};    // Weighting types: "volume", "mass", "cell_counts"
	std::vector<std::string> m_varNames{}; // Variable names
	std::vector<int> m_nBins{};	       // Number of bins
	std::vector<int> m_useLogSpacedBins{}; // Use log-spaced bins
	std::vector<bool> m_useFieldMinMax{};  // Use min/max from field
	std::vector<amrex::Real> m_lowBnd{};   // User-defined low bound
	std::vector<amrex::Real> m_highBnd{};  // User-defined high bound
	bool m_reportedRange{false};

	// Geometrical data
	amrex::Vector<amrex::Geometry> m_geoms; // Squirrel away the geoms
	amrex::Vector<amrex::IntVect> m_refRatio;

	// helper functions
	static auto getIdxVec(int linidx, std::vector<int> const &nBins) -> std::vector<int>;

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
								     const amrex::Real &transformedBinWidth, bool doLog) -> int;

	AMREX_GPU_HOST_DEVICE AMREX_INLINE auto getTotalBinCount() -> amrex::Long;
};

// Inline member function implementations
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
							       const amrex::Real &transformedBinWidth, const bool doLog) -> int
{
	amrex::Real const val = doLog ? std::log10(realInputVal) : realInputVal;
	int const cbin = static_cast<int>(std::floor((val - transformedLowBnd) / transformedBinWidth));
	return cbin;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getTotalBinCount() -> amrex::Long
{
	amrex::Long bincount = 1;
	for (const int nbins : m_nBins) {
		bincount *= nbins;
	}
	return bincount;
}

// Template implementation
template <typename problem_t> void DiagPDF::processDiag(int a_nstep, const amrex::Real &a_time)
{
	BL_PROFILE("DiagPDF::processDiag()");

	// Access diagnostic data through protected members
	auto const &a_state = *m_diagMF;
	auto const &a_stateVar = *m_diagVars;

	// Set PDF range
	const int nvars = static_cast<int>(m_varNames.size());
	const int gasDensity_idx = getFieldIndex("gasDensity", a_stateVar);
	amrex::Vector<int> const fieldIdx = getFieldIndexVec(m_varNames, a_stateVar);
	amrex::Vector<amrex::Real> transformed_range(nvars);
	amrex::Vector<amrex::Real> transformed_binWidth(nvars);
	amrex::Vector<amrex::Real> transformed_lowBnd(nvars);

	if (!m_reportedRange) {
		for (int n = 0; n < fieldIdx.size(); ++n) {
			amrex::ignore_unused(MFVecMin(a_state, fieldIdx[n]));
			amrex::ignore_unused(MFVecMax(a_state, fieldIdx[n]));
		}
		m_reportedRange = true;
	}

	for (int n = 0; n < fieldIdx.size(); ++n) {
		if (m_useFieldMinMax[n]) {
			m_lowBnd[n] = MFVecMin(a_state, fieldIdx[n]);
			m_highBnd[n] = MFVecMax(a_state, fieldIdx[n]);
		}
		transformed_range[n] = (m_useLogSpacedBins[n] != 0) ? (std::log10(m_highBnd[n]) - std::log10(m_lowBnd[n])) : (m_highBnd[n] - m_lowBnd[n]);
		transformed_binWidth[n] = transformed_range[n] / m_nBins[n];
		transformed_lowBnd[n] = (m_useLogSpacedBins[n] != 0) ? std::log10(m_lowBnd[n]) : m_lowBnd[n];
	}

	// Data holders
	amrex::Gpu::AsyncVector<amrex::Real> pdf_d(getTotalBinCount(), 0.0);
	amrex::Vector<amrex::Real> pdf(getTotalBinCount(), 0.0);

	// Populate the data from each level on each proc
	for (int lev = 0; lev < a_state.size(); ++lev) {
		// Make mask tagging fine-covered and filtered cells
		amrex::iMultiFab mask;
		if (lev == a_state.size() - 1) {
			mask.define(a_state[lev]->boxArray(), a_state[lev]->DistributionMap(), 1, amrex::IntVect(0));
			mask.setVal(1);
		} else {
			mask =
			    amrex::makeFineMask(*a_state[lev], *a_state[lev + 1], amrex::IntVect(0), m_refRatio[lev], amrex::Periodicity::NonPeriodic(), 1, 0);
		}
		auto const &sarrs = a_state[lev]->const_arrays();
		auto const &marrs = mask.arrays();
		auto *fdata_p = m_filterData.data();
		amrex::ParallelFor(
		    *a_state[lev], amrex::IntVect(0), [=, nFilters = m_filters.size()] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
			    for (int f{0}; f < nFilters; ++f) {
				    amrex::Real const fval =
					sarrs[box_no](i, j, k, fdata_p[f].m_filterVarIdx); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				    if (fval < fdata_p[f].m_low_val ||			   // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					fval > fdata_p[f].m_high_val) {			   // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					    marrs[box_no](i, j, k) = 0;
				    }
			    }
		    });
		amrex::Gpu::streamSynchronize();

		/// accumulate values in histogram

		// set weight factor per cell value
		amrex::Real weightFac{NAN};
		if (m_weightType == "volume" || m_weightType == "mass") {
			auto const cellSize = m_geoms[lev].CellSizeArray();
			weightFac = AMREX_D_TERM(cellSize[0], *cellSize[1], *cellSize[2]);
		} else if (m_weightType == "cell_counts") {
			weightFac = 1.0;
		} else {
			amrex::Abort("Invalid histogram weight type! Must be either 'volume', 'mass' or 'cell_counts'.");
		}

		// set flag to check within ParallelFor below
		bool multiply_by_gasDensity = false;
		if (m_weightType == "mass") {
			multiply_by_gasDensity = true;
		}

		amrex::Gpu::AsyncVector<int> idx_d(nvars);
		amrex::Gpu::AsyncVector<int> nbins_d(nvars);
		amrex::Gpu::AsyncVector<int> doLog_d(nvars);
		amrex::Gpu::AsyncVector<amrex::Real> lowBnd_d(nvars);
		amrex::Gpu::AsyncVector<amrex::Real> binWidth_d(nvars);

		// copy arrays to device
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, fieldIdx.begin(), fieldIdx.end(), idx_d.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_nBins.begin(), m_nBins.end(), nbins_d.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_useLogSpacedBins.begin(), m_useLogSpacedBins.end(), doLog_d.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, transformed_lowBnd.begin(), transformed_lowBnd.end(), lowBnd_d.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, transformed_binWidth.begin(), transformed_binWidth.end(), binWidth_d.begin());
		amrex::Gpu::streamSynchronize();

		// get device pointers
		auto *pdf_d_p = pdf_d.dataPtr();
		auto const *idx_p = idx_d.data();
		auto const *nbins_p = nbins_d.data();
		auto const *lowBnd_p = lowBnd_d.data();
		auto const *binWidth_p = binWidth_d.data();
		auto const *doLog_p = doLog_d.data();

		// compute histogram
		amrex::ParallelFor(*a_state[lev], amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
			if (marrs[box_no](i, j, k) != 0) {
				bool within_range = true;
				amrex::Real weight = weightFac;
				amrex::Long cbin = 0;

				for (int n = 0; n < nvars; ++n) {
					// compute 1D index
					const int bin = getBinIndex1D(sarrs[box_no](i, j, k, idx_p[n]), lowBnd_p[n], binWidth_p[n], doLog_p[n]); // NOLINT
					if (bin < 0 || bin >= nbins_p[n]) {									 // NOLINT
						within_range = false;
					}
					// compute linear index in N-D histogram
					cbin = cbin * nbins_p[n] + static_cast<amrex::Long>(bin); // NOLINT
				}

				if (multiply_by_gasDensity) {
					weight *= sarrs[box_no](i, j, k, gasDensity_idx);
				}

				if (within_range) {
					amrex::HostDevice::Atomic::Add(&(pdf_d_p[cbin]), weight); // NOLINT
				}
			}
		});
		amrex::Gpu::streamSynchronize();
	}

	// sum over all MPI ranks
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, pdf_d.begin(), pdf_d.end(), pdf.begin());
	amrex::Gpu::streamSynchronize();
	amrex::ParallelDescriptor::ReduceRealSum(pdf.data(), static_cast<int>(pdf.size()));

	writePDFToFile(a_nstep, a_time, pdf);
}

#endif
