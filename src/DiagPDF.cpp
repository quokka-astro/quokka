#include <ios>

#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_SPACE.H"

#include "DiagPDF.H"

void DiagPDF::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const hist_pp(a_prefix);
	hist_pp.query("normalized", m_normalized);
	hist_pp.query("volume_weighted", m_volWeighted);

	// get number of histogram axes
	const int ndims = hist_pp.countval("var_names");
	m_varNames.resize(ndims);
	m_nBins.resize(ndims);
	m_useLogSpacedBins.resize(ndims);
	m_lowBnd.resize(ndims);
	m_highBnd.resize(ndims);
	m_useFieldMinMax.resize(ndims);

	hist_pp.getarr("var_names", m_varNames, 0, ndims);

	for (int n = 0; n < ndims; ++n) {
		std::string const var_prefix = a_prefix + "." + m_varNames[n];
		amrex::Print() << "[DiagPDF] Reading parameters: " + var_prefix + "\n";

		amrex::ParmParse const var_pp(var_prefix);
		var_pp.get("nBins", m_nBins[n]);
		var_pp.query("log_spaced_bins", m_useLogSpacedBins[n]);
		AMREX_ALWAYS_ASSERT(m_nBins[n] > 0);

		if (var_pp.countval("range") != 0) {
			amrex::Vector<amrex::Real> range{0.0};
			var_pp.getarr("range", range, 0, 2);

			m_lowBnd[n] = std::min(range[0], range[1]);
			m_highBnd[n] = std::max(range[0], range[1]);
			if (m_useLogSpacedBins[n] != 0) {
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_lowBnd[n] > 0) && (m_highBnd[n] > 0),
								 "For log-spaced bins, histogram bounds must be positive!");
			}
			m_useFieldMinMax[n] = false;
		} else {
			m_useFieldMinMax[n] = true;
		}
	}
}

void DiagPDF::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	for (const std::string &var : m_varNames) {
		a_varList.push_back(var);
	}
}

void DiagPDF::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		      const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
		first_time = false;
	}

	m_geoms.resize(a_nlevels);
	m_refRatio.resize(a_nlevels - 1);
	for (int lev = 0; lev < a_nlevels; lev++) {
		m_geoms[lev] = a_geoms[lev];
		if (lev > 0) {
			m_refRatio[lev - 1] = amrex::IntVect(static_cast<int>(a_geoms[lev - 1].CellSize(0) / a_geoms[lev].CellSize(0)));
		}
	}
}

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

auto DiagPDF::getIdxVec(const int linidx, std::vector<int> const &nBins) -> std::vector<int>
{
	const int nvar = static_cast<int>(nBins.size());
	std::vector<int> idxVec(nvar);

	amrex::Long cbin = linidx;
	for (int n = nvar - 1; n >= 0; --n) {
		// original operation: compute linear index in N-D histogram:
		//     cbin = cbin * nBins[n] + static_cast<amrex::Long>(bin);
		// now, we compute the inverse:
		const int bin = static_cast<int>(cbin % nBins[n]);
		cbin = (cbin - bin) / nBins[n];
		idxVec[n] = bin;
	}
	return idxVec;
}

void DiagPDF::processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			  const amrex::Vector<std::string> &a_stateVar)
{
	// Set PDF range
	const int nvars = static_cast<int>(m_varNames.size());
	amrex::Vector<int> const fieldIdx = getFieldIndexVec(m_varNames, a_stateVar);
	amrex::Vector<amrex::Real> transformed_range(nvars);
	amrex::Vector<amrex::Real> transformed_binWidth(nvars);
	amrex::Vector<amrex::Real> transformed_lowBnd(nvars);

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
	amrex::Gpu::DeviceVector<amrex::Real> pdf_d(getTotalBinCount(), 0.0);
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

		// accumulate values in histogram
		amrex::Real weightFac{1.0};
		if (m_volWeighted != 0) {
			auto const cellSize = m_geoms[lev].CellSizeArray();
			weightFac = AMREX_D_TERM(cellSize[0], *cellSize[1], *cellSize[2]);
		}

		amrex::Gpu::DeviceVector<int> idx_d(nvars);
		amrex::Gpu::DeviceVector<int> nbins_d(nvars);
		amrex::Gpu::DeviceVector<int> doLog_d(nvars);
		amrex::Gpu::DeviceVector<amrex::Real> lowBnd_d(nvars);
		amrex::Gpu::DeviceVector<amrex::Real> binWidth_d(nvars);

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

		amrex::ParallelFor(*a_state[lev], amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
			if (marrs[box_no](i, j, k) != 0) {
				bool within_range = true;
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
				if (within_range) {
					amrex::HostDevice::Atomic::Add(&(pdf_d_p[cbin]), weightFac); // NOLINT
				}
			}
		});
		amrex::Gpu::streamSynchronize();
	}

	// sum over all MPI ranks
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, pdf_d.begin(), pdf_d.end(), pdf.begin());
	amrex::Gpu::streamSynchronize();
	amrex::ParallelDescriptor::ReduceRealSum(pdf.data(), static_cast<int>(pdf.size()));

	// compute sum (used to normalize histogram)
	auto sum = std::accumulate(pdf.begin(), pdf.end(), static_cast<decltype(pdf)::value_type>(0));

	writePDFToFile(a_nstep, a_time, pdf, sum);
}

auto DiagPDF::MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real
{
	// TODO(bwibking): skip fine-covered in search
	amrex::Real mmin{AMREX_REAL_MAX};
	for (const auto *st : a_state) {
		mmin = std::min(mmin, st->min(comp, 0, true));
	}

	amrex::ParallelDescriptor::ReduceRealMin(mmin);
	return mmin;
}

auto DiagPDF::MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real
{
	// TODO(bwibking): skip fine-covered in search
	amrex::Real mmax{AMREX_REAL_LOWEST};
	for (const auto *st : a_state) {
		mmax = std::max(mmax, st->max(comp, 0, true));
	}

	amrex::ParallelDescriptor::ReduceRealMax(mmax);
	return mmax;
}

void DiagPDF::writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf, const amrex::Real &a_sum)
{
	std::string diagfile;
	if (m_interval > 0) {
		diagfile = amrex::Concatenate(m_diagfile, a_nstep, 6);
	}
	if (m_per > 0.0) {
		diagfile = m_diagfile + std::to_string(a_time);
	}
	diagfile = diagfile + ".dat";

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream pdfFile;
		pdfFile.open(diagfile.c_str(), std::ios::out);
		const int prec = 17;
		const int width = 25;

		const int nvars = static_cast<int>(m_varNames.size());
		amrex::Vector<amrex::Vector<int>> widths(nvars);

		for (int n = 0; n < nvars; ++n) {
			widths[n].resize(3, width);
			widths[n][0] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);
			widths[n][1] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);
			widths[n][2] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);

			pdfFile << std::setw(widths[n][0]) << m_varNames[n] << "_idx"
				<< " " << std::setw(widths[n][0]) << m_varNames[n] << "_min"
				<< " " << std::setw(widths[n][1]) << m_varNames[n] << "_max";
		}
		pdfFile << " " << std::setw(width) << "PDF"
			<< "\n";

		std::vector<amrex::Real> transformed_range(nvars);
		std::vector<amrex::Real> transformed_binWidth(nvars);
		std::vector<amrex::Real> transformed_lowBnd(nvars);
		for (int n = 0; n < nvars; ++n) {
			transformed_range[n] =
			    (m_useLogSpacedBins[n] != 0) ? (std::log10(m_highBnd[n]) - std::log10(m_lowBnd[n])) : (m_highBnd[n] - m_lowBnd[n]);
			transformed_binWidth[n] = transformed_range[n] / m_nBins[n];
			transformed_lowBnd[n] = (m_useLogSpacedBins[n] != 0) ? std::log10(m_lowBnd[n]) : m_lowBnd[n];
		}

		for (int linidx{0}; linidx < a_pdf.size(); ++linidx) {
			std::vector<int> const idxVec = getIdxVec(linidx, m_nBins);
			std::vector<amrex::Real> bin_min(nvars);
			std::vector<amrex::Real> bin_max(nvars);
			amrex::Real binvol = 1;

			// calculate bin edges, bin volume
			for (int n = 0; n < nvars; ++n) {
				int const i = idxVec[n];
				amrex::Real const transformed_bin_left = transformed_lowBnd[n] + static_cast<amrex::Real>(i) * transformed_binWidth[n];
				amrex::Real const transformed_bin_right = transformed_bin_left + transformed_binWidth[n];
				amrex::Real bin_left{NAN};
				amrex::Real bin_right{NAN};

				if (m_useLogSpacedBins[n] != 0) {
					bin_left = std::pow(10., transformed_bin_left);
					bin_right = std::pow(10., transformed_bin_right);
				} else {
					bin_left = transformed_bin_left;
					bin_right = transformed_bin_right;
				}
				binvol *= (bin_right - bin_left);
				bin_min[n] = bin_left;
				bin_max[n] = bin_right;
			}

			// write out bin edges
			for (int n = 0; n < nvars; ++n) {
				pdfFile << idxVec[n] << " " << std::setw(widths[n][0]) << std::setprecision(prec) << std::scientific << bin_min[n] << " "
					<< std::setw(widths[n][1]) << std::setprecision(prec) << std::scientific << bin_max[n] << " ";
			}

			// write normalized PDF value
			const amrex::Real value = (a_sum != 0) ? (a_pdf[linidx] / a_sum / binvol) : 0;
			pdfFile << std::setw(width) << std::setprecision(prec) << std::scientific << value << "\n";
		}

		pdfFile.flush();
		pdfFile.close();
	}
}
