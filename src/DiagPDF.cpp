#include <ios>

#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_SPACE.H"

#include "DiagPDF.H"

void DiagPDF::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);
	pp.get("field_name", m_fieldName);
	pp.get("nBins", m_nBins);
	AMREX_ALWAYS_ASSERT(m_nBins > 0);
	pp.query("log_spaced_bins", m_useLogSpacedBins);
	pp.query("normalized", m_normalized);
	pp.query("volume_weighted", m_volWeighted);

	if (pp.countval("range") != 0) {
		amrex::Vector<amrex::Real> range{0.0};
		pp.getarr("range", range, 0, 2);
		m_lowBnd = std::min(range[0], range[1]);
		m_highBnd = std::max(range[0], range[1]);
		if (m_useLogSpacedBins) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_lowBnd > 0) && (m_highBnd > 0), "For log-spaced bins, histogram bounds must be positive!");
		}
		m_useFieldMinMax = false;
	}
}

void DiagPDF::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	a_varList.push_back(m_fieldName);
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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getBinIndex(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
							     const amrex::Real &transformedBinWidth, const bool doLog) -> int
{
	amrex::Real const val = doLog ? std::log10(realInputVal) : realInputVal;
	int const cbin = static_cast<int>(std::floor((val - transformedLowBnd) / transformedBinWidth));
	return cbin;
}

void DiagPDF::processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			  const amrex::Vector<std::string> &a_stateVar)
{
	// Set PDF range
	int const fieldIdx = getFieldIndex(m_fieldName, a_stateVar);
	if (m_useFieldMinMax) {
		m_lowBnd = MFVecMin(a_state, fieldIdx);
		m_highBnd = MFVecMax(a_state, fieldIdx);
	}
	amrex::Real const transformed_range = m_useLogSpacedBins ? (std::log10(m_highBnd) - std::log10(m_lowBnd)) : (m_highBnd - m_lowBnd);
	amrex::Real const transformed_binWidth = transformed_range / m_nBins;
	amrex::Real const transformed_lowBnd = m_useLogSpacedBins ? std::log10(m_lowBnd) : m_lowBnd;

	// Data holders
	amrex::Gpu::DeviceVector<amrex::Real> pdf_d(m_nBins, 0.0);
	amrex::Vector<amrex::Real> pdf(m_nBins, 0.0);

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
		auto *pdf_d_p = pdf_d.dataPtr();
		amrex::ParallelFor(*a_state[lev], amrex::IntVect(0),
				   [=, nBins = m_nBins, doLog = m_useLogSpacedBins] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
					   if (marrs[box_no](i, j, k) != 0) {
						   const int cbin =
						       getBinIndex(sarrs[box_no](i, j, k, fieldIdx), transformed_lowBnd, transformed_binWidth, doLog);
						   if (cbin >= 0 && cbin < nBins) {
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

	// normalize histogram
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
		const int prec = 8;
		const int width = 16;
		amrex::Vector<int> widths(3, width);

		widths[0] = std::max(width, static_cast<int>(m_fieldName.length()) + 6);
		widths[1] = std::max(width, static_cast<int>(m_fieldName.length()) + 7);
		widths[2] = std::max(width, static_cast<int>(m_fieldName.length()) + 5);

		pdfFile << std::setw(widths[0]) << m_fieldName << "_left"
			<< " " << std::setw(widths[1]) << m_fieldName << "_right"
			<< " " << std::setw(widths[2]) << m_fieldName + "_PDF"
			<< "\n";

		amrex::Real const transformed_range = m_useLogSpacedBins ? (std::log10(m_highBnd) - std::log10(m_lowBnd)) : (m_highBnd - m_lowBnd);
		amrex::Real const transformed_binWidth = transformed_range / m_nBins;
		amrex::Real const transformed_lowBnd = m_useLogSpacedBins ? std::log10(m_lowBnd) : m_lowBnd;

		for (int i{0}; i < a_pdf.size(); ++i) {
			// calculate bin edges
			amrex::Real const transformed_bin_left = transformed_lowBnd + static_cast<amrex::Real>(i) * transformed_binWidth;
			amrex::Real const transformed_bin_right = transformed_bin_left + transformed_binWidth;
			amrex::Real bin_left{NAN};
			amrex::Real bin_right{NAN};
			if (m_useLogSpacedBins) {
				bin_left = std::pow(10., transformed_bin_left);
				bin_right = std::pow(10., transformed_bin_right);
			} else {
				bin_left = transformed_bin_left;
				bin_right = transformed_bin_right;
			}

			const amrex::Real value = (a_sum != 0) ? (a_pdf[i] / a_sum / (bin_right - bin_left)) : 0;

			pdfFile << std::setw(widths[0]) << std::setprecision(prec) << std::scientific << bin_left << " " << std::setw(widths[1])
				<< std::setprecision(prec) << std::scientific << bin_right << " " << std::setw(widths[2]) << std::setprecision(prec)
				<< std::scientific << value << "\n";
		}

		pdfFile.flush();
		pdfFile.close();
	}
}
