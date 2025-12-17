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
	hist_pp.query("weight_by", m_weightType); // "volume", "mass", "cell_counts"

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
	if (m_weightType == "mass") {
		a_varList.push_back("gasDensity");
	}

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

void DiagPDF::writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf)
{
	std::string diagfile;
	if (m_interval > 0 || m_time_interval > 0.0) {
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

		// compute fixed-width column sizes
		for (int n = 0; n < nvars; ++n) {
			widths[n].resize(3, width);
			widths[n][0] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);
			widths[n][1] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);
			widths[n][2] = std::max(width, static_cast<int>(m_varNames[n].length()) + 5);
		}

		// write cycle and simulation time
		pdfFile << "# " << std::setw(width) << "time:"
			<< " " << std::setw(width) << std::setprecision(prec) << std::scientific << a_time << "\n";
		pdfFile << "# " << std::setw(width) << "cycle:"
			<< " " << std::setw(width) << a_nstep << "\n";

		// write variable names
		pdfFile << "# " << std::setw(width) << "variables:";
		for (int n = 0; n < nvars; ++n) {
			pdfFile << " " << std::setw(width) << m_varNames[n];
		}
		pdfFile << "\n";

		// write flag for log-spaced bins
		pdfFile << "# " << std::setw(width) << "is_log_spaced:";
		for (int n = 0; n < nvars; ++n) {
			pdfFile << " " << std::setw(width) << m_useLogSpacedBins[n];
		}
		pdfFile << "\n";

		// write column names for variables
		for (int n = 0; n < nvars; ++n) {
			pdfFile << std::setw(widths[n][0]) << m_varNames[n] + "_idx" << " " << std::setw(widths[n][0]) << m_varNames[n] + "_min" << " "
				<< std::setw(widths[n][1]) << m_varNames[n] + "_max";
		}
		// write out column name for histogram value: "mass_sum", "volume_sum", or "cell_counts_sum"
		pdfFile << " " << std::setw(width) << m_weightType + "_sum" << "\n";

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

			// calculate bin edges
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
				bin_min[n] = bin_left;
				bin_max[n] = bin_right;
			}

			// write out bin edges
			for (int n = 0; n < nvars; ++n) {
				pdfFile << std::setw(width) << idxVec[n] << " " << std::setw(widths[n][0]) << std::setprecision(prec) << std::scientific
					<< bin_min[n] << " " << std::setw(widths[n][1]) << std::setprecision(prec) << std::scientific << bin_max[n] << " ";
			}

			// write histogram value (i.e., un-normalized probability *mass* function)
			amrex::Real const value = a_pdf[linidx];
			pdfFile << std::setw(width) << std::setprecision(prec) << std::scientific << value << "\n";
		}

		pdfFile.flush();
		pdfFile.close();
	}
}
