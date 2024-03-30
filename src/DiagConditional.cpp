#include <string>

#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"

#include "DiagConditional.H"

void DiagConditional::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	amrex::ParmParse const pp(a_prefix);

	std::string condType;
	pp.get("conditional_type", condType);
	if (condType == "Average") {
		m_condType = Average;
	} else if (condType == "Integral") {
		m_condType = Integral;
	} else if (condType == "Sum") {
		m_condType = Sum;
	} else {
		amrex::Abort("Unknown conditional_type: " + condType);
	}
	pp.get("nBins", m_nBins);
	AMREX_ASSERT(m_nBins > 0);
	pp.get("condition_field_name", m_cFieldName);
	if (pp.countval("range") != 0) {
		amrex::Vector<amrex::Real> range{0.0};
		pp.getarr("range", range, 0, 2);
		m_lowBnd = std::min(range[0], range[1]);
		m_highBnd = std::max(range[0], range[1]);
		m_usecFieldMinMax = false;
	}
	int nProcessFields = -1;
	nProcessFields = pp.countval("field_names");
	AMREX_ASSERT(nProcessFields > 0);
	m_fieldNames.resize(nProcessFields);
	m_fieldIndices_d.resize(nProcessFields);
	for (int f{0}; f < nProcessFields; ++f) {
		pp.get("field_names", m_fieldNames[f], f);
	}
}

void DiagConditional::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	a_varList.push_back(m_cFieldName);
	for (const auto &v : m_fieldNames) {
		a_varList.push_back(v);
	}
}

void DiagConditional::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			      const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
		first_time = false;
		int const nProcessFields = static_cast<int>(m_fieldIndices_d.size());
		amrex::Vector<int> m_fieldIndices(nProcessFields, 0);
		for (int f{0}; f < nProcessFields; ++f) {
			m_fieldIndices[f] = getFieldIndex(m_fieldNames[f], a_varNames);
		}
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_fieldIndices.begin(), m_fieldIndices.end(), m_fieldIndices_d.begin());
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

void DiagConditional::processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
				  const amrex::Vector<std::string> &a_stateVar)
{
	// Set conditional range
	int const cFieldIdx = getFieldIndex(m_cFieldName, a_stateVar);
	if (m_usecFieldMinMax) {
		m_lowBnd = MFVecMin(a_state, cFieldIdx);
		m_highBnd = MFVecMax(a_state, cFieldIdx);
	}
	amrex::Real const binWidth = (m_highBnd - m_lowBnd) / static_cast<amrex::Real>(m_nBins);

	// Data holders
	int const nProcessFields = static_cast<int>(m_fieldIndices_d.size());
	int const vecSize = m_nBins * nProcessFields;
	amrex::Gpu::DeviceVector<amrex::Real> cond_d(vecSize, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> condSq_d(vecSize, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> condAbs_d(m_nBins, 0.0);
	amrex::Gpu::DeviceVector<amrex::Real> condVol_d(m_nBins, 0.0);
	amrex::Vector<amrex::Real> cond(vecSize, 0.0);
	amrex::Vector<amrex::Real> condSq(vecSize, 0.0);
	amrex::Vector<amrex::Real> condAbs(m_nBins, 0.0);
	amrex::Vector<amrex::Real> condVol(m_nBins, 0.0);

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
		amrex::ParallelFor(*a_state[lev], amrex::IntVect(0),
				   [=, nFilters = m_filters.size()] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
					   for (int f{0}; f < nFilters; ++f) {
						   amrex::Real const fval = sarrs[box_no](i, j, k, fdata_p[f].m_filterVarIdx); // NOLINT
						   if (fval < fdata_p[f].m_low_val || fval > fdata_p[f].m_high_val) {	       // NOLINT
							   marrs[box_no](i, j, k) = 0;
						   }
					   }
				   });
		amrex::Gpu::streamSynchronize();

		// Get the geometry volume to account for 2D-RZ
		amrex::MultiFab volume(a_state[lev]->boxArray(), a_state[lev]->DistributionMap(), 1, 0);
		m_geoms[lev].GetVolume(volume);
		auto const &varrs = volume.const_arrays();

		auto *cond_d_p = cond_d.dataPtr();
		auto *condAbs_d_p = condAbs_d.dataPtr();
		auto *idx_d_p = m_fieldIndices_d.dataPtr();
		if (m_condType == Average) {
			auto *condSq_d_p = condSq_d.dataPtr();
			auto *condVol_d_p = condVol_d.dataPtr();
			amrex::ParallelFor(
			    *a_state[lev], amrex::IntVect(0),
			    [=, nBins = m_nBins, lowBnd = m_lowBnd] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
				    if (marrs[box_no](i, j, k) != 0) {
					    int const cbin = static_cast<int>(std::floor((sarrs[box_no](i, j, k, cFieldIdx) - lowBnd) / binWidth));
					    if (cbin >= 0 && cbin < nBins) {
						    for (int f{0}; f < nProcessFields; ++f) {
							    int const fidx = idx_d_p[f]; // NOLINT
							    int const binOffset = f * nBins;
							    amrex::HostDevice::Atomic::Add(&(cond_d_p[binOffset + cbin]), // NOLINT
											   varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, fidx));
							    amrex::HostDevice::Atomic::Add(&(condSq_d_p[binOffset + cbin]), // NOLINT
											   varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, fidx) *
											       sarrs[box_no](i, j, k, fidx));
						    }
						    amrex::HostDevice::Atomic::Add(&(condVol_d_p[cbin]), varrs[box_no](i, j, k)); // NOLINT
						    amrex::HostDevice::Atomic::Add(&(condAbs_d_p[cbin]),			  // NOLINT
										   varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, cFieldIdx));
					    }
				    }
			    });
		} else if (m_condType == Integral) {
			amrex::ParallelFor(
			    *a_state[lev], amrex::IntVect(0),
			    [=, nBins = m_nBins, lowBnd = m_lowBnd] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
				    if (marrs[box_no](i, j, k) != 0) {
					    int const cbin = static_cast<int>(std::floor((sarrs[box_no](i, j, k, cFieldIdx) - lowBnd) / binWidth));
					    if (cbin >= 0 && cbin < nBins) {
						    for (int f{0}; f < nProcessFields; ++f) {
							    int const fidx = idx_d_p[f]; // NOLINT
							    int const binOffset = f * nBins;
							    amrex::HostDevice::Atomic::Add(&(cond_d_p[binOffset + cbin]), // NOLINT
											   varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, fidx));
						    }
						    amrex::HostDevice::Atomic::Add(&(condAbs_d_p[cbin]), // NOLINT
										   varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, cFieldIdx));
					    }
				    }
			    });
		} else if (m_condType == Sum) {
			amrex::ParallelFor(*a_state[lev], amrex::IntVect(0),
					   [=, nBins = m_nBins, lowBnd = m_lowBnd] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
						   if (marrs[box_no](i, j, k) != 0) {
							   int const cbin =
							       static_cast<int>(std::floor((sarrs[box_no](i, j, k, cFieldIdx) - lowBnd) / binWidth));
							   if (cbin >= 0 && cbin < nBins) {
								   for (int f{0}; f < nProcessFields; ++f) {
									   int const fidx = idx_d_p[f]; // NOLINT
									   int const binOffset = f * nBins;
									   amrex::HostDevice::Atomic::Add(&(cond_d_p[binOffset + cbin]), // NOLINT
													  sarrs[box_no](i, j, k, fidx));
								   }
								   amrex::HostDevice::Atomic::Add(&(condAbs_d_p[cbin]), // NOLINT
												  varrs[box_no](i, j, k) * sarrs[box_no](i, j, k, cFieldIdx));
							   }
						   }
					   });
		}
		amrex::Gpu::streamSynchronize();
	}

	amrex::Gpu::copy(amrex::Gpu::deviceToHost, cond_d.begin(), cond_d.end(), cond.begin());
	amrex::Gpu::streamSynchronize();
	amrex::ParallelDescriptor::ReduceRealSum(cond.data(), static_cast<int>(cond.size()));
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, condAbs_d.begin(), condAbs_d.end(), condAbs.begin());
	amrex::Gpu::streamSynchronize();
	amrex::ParallelDescriptor::ReduceRealSum(condAbs.data(), static_cast<int>(condAbs.size()));
	if (m_condType == Average) {
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, condVol_d.begin(), condVol_d.end(), condVol.begin());
		amrex::Gpu::streamSynchronize();
		amrex::ParallelDescriptor::ReduceRealSum(condVol.data(), static_cast<int>(condVol.size()));
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, condSq_d.begin(), condSq_d.end(), condSq.begin());
		amrex::Gpu::streamSynchronize();
		amrex::ParallelDescriptor::ReduceRealSum(condSq.data(), static_cast<int>(condSq.size()));
		for (int f{0}; f < nProcessFields; ++f) {
			int const binOffset = f * m_nBins;
			for (int n{0}; n < m_nBins; ++n) {
				if (condVol[n] != 0.0) {
					cond[binOffset + n] /= condVol[n];
					condSq[binOffset + n] /= condVol[n];
				}
			}
		}
		for (int n{0}; n < m_nBins; ++n) {
			if (condVol[n] != 0.0) {
				condAbs[n] /= condVol[n];
			}
		}
	}

	// Write data to file
	if (m_condType == Average) {
		writeAverageDataToFile(a_nstep, a_time, condAbs, cond, condSq, condVol);
	} else if (m_condType == Integral) {
		writeIntegralDataToFile(a_nstep, a_time, condAbs, cond);
	} else if (m_condType == Sum) {
		writeSumDataToFile(a_nstep, a_time, condAbs, cond);
	}
}

auto DiagConditional::MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real
{
	// TODO(bwibking): skip fine-covered in search
	amrex::Real mmin{AMREX_REAL_MAX};
	for (const auto *st : a_state) {
		mmin = std::min(mmin, st->min(comp, 0, true));
	}

	amrex::ParallelDescriptor::ReduceRealMin(mmin);
	return mmin;
}

auto DiagConditional::MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real
{
	// TODO(bwibking): skip fine-covered in search
	amrex::Real mmax{AMREX_REAL_LOWEST};
	for (const auto *st : a_state) {
		mmax = std::max(mmax, st->max(comp, 0, true));
	}

	amrex::ParallelDescriptor::ReduceRealMax(mmax);
	return mmax;
}

void DiagConditional::writeAverageDataToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_condAbs,
					     const amrex::Vector<amrex::Real> &a_cond, const amrex::Vector<amrex::Real> &a_condSq,
					     const amrex::Vector<amrex::Real> &a_condVol)
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
		std::ofstream condFile;
		condFile.open(diagfile.c_str(), std::ios::out);
		const int prec = 8;
		const int width = 16;
		const int nProcessFields = static_cast<int>(m_fieldIndices_d.size());
		amrex::Vector<int> widths(3 + 2 * nProcessFields, width);

		condFile << std::left << std::setw(widths[0]) << "BinCenter"
			 << " ";
		widths[1] = std::max(width, static_cast<int>(m_cFieldName.length()) + 1);
		condFile << std::left << std::setw(widths[1]) << m_cFieldName << " ";
		condFile << std::left << std::setw(widths[2]) << "Volume"
			 << " ";
		for (int f{0}; f < nProcessFields; ++f) {
			widths[3 + 2 * f] = std::max(width, static_cast<int>(m_fieldNames[f].length()) + 5);
			condFile << std::left << std::setw(widths[3 + 2 * f]) << m_fieldNames[f] + "_Avg"
				 << " ";
			widths[4 + 2 * f] = std::max(width, static_cast<int>(m_fieldNames[f].length()) + 7);
			condFile << std::left << std::setw(widths[4 + 2 * f]) << m_fieldNames[f] + "_StdDev"
				 << " ";
		}
		condFile << "\n";

		// Retrieve some data
		amrex::Real const binWidth = (m_highBnd - m_lowBnd) / (m_nBins);

		for (int n{0}; n < m_nBins; ++n) {
			condFile << std::left << std::setw(widths[0]) << std::setprecision(prec) << std::scientific << m_lowBnd + (n + 0.5) * binWidth << " ";
			condFile << std::left << std::setw(widths[1]) << std::setprecision(prec) << std::scientific << a_condAbs[n] << " ";
			condFile << std::left << std::setw(widths[2]) << std::setprecision(prec) << std::scientific << a_condVol[n] << " ";
			for (int f{0}; f < nProcessFields; ++f) {
				int const binOffset = f * m_nBins;
				condFile << std::left << std::setw(widths[3 + 2 * f]) << std::setprecision(prec) << std::scientific << a_cond[binOffset + n]
					 << " " << std::setw(widths[4 + 2 * f]) << std::setprecision(prec) << std::scientific
					 << std::sqrt(std::abs(a_condSq[binOffset + n] - a_cond[binOffset + n] * a_cond[binOffset + n])) << " ";
			}
			condFile << "\n";
		}
		condFile.flush();
		condFile.close();
	}
}

void DiagConditional::writeIntegralDataToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_condAbs,
					      const amrex::Vector<amrex::Real> &a_cond)
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
		std::ofstream condFile;
		condFile.open(diagfile.c_str(), std::ios::out | std::ios::app);
		const int prec = 8;
		const int width = 16;
		const int nProcessFields = static_cast<int>(m_fieldIndices_d.size());
		amrex::Vector<int> widths(1 + nProcessFields, width);

		widths[0] = std::max(width, static_cast<int>(m_cFieldName.length()) + 1);
		condFile << std::left << std::setw(widths[0]) << m_cFieldName << " ";
		for (int f{0}; f < nProcessFields; ++f) {
			widths[1 + f] = std::max(width, static_cast<int>(m_fieldNames[f].length()) + 5);
			condFile << std::left << std::setw(widths[1 + f]) << m_fieldNames[f] + "_Int"
				 << " ";
		}
		condFile << "\n";

		for (int n{0}; n < m_nBins; ++n) {
			condFile << std::left << std::setw(widths[0]) << std::setprecision(prec) << std::scientific << a_condAbs[n] << " ";
			for (int f{0}; f < nProcessFields; ++f) {
				int const binOffset = f * m_nBins;
				condFile << std::left << std::setw(widths[1 + f]) << std::setprecision(prec) << std::scientific << a_cond[binOffset + n] << " ";
			}
			condFile << "\n";
		}
		condFile.flush();
		condFile.close();
	}
}

void DiagConditional::writeSumDataToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_condAbs,
					 const amrex::Vector<amrex::Real> &a_cond)
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
		std::ofstream condFile;
		condFile.open(diagfile.c_str(), std::ios::out | std::ios::app);
		const int prec = 8;
		const int width = 16;
		const int nProcessFields = static_cast<int>(m_fieldIndices_d.size());
		amrex::Vector<int> widths(1 + nProcessFields, width);

		widths[0] = std::max(width, static_cast<int>(m_cFieldName.length()) + 1);
		condFile << std::left << std::setw(widths[0]) << m_cFieldName << " ";
		for (int f{0}; f < nProcessFields; ++f) {
			widths[1 + f] = std::max(width, static_cast<int>(m_fieldNames[f].length()) + 5);
			condFile << std::left << std::setw(widths[1 + f]) << m_fieldNames[f] + "_Sum"
				 << " ";
		}
		condFile << "\n";

		for (int n{0}; n < m_nBins; ++n) {
			condFile << std::left << std::setw(widths[0]) << std::setprecision(prec) << std::scientific << a_condAbs[n] << " ";
			for (int f{0}; f < nProcessFields; ++f) {
				int const binOffset = f * m_nBins;
				condFile << std::left << std::setw(widths[1 + f]) << std::setprecision(prec) << std::scientific << a_cond[binOffset + n] << " ";
			}
			condFile << "\n";
		}
		condFile.flush();
		condFile.close();
	}
}
