#include "DiagFramePlane.H"
#include "AMReX_FPC.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_VisMF.H"

void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim)
{
	int doneDim = 0;
	a_File << '(';
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		if (idim != skipDim) {
			a_File << a_IntVect[idim];
			doneDim++;
			if (doneDim < AMREX_SPACEDIM - 1) {
				a_File << ",";
			}
		}
	}
	a_File << ')';
}

void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim)
{
	a_File << '(';
	printLowerDimIntVect(a_File, a_box.smallEnd(), skipDim);
	a_File << ' ';
	printLowerDimIntVect(a_File, a_box.bigEnd(), skipDim);
	a_File << ' ';
	printLowerDimIntVect(a_File, a_box.type(), skipDim);
	a_File << ')';
}

void DiagFramePlane::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	if (m_filters.empty()) {
		amrex::Print() << " Filters are not available on DiagFramePlane and will "
				  "be discarded \n";
	}

	amrex::ParmParse const pp(a_prefix);

	// Outputted variables
	int const nOutFields = pp.countval("field_names");
	AMREX_ASSERT(nOutFields > 0);
	m_fieldNames.resize(nOutFields);
	m_fieldIndices_d.resize(nOutFields);
	for (int f{0}; f < nOutFields; ++f) {
		pp.get("field_names", m_fieldNames[f], f);
	}

	// Plane normal
	pp.get("normal", m_normal);
	AMREX_ASSERT(m_normal >= 0 && m_normal < AMREX_SPACEDIM);

	// Plane center
	amrex::Vector<amrex::Real> center;
	pp.getarr("center", center, 0, pp.countval("center"));
	if (center.size() == AMREX_SPACEDIM) {
		for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
			m_center[idim] = center[idim];
		}
	} else if (center.size() == 1) {
		m_center[m_normal] = center[0];
	}

	// Interpolation
	std::string intType = "Quadratic";
	pp.query("interpolation", intType);
	if (intType == "Linear") {
		m_interpType = Linear;
	} else if (intType == "Quadratic") {
		m_interpType = Quadratic;
	} else {
		amrex::Abort("Unknown interpolation type for " + a_prefix);
	}
}

void DiagFramePlane::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	for (const auto &v : m_fieldNames) {
		a_varList.push_back(v);
	}
}

void DiagFramePlane::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		// Store the level0 geometry
		auto initDomain = a_geoms[0].Domain();
		auto initRealBox = a_geoms[0].ProbDomain();
		amrex::GpuArray const dxlcl = a_geoms[0].CellSizeArray();
		int cdim = 0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (idim != m_normal) {
				initDomain.setRange(cdim, initDomain.smallEnd(idim), initDomain.bigEnd(idim) + 1);
				initRealBox.setLo(cdim, a_geoms[0].ProbLo(idim));
				initRealBox.setHi(cdim, a_geoms[0].ProbHi(idim));
				cdim += 1;
			}
		}
		initDomain.setRange(2, 0, 1);
		initRealBox.setLo(2, 0.0);
		initRealBox.setHi(2, dxlcl[2]);
		m_geomLev0.define(initDomain, initRealBox, a_geoms[0].Coord(), amrex::Array<int, AMREX_SPACEDIM>({AMREX_D_DECL(0, 0, 0)}));

		int const nOutFields = static_cast<int>(m_fieldIndices_d.size());
		amrex::Vector<int> m_fieldIndices(nOutFields, 0);
		for (int f{0}; f < nOutFields; ++f) {
			m_fieldIndices[f] = getFieldIndex(m_fieldNames[f], a_varNames);
		}
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_fieldIndices.begin(), m_fieldIndices.end(), m_fieldIndices_d.begin());

		first_time = false;
	}

	// Resize internal vectors
	m_intwgt.resize(a_nlevels);
	m_k0.resize(a_nlevels);

	// On each level, find the k0 where the plane lays
	// and the weight of the directionnal interpolation
	for (int lev = 0; lev < a_nlevels; lev++) {
		amrex::GpuArray const dx = a_geoms[lev].CellSizeArray();
		amrex::GpuArray const problo = a_geoms[lev].ProbLoArray();
		// How many dx away from the lowest cell-center ?
		amrex::Real dist = (m_center[m_normal] - (problo[m_normal] + 0.5 * dx[m_normal])) / dx[m_normal];
		int const k0 = static_cast<int>(std::round(dist));
		dist -= static_cast<amrex::Real>(k0);
		m_k0[lev] = k0;
		if (m_interpType == Quadratic) {
			// Quadratic interp. weights on k0-1, k0, k0+1
			m_intwgt[lev][0] = 0.5 * (dist - 1.0) * (dist - 2.0);
			m_intwgt[lev][1] = dist * (2.0 - dist);
			m_intwgt[lev][2] = 0.5 * dist * (dist - 1.0);
			;
		} else if (m_interpType == Linear) {
			// linear interp. weights on k0-1, k0, k0+1
			if (dist > 0.0) {
				m_intwgt[lev][0] = 0.0;
				m_intwgt[lev][1] = 1.0 - dist;
				m_intwgt[lev][2] = dist;
			} else if (dist < 0.0) {
				m_intwgt[lev][0] = -dist;
				m_intwgt[lev][1] = 1.0 + dist;
				m_intwgt[lev][2] = 0.0;
			} else {
				m_intwgt[lev][0] = 0.0;
				m_intwgt[lev][1] = 1.0;
				m_intwgt[lev][2] = 0.0;
			}
		}
	}

	// Assemble the 2D slice boxArray
	m_sliceBA.resize(a_nlevels);
	m_sliceDM.resize(a_nlevels);
	m_dmConvert.resize(a_nlevels);
	for (int lev = 0; lev < a_nlevels; lev++) {
		amrex::Vector<int> pmap;
		amrex::BoxList bl(a_grids[lev].ixType());
		bl.reserve(a_grids[lev].size());
		amrex::Vector<int> dmConvertLev;
		for (int i = 0; i < a_grids[lev].size(); ++i) {
			auto cBox = a_grids[lev][i];
			amrex::IntVect ploc(AMREX_D_DECL(cBox.smallEnd(0), cBox.smallEnd(1), cBox.smallEnd(2)));
			ploc[m_normal] = m_k0[lev];
			if (cBox.contains(ploc)) {
				amrex::Box zNormalBax;
				int idx = 0;
				for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
					if (idim != m_normal) {
						zNormalBax.setRange(idx, cBox.smallEnd(idim), cBox.length(idim));
						idx++;
					}
				}
				zNormalBax.setRange(AMREX_SPACEDIM - 1, 0, 1);
				bl.push_back(zNormalBax);
				pmap.push_back(a_dmap[lev][i]);
				dmConvertLev.push_back(i);
			}
			m_dmConvert[lev] = dmConvertLev;
		}
		m_sliceBA[lev].define(bl);
		m_sliceDM[lev].define(pmap);
	}
}

void DiagFramePlane::processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
				 const amrex::Vector<std::string> & /*a_stateVar*/)
{
	// Interpolate data to slice
	amrex::Vector<amrex::MultiFab> planeData(a_state.size());
	for (int lev = 0; lev < a_state.size(); ++lev) {
		planeData[lev].define(m_sliceBA[lev], m_sliceDM[lev], static_cast<int>(m_fieldNames.size()), 0);
		int const p0 = m_k0[lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
		for (amrex::MFIter mfi(planeData[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
			const auto &bx = mfi.tilebox();
			const int state_idx = m_dmConvert[lev][mfi.index()];
			auto const &state = a_state[lev]->const_array(state_idx, 0);
			auto const &plane = planeData[lev].array(mfi);
			auto const &intwgt = m_intwgt[lev];
			auto *idx_d_p = m_fieldIndices_d.dataPtr();
			if (m_normal == 0) {
				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
					int const stIdx = idx_d_p[n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					plane(i, j, k, n) = intwgt[0] * state(p0 - 1, i, j, stIdx) + intwgt[1] * state(p0, i, j, stIdx) +
							    intwgt[2] * state(p0 + 1, i, j, stIdx);
				});
			} else if (m_normal == 1) {
				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
					int const stIdx = idx_d_p[n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					plane(i, j, k, n) = intwgt[0] * state(i, p0 - 1, j, stIdx) + intwgt[1] * state(i, p0, j, stIdx) +
							    intwgt[2] * state(i, p0 + 1, j, stIdx);
				});
			} else if (m_normal == 2) {
				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
					int const stIdx = idx_d_p[n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					plane(i, j, k, n) = intwgt[0] * state(i, j, p0 - 1, stIdx) + intwgt[1] * state(i, j, p0, stIdx) +
							    intwgt[2] * state(i, j, p0 + 1, stIdx);
				});
			}
		}
	}

	// Count the number of level where the cut exists
	int nlevs = 0;
	for (int lev = 0; lev < a_state.size(); lev++) {
		if (!m_sliceBA[lev].empty()) {
			nlevs += 1;
		}
	}

	if (nlevs > 0) {
		// Build up a z-normal 2D Geom
		amrex::Vector<amrex::Geometry> pltGeoms(nlevs);
		pltGeoms[0] = m_geomLev0;
		amrex::Vector<amrex::IntVect> ref_ratio;
		amrex::IntVect const rref(AMREX_D_DECL(2, 2, 1));
		for (int lev = 1; lev < nlevs; ++lev) {
			pltGeoms[lev] = amrex::refine(pltGeoms[lev - 1], rref);
			ref_ratio.push_back(rref);
		}

		// File name based on tep or time
		std::string diagfile;
		if (m_interval > 0) {
			diagfile = amrex::Concatenate(m_diagfile, a_nstep, 6);
		}
		if (m_per > 0.0) {
			diagfile = m_diagfile + std::to_string(a_time);
		}
		amrex::Vector<int> const step_array(nlevs, a_nstep);
		Write2DMultiLevelPlotfile(diagfile, nlevs, GetVecOfConstPtrs(planeData), m_fieldNames, pltGeoms, a_time, step_array, ref_ratio);
	}
}

void DiagFramePlane::Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice,
					       const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms,
					       const amrex::Real &a_time, const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref)
{
	const std::string levelPrefix = "Level_";
	const std::string mfPrefix = "Cell";
	const std::string versionName = "HyperCLaw-V1.1";

	bool const callBarrier(false);
	amrex::PreBuildDirectorHierarchy(a_pltfile, levelPrefix, a_nlevels, callBarrier);
	amrex::ParallelDescriptor::Barrier();

	if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::NProcs() - 1) {

		// Write 2D pltfile header
		amrex::Vector<amrex::BoxArray> boxArrays(a_nlevels);
		for (int level(0); level < boxArrays.size(); ++level) {
			boxArrays[level] = a_slice[level]->boxArray();
		}

		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::string const HeaderFileName(a_pltfile + "/Header");
		std::ofstream HeaderFile;
		HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!HeaderFile.good()) {
			amrex::FileOpenFailed(HeaderFileName);
		}
		Write2DPlotfileHeader(HeaderFile, a_nlevels, boxArrays, a_varnames, a_geoms, a_time, a_steps, a_rref, versionName, levelPrefix, mfPrefix);
		HeaderFile.flush();
		HeaderFile.close();

		// Add extra file providing the plane information
		// because the output is always a z-normal plane at z=0
		std::string const PlaneFileName(a_pltfile + "/PlaneData");
		std::ofstream PlaneFile;
		PlaneFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		PlaneFile.open(PlaneFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!PlaneFile.good()) {
			amrex::FileOpenFailed(PlaneFileName);
		}

		PlaneFile << "Pele plane definition: \n";
		PlaneFile << m_normal << "\n";
		PlaneFile << AMREX_D_TERM(m_center[0], << " " << m_center[1], << " " << m_center[2]) << "\n";

		PlaneFile.flush();
		PlaneFile.close();
	}

	// Write a 2D version of the MF at each level
	for (int level = 0; level < a_nlevels; ++level) {
		VisMF2D(*a_slice[level], amrex::MultiFabFileFullPrefix(level, a_pltfile, levelPrefix, mfPrefix));
	}
}

void DiagFramePlane::Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray,
					   const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time,
					   const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio,
					   const std::string &versionName, const std::string &levelPrefix, const std::string &mfPrefix)
{
	int const finest_level(nlevels - 1);
	HeaderFile.precision(17);

	int const lowerSpaceDim = AMREX_SPACEDIM - 1;

	HeaderFile << versionName << '\n';
	HeaderFile << varnames.size() << '\n';
	for (const auto &varname : varnames) {
		HeaderFile << varname << "\n";
	}
	HeaderFile << lowerSpaceDim << '\n';
	HeaderFile << time << '\n';
	HeaderFile << finest_level << '\n';
	for (int idim = 0; idim < lowerSpaceDim; ++idim) {
		HeaderFile << geom[0].ProbLo(idim) << ' ';
	}
	HeaderFile << '\n';
	for (int idim = 0; idim < lowerSpaceDim; ++idim) {
		HeaderFile << geom[0].ProbHi(idim) << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i < finest_level; ++i) {
		HeaderFile << ref_ratio[i][0] << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		printLowerDimBox(HeaderFile, geom[i].Domain(), 2);
		HeaderFile << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		HeaderFile << level_steps[i] << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		for (int idim = 0; idim < lowerSpaceDim; ++idim) {
			HeaderFile << geom[i].CellSizeArray()[idim] << ' ';
		}
		HeaderFile << '\n';
	}
	HeaderFile << static_cast<int>(geom[0].Coord()) << '\n';
	HeaderFile << "0\n";

	for (int level = 0; level <= finest_level; ++level) {
		HeaderFile << level << ' ' << bArray[level].size() << ' ' << time << '\n';
		HeaderFile << level_steps[level] << '\n';

		const amrex::IntVect &domain_lo = geom[level].Domain().smallEnd();
		for (int i = 0; i < bArray[level].size(); ++i) {
			// Need to shift because the RealBox ctor we call takes the
			// physical location of index (0,0,0).  This does not affect
			// the usual cases where the domain index starts with 0.
			const amrex::Box &b = amrex::shift(bArray[level][i], -domain_lo);
			amrex::RealBox const loc = amrex::RealBox(b, geom[level].CellSize(), geom[level].ProbLo());
			for (int idim = 0; idim < lowerSpaceDim; ++idim) {
				HeaderFile << loc.lo(idim) << ' ' << loc.hi(idim) << '\n';
			}
		}
		HeaderFile << amrex::MultiFabHeaderPath(level, levelPrefix, mfPrefix) << '\n';
	}
}

void DiagFramePlane::VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name)
{
	auto whichRD = amrex::FArrayBox::getDataDescriptor();
	bool const doConvert(*whichRD != amrex::FPC::NativeRealDescriptor());

	amrex::Long bytesWritten(0);

	std::string const filePrefix(a_mf_name + "_D_");

	bool const calcMinMax = false;
	amrex::VisMF::Header::Version const currentVersion = amrex::VisMF::Header::Version_v1;
	amrex::VisMF::How const how = amrex::VisMF::How::NFiles;
	amrex::VisMF::Header hdr(a_mf, how, currentVersion, calcMinMax);

	int const nOutFiles = std::max(1, std::min(amrex::ParallelDescriptor::NProcs(), 256));
	bool const groupSets = false;
	bool const setBuf = true;

	amrex::NFilesIter nfi(nOutFiles, filePrefix, groupSets, setBuf);

	// Check if mf has sparse data
	bool useSparseFPP = false;
	const amrex::Vector<int> &pmap = a_mf.DistributionMap().ProcessorMap();
	std::set<int> procsWithData;
	amrex::Vector<int> procsWithDataVector;
	for (const int i : pmap) {
		procsWithData.insert(i);
	}
	if (static_cast<int>(procsWithData.size()) < nOutFiles) {
		useSparseFPP = true;
		for (const int it : procsWithData) {
			procsWithDataVector.push_back(it);
		}
	}

	if (useSparseFPP) {
		nfi.SetSparseFPP(procsWithDataVector);
	} else {
		nfi.SetDynamic();
	}
	for (; nfi.ReadyToWrite(); ++nfi) {
		int const whichRDBytes(whichRD->numBytes());
		int nFABs(0);
		amrex::Long writeDataItems(0);
		amrex::Long writeDataSize(0);
		for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
			const amrex::FArrayBox &fab = a_mf[mfi];
			std::stringstream hss;
			write_2D_header(hss, fab, fab.nComp());
			bytesWritten += static_cast<std::streamoff>(hss.tellp());
			bytesWritten += fab.box().numPts() * a_mf.nComp() * whichRDBytes;
			++nFABs;
		}
		std::unique_ptr<std::vector<char>> allFabData{};
		bool canCombineFABs = false;
		if ((nFABs > 1 || doConvert) && amrex::VisMF::GetUseSingleWrite()) {
			allFabData = std::make_unique<std::vector<char>>(bytesWritten);
		} // ---- else { no need to make a copy for one fab }
		canCombineFABs = allFabData != nullptr;

		if (canCombineFABs) {
			amrex::Long writePosition = 0;
			for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
				int hLength(0);
				const amrex::FArrayBox &fab = a_mf[mfi];
				writeDataItems = fab.box().numPts() * a_mf.nComp();
				writeDataSize = writeDataItems * whichRDBytes;
				char *afPtr = allFabData->data() + writePosition; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				std::stringstream hss;
				write_2D_header(hss, fab, fab.nComp());
				hLength = static_cast<int>(hss.tellp());
				auto tstr = hss.str();
				std::memcpy(afPtr, tstr.c_str(), hLength); // ---- the fab header
				amrex::Real const *fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
				std::unique_ptr<amrex::FArrayBox> hostfab;
				if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
					hostfab = std::make_unique<amrex::FArrayBox>(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
					amrex::Gpu::streamSynchronize();
					fabdata = hostfab->dataPtr();
				}
#endif
				memcpy(afPtr + hLength, fabdata, writeDataSize); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				writePosition += hLength + writeDataSize;
			}
			nfi.Stream().write(allFabData->data(), bytesWritten);
			nfi.Stream().flush();
			// delete[] allFabData;
		} else {
			for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
				int hLength = 0;
				const amrex::FArrayBox &fab = a_mf[mfi];
				writeDataItems = fab.box().numPts() * a_mf.nComp();
				writeDataSize = writeDataItems * whichRDBytes;
				std::stringstream hss;
				write_2D_header(hss, fab, fab.nComp());
				hLength = static_cast<int>(hss.tellp());
				auto tstr = hss.str();
				nfi.Stream().write(tstr.c_str(), hLength); // ---- the fab header
				nfi.Stream().flush();
				amrex::Real const *fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
				std::unique_ptr<amrex::FArrayBox> hostfab;
				if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
					hostfab = std::make_unique<amrex::FArrayBox>(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
					amrex::Gpu::streamSynchronize();
					fabdata = hostfab->dataPtr();
				}
#endif
				nfi.Stream().write(reinterpret_cast<const char *>(fabdata),
						   writeDataSize); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
				nfi.Stream().flush();
			}
		}
	}

	int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber());
	if (nfi.GetDynamic()) {
		coordinatorProc = nfi.CoordinatorProc();
	}
	hdr.CalculateMinMax(a_mf, coordinatorProc);

	Find2FOffsets(a_mf, filePrefix, hdr, currentVersion, nfi, nOutFiles, amrex::ParallelDescriptor::Communicator());

	Write2DMFHeader(a_mf_name, hdr, coordinatorProc, amrex::ParallelDescriptor::Communicator());
}

void DiagFramePlane::Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm)
{
	const int myProc(amrex::ParallelDescriptor::MyProc(comm));
	if (myProc == coordinatorProc) {

		std::string MFHdrFileName(a_mf_name);

		MFHdrFileName += "_H";

		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);

		std::ofstream MFHdrFile;

		MFHdrFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		MFHdrFile.open(MFHdrFileName.c_str(), std::ios::out | std::ios::trunc);

		if (!MFHdrFile.good()) {
			amrex::FileOpenFailed(MFHdrFileName);
		}

		MFHdrFile.setf(std::ios::floatfield, std::ios::scientific);
		MFHdrFile << hdr.m_vers << '\n';
		MFHdrFile << static_cast<int>(hdr.m_how) << '\n';
		MFHdrFile << hdr.m_ncomp << '\n';
		if (hdr.m_ngrow == hdr.m_ngrow[0]) {
			MFHdrFile << hdr.m_ngrow[0] << '\n';
		} else {
			MFHdrFile << hdr.m_ngrow << '\n';
		}

		MFHdrFile << "(" << hdr.m_ba.size() << " 0" << '\n';
		for (int i = 0; i < hdr.m_ba.size(); ++i) {
			printLowerDimBox(MFHdrFile, hdr.m_ba[i], 2);
			MFHdrFile << '\n';
		}
		MFHdrFile << ") \n";

		MFHdrFile << hdr.m_fod << '\n';

		MFHdrFile << hdr.m_min.size() << "," << hdr.m_min[0].size() << '\n';
		MFHdrFile.precision(16);
		for (auto &hdr_i : hdr.m_min) {
			for (const double j : hdr_i) {
				MFHdrFile << j << ",";
			}
			MFHdrFile << "\n";
		}

		MFHdrFile << "\n";

		MFHdrFile << hdr.m_max.size() << "," << hdr.m_max[0].size() << '\n';
		for (auto &hdr_i : hdr.m_max) {
			for (const double j : hdr_i) {
				MFHdrFile << j << ",";
			}
			MFHdrFile << "\n";
		}

		MFHdrFile.flush();
		MFHdrFile.close();
	}
}

void DiagFramePlane::Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr,
				   amrex::VisMF::Header::Version /*whichVersion*/, amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm)
{
	bool const groupSets = false;

	const int myProc(amrex::ParallelDescriptor::MyProc(comm));
	const int nProcs(amrex::ParallelDescriptor::NProcs(comm));
	int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber(comm));
	if (nfi.GetDynamic()) {
		coordinatorProc = nfi.CoordinatorProc();
	}

	auto whichRD = amrex::FArrayBox::getDataDescriptor();
	int const whichRDBytes(whichRD->numBytes());
	int const nComps(mf.nComp());

	if (myProc == coordinatorProc) { // ---- calculate offsets
		const amrex::BoxArray &mfBA = mf.boxArray();
		const amrex::DistributionMapping &mfDM = mf.DistributionMap();
		amrex::Vector<amrex::Long> fabHeaderBytes(mfBA.size(), 0);
		int const nFiles(amrex::NFilesIter::ActualNFiles(nOutFiles));
		int whichFileNumber(-1);
		std::string whichFileName;
		amrex::Vector<amrex::Long> currentOffset(nProcs, 0L);

		// ---- find the length of the fab header instead of asking the file system
		for (int i(0); i < mfBA.size(); ++i) {
			std::stringstream hss;
			amrex::FArrayBox const tempFab(mf.fabbox(i), nComps, false); // ---- no alloc
			write_2D_header(hss, tempFab, tempFab.nComp());
			fabHeaderBytes[i] = static_cast<std::streamoff>(hss.tellp());
		}

		std::map<int, amrex::Vector<int>> rankBoxOrder; // ---- [rank, boxarray index array]
		for (int i(0); i < mfBA.size(); ++i) {
			rankBoxOrder[mfDM[i]].push_back(i);
		}

		amrex::Vector<int> fileNumbers;
		if (nfi.GetDynamic()) {
			fileNumbers = nfi.FileNumbersWritten();
		} else if (nfi.GetSparseFPP()) { // if sparse, write to (file number = rank)
			fileNumbers.resize(nProcs);
			for (int i(0); i < nProcs; ++i) {
				fileNumbers[i] = i;
			}
		} else {
			fileNumbers.resize(nProcs);
			for (int i(0); i < nProcs; ++i) {
				fileNumbers[i] = amrex::NFilesIter::FileNumber(nFiles, i, groupSets);
			}
		}
		const amrex::Vector<amrex::Vector<int>> &fileNumbersWriteOrder = nfi.FileNumbersWriteOrder();

		for (const auto &fn : fileNumbersWriteOrder) {
			for (const int rank : fn) {
				auto rboIter = rankBoxOrder.find(rank);

				if (rboIter != rankBoxOrder.end()) {
					amrex::Vector<int> const &index = rboIter->second;
					whichFileNumber = fileNumbers[rank];
					whichFileName = amrex::VisMF::BaseName(amrex::NFilesIter::FileName(whichFileNumber, filePrefix));

					for (const int idx : index) {
						hdr.m_fod[idx].m_name = whichFileName;
						hdr.m_fod[idx].m_head = currentOffset[whichFileNumber];
						currentOffset[whichFileNumber] += mf.fabbox(idx).numPts() * nComps * whichRDBytes + fabHeaderBytes[idx];
					}
				}
			}
		}
	}
}

void DiagFramePlane::write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar)
{
	os << "FAB " << amrex::FPC::NativeRealDescriptor();
	amrex::StreamRetry sr(os, "FABio_write_header", 4);
	while (sr.TryOutput()) {
		printLowerDimBox(os, f.box(), 2);
		os << ' ' << nvar << '\n';
	}
}
