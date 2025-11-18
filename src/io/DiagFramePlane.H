#ifndef DIAGFRAMEPLANE_H
#define DIAGFRAMEPLANE_H

#include <string>
#include <vector>

#include "AMReX_PlotFileUtil.H"
#include "AMReX_VisMF.H"
#include "DiagBase.H"

// Forward declaration
namespace quokka
{
template <typename problem_t> class PhysicsParticleRegister;
} // namespace quokka

class DiagFramePlane : public DiagBase::Register<DiagFramePlane>
{
      public:
	static auto identifier() -> std::string { return "DiagFramePlane"; }

	enum InterpType { Linear, Quadratic };

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

	void addVars(amrex::Vector<std::string> &a_varList) override;

	// Get the list of particle types to include (empty means none)
	[[nodiscard]] auto getParticleTypes() const -> const std::vector<std::string> & { return m_particleTypes; }

	void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice,
				       const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time,
				       const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref, const YAML::Node &simulationMetadata);

	static void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name);

	static void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm);

	static void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr,
				  amrex::VisMF::Header::Version /*whichVersion*/, amrex::NFilesIter &nfi, int nOutFile, MPI_Comm comm);

	static void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar);

	static void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray,
					  const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time,
					  const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio,
					  const std::string &versionName = "HyperCLaw-V1.1", const std::string &levelPrefix = "Level_",
					  const std::string &mfPrefix = "Cell");

	void close() override {}

      private:
	// Variables output
	amrex::Vector<std::string> m_fieldNames;
	amrex::Gpu::DeviceVector<int> m_fieldIndices_d;

	// Particle types to include (default: empty)
	std::vector<std::string> m_particleTypes;

	// Plane definition
	int m_normal;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_center;

	// Interpolation data
	InterpType m_interpType;
	amrex::Vector<amrex::GpuArray<amrex::Real, 3>> m_intwgt;
	amrex::Vector<int> m_k0;

	// 2D-plane boxArray vector
	amrex::Geometry m_geomLev0;
	amrex::Vector<amrex::BoxArray> m_sliceBA;
	amrex::Vector<amrex::Vector<int>> m_dmConvert;
	amrex::Vector<amrex::DistributionMapping> m_sliceDM;
};

// Template implementation
template <typename problem_t> void DiagFramePlane::processDiag(int a_nstep, const amrex::Real &a_time)
{
	BL_PROFILE("DiagFramePlane::processDiag()");

	// Access diagnostic data through protected members
	auto const &a_state = *m_diagMF;
	auto const &simulationMetadata = *m_metadata;

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
		Write2DMultiLevelPlotfile(diagfile, nlevs, GetVecOfConstPtrs(planeData), m_fieldNames, pltGeoms, a_time, step_array, ref_ratio,
					  simulationMetadata);

		// If particles are requested, write them
		if (!m_particleTypes.empty()) {
			auto *sim = getSim<problem_t>();
			sim->particleRegister_.writePlotFileFiltered(diagfile, m_particleTypes);
			amrex::Print() << "  Wrote particles to frame plane " << diagfile << "\n";
		}
	}
}

#endif
