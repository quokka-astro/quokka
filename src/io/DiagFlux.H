#ifndef DIAGFLUX_H
#define DIAGFLUX_H

#include <string>
#include <utility>
#include <vector>

#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_iMultiFab.H"
#include "DiagBase.H"
#include "surface_flux.hpp"

class DiagFlux : public DiagBase::Register<DiagFlux>
{
      public:
	static auto identifier() -> std::string { return "DiagFlux"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

      private:
	void writeFluxToFile(int a_nstep, const amrex::Real &a_time,
			     std::vector<std::pair<amrex::Real, quokka::diagnostics::SurfaceFluxes>> const &fluxes_by_radius) const;

	amrex::Vector<amrex::Real> m_radii{};
	amrex::Vector<amrex::Real> m_outputRadii{};
	std::string m_radiusLabel{"radius"};
};

template <typename problem_t> void DiagFlux::processDiag(int a_nstep, const amrex::Real &a_time)
{
	BL_PROFILE("DiagFlux::processDiag()");

#if AMREX_SPACEDIM != 3
	amrex::ignore_unused(a_nstep, a_time);
	amrex::Abort("DiagFlux requires AMREX_SPACEDIM == 3.");
#else
	auto *sim = getSim<problem_t>();
	auto const &state_cc = sim->getNewMF_cc();
	auto const &state_fc = sim->getNewMF_fc();
	auto const &a_state = *m_diagMF;
	auto const &a_stateVar = *m_diagVars;
	auto const &geoms = *m_geoms;
	auto const &ref_ratio = *m_refRatio;

	amrex::Vector<amrex::iMultiFab> flux_mask(state_cc.size());
	for (int lev = 0; lev < state_cc.size(); ++lev) {
		if (lev == state_cc.size() - 1) {
			flux_mask[lev].define(state_cc[lev].boxArray(), state_cc[lev].DistributionMap(), 1, 0);
			flux_mask[lev].setVal(1);
		} else {
			flux_mask[lev] =
			    amrex::makeFineMask(state_cc[lev], state_cc[lev + 1], amrex::IntVect(0), ref_ratio[lev], geoms[lev].periodicity(), 1, 0);
		}

		if (!m_filters.empty()) {
			auto const &sarrs = a_state[lev]->const_arrays();
			auto const &marrs = flux_mask[lev].arrays();
			auto *fdata_p = m_filterData.data();
			amrex::ParallelFor(*a_state[lev], amrex::IntVect(0),
					   [=, nFilters = m_filters.size()] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
						   if (marrs[box_no](i, j, k) == 0) {
							   return;
						   }
						   for (int f = 0; f < nFilters; ++f) {
							   amrex::Real const fval = sarrs[box_no](i, j, k, fdata_p[f].m_filterVarIdx); // NOLINT
							   if (fval < fdata_p[f].m_low_val || fval > fdata_p[f].m_high_val) {	       // NOLINT
								   marrs[box_no](i, j, k) = 0;
								   return;
							   }
						   }
					   });
		}
	}
	amrex::Gpu::streamSynchronize();

	std::vector<std::pair<amrex::Real, quokka::diagnostics::SurfaceFluxes>> fluxes_by_radius;
	fluxes_by_radius.reserve(m_radii.size());
	for (int n = 0; n < m_radii.size(); ++n) {
		auto const fluxes = quokka::diagnostics::computeSphericalSurfaceFluxes<problem_t>(state_cc, state_fc, geoms, flux_mask, m_radii[n]);
		fluxes_by_radius.emplace_back(m_outputRadii[n], fluxes);
	}

	writeFluxToFile(a_nstep, a_time, fluxes_by_radius);
	amrex::ignore_unused(a_stateVar);
#endif
}

#endif
