#ifndef MHD_SYSTEM_HPP_ // NOLINT
#define MHD_SYSTEM_HPP_

//==============================================================================
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file mhd_system.hpp
/// \brief A class for solving the resistive MHD induction equation, including Ohmic heating for non-isothermal plasmas.
///

// library headers
#include <memory>
#include <vector>

#include "AMReX.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_GpuControl.H"
#include "AMReX_MFIter.H"
#include "AMReX_ParmParse.H"

// internal headers
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"

AMREX_ENUM(EMFComputeScheme, Quokka2026, Balsara2025a, Balsara2025 = Balsara2025a, FelkerStone2018, FelkerStone2017 = FelkerStone2018); // NOLINT
// Quokka2026: work in preparation; variant of Mignone21a: Mignone & Del Zanna (2021), JCP 424:109748.
// Balsara2025a: Balsara et al. (2025a), ApJ 988:134; EMF reconstructed from cc->ec.
// Balsara2025: alias kept for backward compatibility with existing input decks.
// FelkerStone2018: Felker & Stone (2018), JCP 375:1365; uses cc v-field.
// FelkerStone2017: alias kept for backward compatibility with existing input decks.

AMREX_ENUM(EMFAvgScheme, Balsara2025b, Balsara2025 = Balsara2025b, LondrilloDelZanna2004); // NOLINT
// Balsara2025b: Balsara et al. (2025b), CAMC 7; higher-order averaging via 2D Riemann solver.
// Balsara2025: alias kept for backward compatibility with existing input decks.
// LondrilloDelZanna2004: Londrillo & Del Zanna (2004), JCP 195:17; wave-speed-weighted quadrant average.

// sign convention: this module defines emf = cross(v, b), while the papers cited use Ohm's law as emf = -cross(v, b), instead.
// every cited formula is transcribed with this sign flip baked in, so individual terms may look sign-flipped
// relative to the paper while the net dB/dt (computed in SolveInductionEqn) remains correct.

AMREX_FORCE_INLINE constexpr auto MinimumHydroRiemannGhost(bool is_mhd_enabled, EMFComputeScheme emf_compute_scheme, EMFAvgScheme emf_ave_scheme,
							   bool require_tracer_ghosts = false) -> int
{
	int nghost = require_tracer_ghosts ? 2 : 0;
	if (is_mhd_enabled) {
		if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
			nghost = std::max(nghost, 3);
		} else {
			switch (emf_ave_scheme) {
				case EMFAvgScheme::LondrilloDelZanna2004:
					nghost = std::max(nghost, 1);
					break;
				case EMFAvgScheme::Balsara2025b:
					nghost = std::max(nghost, 2);
					break;
			}
		}
	}
	return nghost;
}

/// Class for solving the MHD induction equation.
template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	static constexpr int bfield_index = Physics_Indices<problem_t>::mhdFirstIndex;

	// EMF dispatch
	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_vs_wcomp,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order, EMFAvgScheme emf_ave_scheme,
			       SlopeLimiter plm_limiter, EMFComputeScheme emf_compute_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
			       amrex::Real resistivity = 0.0);

	// EMF compute schemes
	static void ComputeEMF_FelkerStone2018(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
					       SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
					       amrex::Real resistivity = 0.0);

	static void ComputeEMF_Balsara2025a(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
					    SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
					    amrex::Real resistivity = 0.0);

	static void ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_vs_wcomp,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
					  SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
					  amrex::Real resistivity = 0.0);

	// EMF averaging schemes: per-edge bodies
	template <typename EA, typename BA>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void EMFAverageEdge_LondrilloDelZanna2004(
	    int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2, EA const &ec_a4_emf_iquad1_wcomp2,
	    EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2, BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p,
	    BA const &ec_a4_b_wcomp1_m, BA const &ec_a4_b_wcomp1_p, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0,
	    amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1, std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
	    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0,
	    amrex::Real dx_wcomp1, amrex::Real resistivity);

	template <typename EA, typename BA>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void
	EMFAverageEdge_Balsara2025b(int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2,
				    EA const &ec_a4_emf_iquad1_wcomp2, EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2,
				    BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p, BA const &ec_a4_b_wcomp1_m, BA const &ec_a4_b_wcomp1_p,
				    amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1,
				    std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
				    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
				    amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

	// level-wide EMF machinery (FelkerStone2018, Balsara2025a and Quokka2026)
	// a register value read through the same a(i, j, k) interface as an Array4
	struct EdgeValue {
		amrex::Real v;
		AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int /*i*/, int /*j*/, int /*k*/) const noexcept -> amrex::Real { return v; }
	};

	// number of cc quantities reconstructed cc->fc->ec per edge orientation by the level-wide EMF kernels:
	// FelkerStone2018 reconstructs the two v-field components perpendicular to the edge; Balsara2025a reconstructs the EMF itself.
	// (Quokka2026 has no cc->fc step: it reconstructs the fc Riemann v-field and the fc b-field fc->ec.)
	template <EMFComputeScheme CS> static constexpr int LevelEMF_NV = (CS == EMFComputeScheme::FelkerStone2018) ? 2 : 1;

	// persistent per-level scratch arrays (cell-centred BoxArray, 4 ghost cells: the integer index range [lo-4, hi+4] holds every
	// cc, fc and ec range used below for all three edge orientations, so the same arrays are reused for each orientation).
	// arrays a scheme does not use are left undefined (0 comps).
	struct EMFWorkspace {
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
		amrex::BoxArray ba;
		amrex::DistributionMapping dm;
		std::array<int, 3> ncomp{-1, -1, -1}; // comps of cc_vs, fc_vs, ec_vs
		amrex::MultiFab cc_vs;		      // 3 comps: cc v-field (FelkerStone2018) or cc EMF (Balsara2025a); unused by Quokka2026
		amrex::MultiFab fc_vs;		      // 2 * NV comps: [icomp][side] fc values (cc->fc), reused for each (wcomp0, iperm); unused by Quokka2026
		amrex::MultiFab ec_vs; // 8 * NV comps: [iperm][icomp][iface][side] ec values (fc->ec); Quokka2026: 4 comps [icomp][side] ec v-field
		amrex::MultiFab ec_bs; // 4 comps: [icomp][side] ec b-field (fc->ec)
		bool defined = false;
	};
	static auto GetEMFWorkspace(amrex::MultiFab const &cc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<int, 3> const &ncomp)
	    -> EMFWorkspace &;

	template <FluxDir DIR, int ORD, SlopeLimiter LIM>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void ReconstructCellLevelEMF(amrex::Array4<const amrex::Real> const &in,
										amrex::Array4<amrex::Real> const &out_left,
										amrex::Array4<amrex::Real> const &out_right, int i, int j, int k);

	template <int NV, FluxDir DIR, int ORD, SlopeLimiter LIM>
	static void LaunchLevelEMF_CC2FC(amrex::MultiFab const &cc_mf_iter, amrex::IntVect const &ng, EMFWorkspace &ws, int wcompA, int wcompB);

	template <int NV, FluxDir DIR, int ORD, SlopeLimiter LIM>
	static void LaunchLevelEMF_FC2EC(amrex::MultiFab const &fc_mf_cVars_face, amrex::IntVect const &ng, EMFWorkspace &ws, int iperm);

	template <FluxDir DIR, int ORD, SlopeLimiter LIM>
	static void LaunchLevelEMF_FC2EC_Quokka2026(amrex::MultiFab const &fc_mf_cVars_face, amrex::MultiFab const &fc_mf_vs_face, amrex::IntVect const &ng,
						    EMFWorkspace &ws, int icomp);

	template <EMFComputeScheme CS, EMFAvgScheme AVG>
	static void LaunchLevelEMF_Edge(amrex::MultiFab &ec_mf_emf, EMFWorkspace &ws, std::array<int, 2> const &reconstruct_dirs,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
					amrex::Real resistivity);

	template <EMFComputeScheme CS, int ORD, SlopeLimiter LIM>
	static void ComputeEMF_Level(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
				     std::array<amrex::MultiFab, AMREX_SPACEDIM> const *fcw_mf_vs_wcomp,
				     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
				     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, EMFAvgScheme emf_ave_scheme,
				     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity);

	template <EMFComputeScheme CS>
	static void ComputeEMF_LevelDispatch(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const *fcw_mf_vs_wcomp,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
					     SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
					     amrex::Real resistivity);

	// resistive corrections
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto computeResistiveEMF(amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
									    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, int i, int j, int k,
									    std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
									    amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
	    -> amrex::Real;

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, int i, int j, int k,
										 amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
										 amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
										 std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
										 amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

	static void AddResistiveEnergyFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fcw_mf_fluxes_wcomp,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity);

	// induction equation
	static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_mf_cVars_old_wcomp,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_cVars_new_wcomp,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_mf_emfs_wcomp, double dt,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp);
};

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeResistivity(int /*i*/, int /*j*/, int /*k*/, amrex::Array4<const amrex::Real> const & /*fc_a4_b_wcomp0*/,
							    amrex::Array4<const amrex::Real> const & /*fc_a4_b_wcomp1*/, amrex::Real /*dx_wcomp0*/,
							    amrex::Real /*dx_wcomp1*/) -> amrex::Real
{
	static_assert(sizeof(problem_t) == 0, "computeResistivity must be specialized in the problem file when using ResistivityModel::problem_defined");
	return 0.0;
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_vs_wcomp,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
				      EMFAvgScheme emf_ave_scheme, SlopeLimiter plm_limiter, EMFComputeScheme emf_compute_scheme,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	if (emf_compute_scheme == EMFComputeScheme::FelkerStone2018) {
		MHDSystem<problem_t>::ComputeEMF_FelkerStone2018(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
								 plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Balsara2025a) {
		MHDSystem<problem_t>::ComputeEMF_Balsara2025a(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
							      plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
		MHDSystem<problem_t>::ComputeEMF_Quokka2026(ec_mf_emfs_wcomp, fcw_mf_vs_wcomp, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
							    plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else {
		throw std::runtime_error("Unsupported EMF-scheme. Expected either FelkerStone2018, Balsara2025a, or Quokka2026.");
	}
}

// compute emf components; FelkerStone2018.
// uses cc v-field and fc b-field reconstructed to ec.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FelkerStone2018(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						      SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme,
						      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FelkerStone2018()");
	// level-wide implementation: every per-edge expression (and its operand order) is the same as in the former per-box
	// implementation, but the work runs as a few ParallelFor(MultiFab) kernels over persistent per-level scratch arrays.
	MHDSystem<problem_t>::template ComputeEMF_LevelDispatch<EMFComputeScheme::FelkerStone2018>(ec_mf_emfs_wcomp, cc_mf_cVars, nullptr, fcw_mf_cVars_wcomp,
												   fcw_mf_fspds_wcomp, reconstruction_order, plm_limiter,
												   emf_ave_scheme, dx_wcomp, resistivity);
}

// select the level-wide EMF kernels for the requested reconstruction order (and PLM limiter)
template <typename problem_t>
template <EMFComputeScheme CS>
void MHDSystem<problem_t>::ComputeEMF_LevelDispatch(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *fcw_mf_vs_wcomp,
						    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						    SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	if (reconstruction_order == 5) {
		MHDSystem<problem_t>::template ComputeEMF_Level<CS, 5, SlopeLimiter::minmod>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp, fcw_mf_cVars_wcomp,
											     fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (reconstruction_order == 3) {
		MHDSystem<problem_t>::template ComputeEMF_Level<CS, 3, SlopeLimiter::minmod>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp, fcw_mf_cVars_wcomp,
											     fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (reconstruction_order == 2) {
		switch (plm_limiter) {
			case SlopeLimiter::minmod:
				MHDSystem<problem_t>::template ComputeEMF_Level<CS, 2, SlopeLimiter::minmod>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp,
													     fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp,
													     emf_ave_scheme, dx_wcomp, resistivity);
				break;
			case SlopeLimiter::sweby:
				MHDSystem<problem_t>::template ComputeEMF_Level<CS, 2, SlopeLimiter::sweby>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp,
													    fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp,
													    emf_ave_scheme, dx_wcomp, resistivity);
				break;
			case SlopeLimiter::mc:
				MHDSystem<problem_t>::template ComputeEMF_Level<CS, 2, SlopeLimiter::mc>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp,
													 fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme,
													 dx_wcomp, resistivity);
				break;
			default:
				amrex::Abort("Invalid PLM slope limiter.");
		}
	} else if (reconstruction_order == 1) {
		MHDSystem<problem_t>::template ComputeEMF_Level<CS, 1, SlopeLimiter::minmod>(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_vs_wcomp, fcw_mf_cVars_wcomp,
											     fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else {
		amrex::Abort("Invalid reconstruction order specified! Supported orders: 1 (constant), 2 (PLM), 3 (PPM), 5 (xPPM).");
	}
}

template <typename problem_t>
auto MHDSystem<problem_t>::GetEMFWorkspace(amrex::MultiFab const &cc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					   std::array<int, 3> const &ncomp) -> EMFWorkspace &
{
	// one workspace per level (identified by its cell size); it is reallocated whenever the level's BoxArray or
	// DistributionMapping changes (i.e., on regrid), or when the EMF scheme needs different component counts (ncomp: cc_vs, fc_vs, ec_vs).
	// every array is fully overwritten (over the ranges read) before each use.
	static std::vector<std::unique_ptr<EMFWorkspace>> cache;
	static bool cleanup_registered = false;
	if (!cleanup_registered) {
		amrex::ExecOnFinalize([]() {
			cache.clear();
			cleanup_registered = false;
		});
		cleanup_registered = true;
	}

	EMFWorkspace *ws = nullptr;
	for (auto &entry : cache) {
		bool same_dx = true;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			same_dx = same_dx && (entry->dx[d] == dx[d]);
		}
		if (same_dx) {
			ws = entry.get();
			break;
		}
	}
	if (ws == nullptr) {
		cache.push_back(std::make_unique<EMFWorkspace>());
		ws = cache.back().get();
		ws->dx = dx;
	}

	if (!ws->defined || !(ws->ba == cc_mf.boxArray()) || !(ws->dm == cc_mf.DistributionMap()) || ws->ncomp != ncomp) {
		// kernels queued by a previous call may still read the old arrays
		amrex::Gpu::streamSynchronize();
		const int nghost_ws = 4;
		ws->ba = cc_mf.boxArray();
		ws->dm = cc_mf.DistributionMap();
		ws->ncomp = ncomp;
		std::array<amrex::MultiFab *, 3> const mfs = {&ws->cc_vs, &ws->fc_vs, &ws->ec_vs};
		for (int n = 0; n < 3; ++n) {
			*mfs[n] = (ncomp[n] > 0) ? amrex::MultiFab(ws->ba, ws->dm, ncomp[n], nghost_ws) : amrex::MultiFab();
		}
		ws->ec_bs = amrex::MultiFab(ws->ba, ws->dm, 4, nghost_ws);
		ws->defined = true;
	}
	return *ws;
}

template <typename problem_t>
template <FluxDir DIR, int ORD, SlopeLimiter LIM>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void MHDSystem<problem_t>::ReconstructCellLevelEMF(amrex::Array4<const amrex::Real> const &in,
										       amrex::Array4<amrex::Real> const &out_left,
										       amrex::Array4<amrex::Real> const &out_right, int i, int j, int k)
{
	// one per-cell reconstruction of a single component (n = 0, no component offsets) along DIR
	quokka::Array4View<amrex::Real const, DIR> q(in);
	quokka::Array4View<amrex::Real, DIR> leftState(out_left);
	quokka::Array4View<amrex::Real, DIR> rightState(out_right);
	if constexpr (ORD == 5) {
		MHDSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(q, leftState, rightState, 0, i, j, k, 0, 0);
	} else if constexpr (ORD == 3) {
		MHDSystem<problem_t>::template ReconstructStatesPPM<DIR>(q, leftState, rightState, 0, i, j, k, 0, 0);
	} else if constexpr (ORD == 2) {
		MHDSystem<problem_t>::template ReconstructStatesPLM<DIR, LIM>(q, leftState, rightState, 0, i, j, k);
	} else {
		MHDSystem<problem_t>::template ReconstructStatesConstant<DIR>(q, leftState, rightState, 0, i, j, k);
	}
}

// FelkerStone2018 sec. 4.1.1 (step 3) / Balsara2025a sec. 3, first half: cc->fc[DIR] for the NV cc quantities needed at the edge
// (FelkerStone2018: v-field components wcompA and wcompB; Balsara2025a: EMF component wcompA).
// outputs: ws.fc_vs comps [2 * icomp + side-of-face].
template <typename problem_t>
template <int NV, FluxDir DIR, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::LaunchLevelEMF_CC2FC(amrex::MultiFab const &cc_mf_iter, amrex::IntVect const &ng, EMFWorkspace &ws, const int wcompA,
						const int wcompB)
{
	static_assert(NV == 1 || NV == 2);
	auto const cc_a4s_vs = ws.cc_vs.const_arrays();
	auto const fc_a4s_vs = ws.fc_vs.arrays();
	amrex::ParallelFor(cc_mf_iter, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::ignore_unused(wcompB);
		// icomp = 0: cc component wcompA
		MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(cc_a4s_vs[bx], wcompA, 1),
										      amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 0, 1),
										      amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 1, 1), i, j, k);
		if constexpr (NV == 2) {
			// icomp = 1: cc component wcompB
			MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(cc_a4s_vs[bx], wcompB, 1),
											      amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 2, 1),
											      amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 3, 1), i, j, k);
		}
	});
}

// FelkerStone2018 sec. 4.1.1 (step 3) / Balsara2025a sec. 3, second half: fc->ec along DIR for the NV quantities and both face
// sides, plus (FelkerStone2018 steps 1 and 2) the b-field component normal to the face, reconstructed along DIR to the edge.
// outputs: ws.ec_vs comps [((iperm * NV + icomp) * 2 + iface) * 2 + side-of-edge], ws.ec_bs comps [2 * iperm + side-of-edge].
template <typename problem_t>
template <int NV, FluxDir DIR, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::LaunchLevelEMF_FC2EC(amrex::MultiFab const &fc_mf_cVars_face, amrex::IntVect const &ng, EMFWorkspace &ws, const int iperm)
{
	constexpr int b_idx = bfield_index;
	auto const fc_a4s_vs = ws.fc_vs.const_arrays();
	auto const fc_a4s_cVars = fc_mf_cVars_face.const_arrays();
	auto const ec_a4s_vs = ws.ec_vs.arrays();
	auto const ec_a4s_bs = ws.ec_bs.arrays();
	amrex::ParallelFor(fc_mf_cVars_face, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int icomp = 0; icomp < NV; ++icomp) {
			for (int iface = 0; iface < 2; ++iface) {
				const int c = ((iperm * NV + icomp) * 2 + iface) * 2;
				MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(
				    amrex::Array4<const amrex::Real>(fc_a4s_vs[bx], 2 * icomp + iface, 1), amrex::Array4<amrex::Real>(ec_a4s_vs[bx], c, 1),
				    amrex::Array4<amrex::Real>(ec_a4s_vs[bx], c + 1, 1), i, j, k);
			}
		}
		// b-field: icomp = iperm (component normal to this face), reconstructed along DIR
		MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(fc_a4s_cVars[bx], b_idx, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * iperm, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * iperm + 1, 1), i, j, k);
	});
}

// Quokka2026: fc->ec along DIR for the fc Riemann v-field and the fc b-field, both the component normal to face reconstruct_dirs[icomp].
// outputs: ws.ec_vs comps [2 * icomp + side-of-edge], ws.ec_bs comps [2 * icomp + side-of-edge].
template <typename problem_t>
template <FluxDir DIR, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::LaunchLevelEMF_FC2EC_Quokka2026(amrex::MultiFab const &fc_mf_cVars_face, amrex::MultiFab const &fc_mf_vs_face,
							   amrex::IntVect const &ng, EMFWorkspace &ws, const int icomp)
{
	constexpr int b_idx = bfield_index;
	auto const fc_a4s_cVars = fc_mf_cVars_face.const_arrays();
	auto const fc_a4s_vs = fc_mf_vs_face.const_arrays();
	auto const ec_a4s_vs = ws.ec_vs.arrays();
	auto const ec_a4s_bs = ws.ec_bs.arrays();
	amrex::ParallelFor(fc_mf_cVars_face, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// same order as the former per-box implementation: b-field first, then v-field
		MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(fc_a4s_cVars[bx], b_idx, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * icomp, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * icomp + 1, 1), i, j, k);
		MHDSystem<problem_t>::template ReconstructCellLevelEMF<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(fc_a4s_vs[bx], 0, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_vs[bx], 2 * icomp, 1),
										      amrex::Array4<amrex::Real>(ec_a4s_vs[bx], 2 * icomp + 1, 1), i, j, k);
	});
}

// per-edge: average the ec values over the two cc->fc->ec permutations; FelkerStone2018 then computes cross(v, b) in all four
// quadrants, while for Balsara2025a the averaged values already are the quadrant EMFs; Quokka2026 has a single fc->ec
// reconstruction and computes cross(v, b) from it directly; finally average the quadrant EMFs (+ resistive correction) into the
// edge EMF.
template <typename problem_t>
template <EMFComputeScheme CS, EMFAvgScheme AVG>
void MHDSystem<problem_t>::LaunchLevelEMF_Edge(amrex::MultiFab &ec_mf_emf, EMFWorkspace &ws, std::array<int, 2> const &reconstruct_dirs,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, const amrex::Real dx_wcomp0,
					       const amrex::Real dx_wcomp1, const amrex::Real resistivity)
{
	constexpr int b_idx = bfield_index;
	constexpr int NV = LevelEMF_NV<CS>;
	int const wcomp0_comp = reconstruct_dirs[0];
	int const wcomp1_comp = reconstruct_dirs[1];
	// indexing: delta_wcomp{0/1}[3: spatial dimension]
	std::array<int, 3> delta_wcomp0 = {0, 0, 0};
	std::array<int, 3> delta_wcomp1 = {0, 0, 0};
	delta_wcomp0[wcomp0_comp] = 1;
	delta_wcomp1[wcomp1_comp] = 1;

	auto const ec_a4s_vs = ws.ec_vs.const_arrays();
	auto const ec_a4s_bs = ws.ec_bs.const_arrays();
	auto const fc_a4s_fspds_wcomp0 = fcw_mf_fspds_wcomp[wcomp0_comp].const_arrays();
	auto const fc_a4s_fspds_wcomp1 = fcw_mf_fspds_wcomp[wcomp1_comp].const_arrays();
	auto const fc_a4s_cVars_wcomp0 = fcw_mf_cVars_wcomp[wcomp0_comp].const_arrays();
	auto const fc_a4s_cVars_wcomp1 = fcw_mf_cVars_wcomp[wcomp1_comp].const_arrays();
	auto const ec_a4s_emf = ec_mf_emf.arrays();

	amrex::ParallelFor(ec_mf_emf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::ignore_unused(fc_a4s_fspds_wcomp0, fc_a4s_fspds_wcomp1, fc_a4s_cVars_wcomp0, fc_a4s_cVars_wcomp1, delta_wcomp0, delta_wcomp1, dx_wcomp0,
				     dx_wcomp1, resistivity);
		const auto &ec_a4_vs = ec_a4s_vs[bx];
		const auto &ec_a4_bs = ec_a4s_bs[bx];

		// NV is only read inside `if constexpr` below; nvcc can't first-capture a variable from inside constexpr-if, so
		// capture it here first
		[[maybe_unused]] auto const &NV_captured = NV;
		amrex::Real emf_iquad[4];
		if constexpr (CS == EMFComputeScheme::Quokka2026) {
			// cross(v, b) at each corner, from the fc->ec reconstructed v-field and b-field
			// (v/b icomp 0: B/T side for dir-0; icomp 1: L/R side for dir-1)
			for (int iquad = 0; iquad < 4; ++iquad) {
				const int idx0 = (iquad == 0 || iquad == 3) ? 0 : 1; // B/T selector for dir-0
				const int idx1 = (iquad < 2) ? 0 : 1;		     // L/R selector for dir-1
				const amrex::Real v_wcomp0 = ec_a4_vs(i, j, k, 0 + idx0);
				const amrex::Real v_wcomp1 = ec_a4_vs(i, j, k, 2 + idx1);
				const amrex::Real b_wcomp0 = ec_a4_bs(i, j, k, 0 + idx0);
				const amrex::Real b_wcomp1 = ec_a4_bs(i, j, k, 2 + idx1);
				emf_iquad[iquad] = v_wcomp0 * b_wcomp1 - v_wcomp1 * b_wcomp0;
			}
		} else {
			// average the two ways of reconstructing cc->fc->ec, in the former setVal(0)/plus(perm 0)/plus(perm 1)/mult(0.5) order.
			// quadrants (dir-0, dir-1): 0 = (-,-), 1 = (-,+), 2 = (+,+), 3 = (+,-).
			// perm 0 feeds quadrants 0, 1, 3, 2 from (iface, side) = (0,0), (0,1), (1,0), (1,1);
			// perm 1 feeds quadrants 0, 3, 1, 2 from (iface, side) = (0,0), (0,1), (1,0), (1,1).
			amrex::Real ec_vs_icomp_iquad[NV][4];
			for (int icomp = 0; icomp < NV; ++icomp) {
				const int p0 = (0 * NV + icomp) * 4; // perm 0, + (iface * 2 + side)
				const int p1 = (1 * NV + icomp) * 4; // perm 1, + (iface * 2 + side)
				ec_vs_icomp_iquad[icomp][0] = ((0.0 + ec_a4_vs(i, j, k, p0 + 0)) + ec_a4_vs(i, j, k, p1 + 0)) * 0.5;
				ec_vs_icomp_iquad[icomp][1] = ((0.0 + ec_a4_vs(i, j, k, p0 + 1)) + ec_a4_vs(i, j, k, p1 + 2)) * 0.5;
				ec_vs_icomp_iquad[icomp][2] = ((0.0 + ec_a4_vs(i, j, k, p0 + 3)) + ec_a4_vs(i, j, k, p1 + 3)) * 0.5;
				ec_vs_icomp_iquad[icomp][3] = ((0.0 + ec_a4_vs(i, j, k, p0 + 2)) + ec_a4_vs(i, j, k, p1 + 1)) * 0.5;
			}

			if constexpr (CS == EMFComputeScheme::FelkerStone2018) {
				// FelkerStone2018 eqns. 36-37: cross(v, b) at each corner
				for (int iquad = 0; iquad < 4; ++iquad) {
					const int idx0 = (iquad == 0 || iquad == 3) ? 0 : 1; // choose from B/T for dir-0
					const int idx1 = (iquad < 2) ? 0 : 1;		     // choose from L/R for dir-1
					const amrex::Real v_wcomp0 = ec_vs_icomp_iquad[0][iquad];
					const amrex::Real v_wcomp1 = ec_vs_icomp_iquad[NV - 1][iquad];
					const amrex::Real b_wcomp0 = ec_a4_bs(i, j, k, 0 + idx0);
					const amrex::Real b_wcomp1 = ec_a4_bs(i, j, k, 2 + idx1);
					emf_iquad[iquad] = v_wcomp0 * b_wcomp1 - v_wcomp1 * b_wcomp0;
				}
			} else {
				// Balsara2025a sec. 3: the cc EMF reconstructed to ec is the quadrant EMF
				for (int iquad = 0; iquad < 4; ++iquad) {
					emf_iquad[iquad] = ec_vs_icomp_iquad[0][iquad];
				}
			}
		}

		const EdgeValue emf_iquad0{emf_iquad[0]};
		const EdgeValue emf_iquad1{emf_iquad[1]};
		const EdgeValue emf_iquad2{emf_iquad[2]};
		const EdgeValue emf_iquad3{emf_iquad[3]};
		const EdgeValue b_wcomp0_m{ec_a4_bs(i, j, k, 0)};
		const EdgeValue b_wcomp0_p{ec_a4_bs(i, j, k, 1)};
		const EdgeValue b_wcomp1_m{ec_a4_bs(i, j, k, 2)};
		const EdgeValue b_wcomp1_p{ec_a4_bs(i, j, k, 3)};
		const amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2 = ec_a4s_emf[bx];
		const amrex::Array4<const amrex::Real> fc_a4_b_wcomp0(fc_a4s_cVars_wcomp0[bx], b_idx, 1);
		const amrex::Array4<const amrex::Real> fc_a4_b_wcomp1(fc_a4s_cVars_wcomp1[bx], b_idx, 1);
		const amrex::Array4<const amrex::Real> fc_a4_fspds_wcomp0 = fc_a4s_fspds_wcomp0[bx];
		const amrex::Array4<const amrex::Real> fc_a4_fspds_wcomp1 = fc_a4s_fspds_wcomp1[bx];

		if constexpr (AVG == EMFAvgScheme::LondrilloDelZanna2004) {
			MHDSystem<problem_t>::EMFAverageEdge_LondrilloDelZanna2004(i, j, k, ec_a4_emf_ave_wcomp2, emf_iquad0, emf_iquad1, emf_iquad2,
										   emf_iquad3, b_wcomp0_m, b_wcomp0_p, b_wcomp1_m, b_wcomp1_p,
										   fc_a4_fspds_wcomp0, fc_a4_fspds_wcomp1, delta_wcomp0, delta_wcomp1,
										   fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
		} else {
			MHDSystem<problem_t>::EMFAverageEdge_Balsara2025b(i, j, k, ec_a4_emf_ave_wcomp2, emf_iquad0, emf_iquad1, emf_iquad2, emf_iquad3,
									  b_wcomp0_m, b_wcomp0_p, b_wcomp1_m, b_wcomp1_p, fc_a4_fspds_wcomp0,
									  fc_a4_fspds_wcomp1, delta_wcomp0, delta_wcomp1, fc_a4_b_wcomp0, fc_a4_b_wcomp1,
									  dx_wcomp0, dx_wcomp1, resistivity);
		}
	});
}

template <typename problem_t>
template <EMFComputeScheme CS, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::ComputeEMF_Level(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const *fcw_mf_vs_wcomp,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, EMFAvgScheme emf_ave_scheme,
					    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	constexpr int NV = LevelEMF_NV<CS>;
	const int nghost_cc = 4; // 4 cc ghost cells needed for cc->fc->ec PPM reconstruction
	// note: all centerings share the same distribution mapping and box ordering, so one box index addresses all of them
	constexpr bool is_q26 = (CS == EMFComputeScheme::Quokka2026);
	const std::array<int, 3> ncomp = is_q26 ? std::array<int, 3>{0, 0, 4} : std::array<int, 3>{3, 2 * NV, 8 * NV};
	EMFWorkspace &ws = MHDSystem<problem_t>::GetEMFWorkspace(cc_mf_cVars, dx_wcomp, ncomp);
	amrex::ignore_unused(nghost_cc, fcw_mf_vs_wcomp); // nghost_cc is unused by Quokka2026, fcw_mf_vs_wcomp is used only by Quokka2026

	if constexpr (CS == EMFComputeScheme::Quokka2026) {
		// Quokka2026 uses the fc Riemann v-field directly: nothing to compute at cell centres
		AMREX_ASSERT(fcw_mf_vs_wcomp != nullptr);
	} else if constexpr (CS == EMFComputeScheme::FelkerStone2018) {
		// extract cc v-fields over the cc box grown by nghost_cc
		auto const cc_a4s_cVars = cc_mf_cVars.const_arrays();
		auto const cc_a4s_vs = ws.cc_vs.arrays();
		amrex::ParallelFor(cc_mf_cVars, amrex::IntVect(nghost_cc), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const auto &cc_a4_cVars = cc_a4s_cVars[bx];
			const auto &cc_a4_vs = cc_a4s_vs[bx];
			const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
			const auto p_wcomp0 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const auto p_wcomp1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const auto p_wcomp2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			cc_a4_vs(i, j, k, 0) = p_wcomp0 / rho;
			cc_a4_vs(i, j, k, 1) = p_wcomp1 / rho;
			cc_a4_vs(i, j, k, 2) = p_wcomp2 / rho;
		});
	} else {
		// Balsara2025a sec. 3: cc EMF = cross(v, b), with the fc b-field averaged to the cell centre, over the cc box grown by
		// nghost_cc; then the ghost cells shared with other boxes on this level are overwritten by their valid values.
		auto const cc_a4s_cVars = cc_mf_cVars.const_arrays();
		auto const fc_a4s_cVars_wcomp0 = fcw_mf_cVars_wcomp[0].const_arrays();
		auto const fc_a4s_cVars_wcomp1 = fcw_mf_cVars_wcomp[1].const_arrays();
		auto const fc_a4s_cVars_wcomp2 = fcw_mf_cVars_wcomp[2].const_arrays();
		auto const cc_a4s_emf = ws.cc_vs.arrays();
		amrex::ParallelFor(cc_mf_cVars, amrex::IntVect(nghost_cc), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const auto &cc_a4_cVars = cc_a4s_cVars[bx];
			const auto &cc_a4_emf = cc_a4s_emf[bx];
			// fc b-field Array4 views; indexing: field[3: fc-normal direction = field component]
			const std::array<amrex::Array4<amrex::Real const>, 3> fc_a4_bs_wcomp = {
			    amrex::Array4<amrex::Real const>(fc_a4s_cVars_wcomp0[bx], bfield_index, 1),
			    amrex::Array4<amrex::Real const>(fc_a4s_cVars_wcomp1[bx], bfield_index, 1),
			    amrex::Array4<amrex::Real const>(fc_a4s_cVars_wcomp2[bx], bfield_index, 1)};
			const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
			// indexing: vs_wcomp[3: world direction = v-field component]
			std::array<amrex::Real, 3> vs_wcomp = {cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho,
							       cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho,
							       cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho};
			// emf component for each dimension
			for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
				int const wcomp1 = (wcomp0 + 1) % 3;
				int const wcomp2 = (wcomp0 + 2) % 3;

				// indexing: delta_wcomp{1/2}[3: spatial dimension]
				std::array<int, 3> delta_wcomp1 = {0, 0, 0};
				std::array<int, 3> delta_wcomp2 = {0, 0, 0};
				delta_wcomp1[wcomp1] = 1;
				delta_wcomp2[wcomp2] = 1;

				// Balsara2025a sec. 3: average fc b-field to cell center.
				amrex::Real const b_ave_wcomp1 = 0.5 * (fc_a4_bs_wcomp[wcomp1](i, j, k) +
									fc_a4_bs_wcomp[wcomp1](i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2]));
				amrex::Real const b_ave_wcomp2 = 0.5 * (fc_a4_bs_wcomp[wcomp2](i, j, k) +
									fc_a4_bs_wcomp[wcomp2](i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2]));

				// Balsara2025a sec. 3
				cc_a4_emf(i, j, k, wcomp0) = vs_wcomp[wcomp1] * b_ave_wcomp2 - vs_wcomp[wcomp2] * b_ave_wcomp1;
			}
		});
		ws.cc_vs.FillBoundary(); // fill ghost cells (same-level only, as in the former per-box implementation)
	}

	// loop (implicitly) over the 3 edge orientations (indexed by wcomp0); the scratch arrays are reused for each orientation
	// (kernels on one stream run in order).
	for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
		// right-hand-rule: dirs perpendicular to wcomp0.
		const std::array<int, 2> reconstruct_dirs = {(wcomp0 + 1) % 3, (wcomp0 + 2) % 3};
		// cc components reconstructed cc->fc->ec: v-field components reconstruct_dirs (FelkerStone2018) or the EMF wcomp0 (Balsara2025a)
		const int wcompA = (NV == 2) ? reconstruct_dirs[0] : wcomp0;
		const int wcompB = (NV == 2) ? reconstruct_dirs[1] : wcomp0;
		amrex::ignore_unused(wcompA, wcompB); // unused by Quokka2026

		if constexpr (CS == EMFComputeScheme::Quokka2026) {
			// single reconstruction fc->ec of the v-field and b-field components normal to face reconstruct_dirs[icomp]
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int wcomp = reconstruct_dirs[icomp];
				const int reconstruct_dir2edge = reconstruct_dirs[(icomp + 1) % 2];
				// cell range = fc[wcomp] box grown by 1 along dir2edge
				amrex::IntVect ng_fc(0);
				ng_fc[reconstruct_dir2edge] = 1;
				amrex::MultiFab const &fc_mf_cVars_face = fcw_mf_cVars_wcomp[wcomp];
				amrex::MultiFab const &fc_mf_vs_face = (*fcw_mf_vs_wcomp)[wcomp];
				switch (reconstruct_dir2edge) {
					case 0:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC_Quokka2026<FluxDir::X1, ORD, LIM>(
						    fc_mf_cVars_face, fc_mf_vs_face, ng_fc, ws, icomp);
						break;
					case 1:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC_Quokka2026<FluxDir::X2, ORD, LIM>(
						    fc_mf_cVars_face, fc_mf_vs_face, ng_fc, ws, icomp);
						break;
					default:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC_Quokka2026<FluxDir::X3, ORD, LIM>(
						    fc_mf_cVars_face, fc_mf_vs_face, ng_fc, ws, icomp);
						break;
				}
			}
		} else {
			// FelkerStone2018 sec. 4.1.1 / Balsara2025a sec. 3: two permutations for reconstructing cc->fc->ec; iperm selects the face direction.
			for (int iperm = 0; iperm < 2; ++iperm) {
				const int reconstruct_dir2face = reconstruct_dirs[iperm];
				const int reconstruct_dir2edge = reconstruct_dirs[(iperm + 1) % 2];

				// cc->fc: cell range = cc box grown by 1 along dir2face (PPM writes one interface outside) and by
				// (nghost_cc - 1) along dir2edge (the fc strip needed by the follow-up fc->ec reconstruction)
				amrex::IntVect ng_cc(0);
				ng_cc[reconstruct_dir2face] = 1;
				ng_cc[reconstruct_dir2edge] = nghost_cc - 1;
				switch (reconstruct_dir2face) {
					case 0:
						MHDSystem<problem_t>::template LaunchLevelEMF_CC2FC<NV, FluxDir::X1, ORD, LIM>(cc_mf_cVars, ng_cc, ws, wcompA,
															       wcompB);
						break;
					case 1:
						MHDSystem<problem_t>::template LaunchLevelEMF_CC2FC<NV, FluxDir::X2, ORD, LIM>(cc_mf_cVars, ng_cc, ws, wcompA,
															       wcompB);
						break;
					default:
						MHDSystem<problem_t>::template LaunchLevelEMF_CC2FC<NV, FluxDir::X3, ORD, LIM>(cc_mf_cVars, ng_cc, ws, wcompA,
															       wcompB);
						break;
				}

				// fc->ec: cell range = fc[dir2face] box grown by 1 along dir2edge
				amrex::IntVect ng_fc(0);
				ng_fc[reconstruct_dir2edge] = 1;
				amrex::MultiFab const &fc_mf_cVars_face = fcw_mf_cVars_wcomp[reconstruct_dir2face];
				switch (reconstruct_dir2edge) {
					case 0:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC<NV, FluxDir::X1, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws,
															       iperm);
						break;
					case 1:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC<NV, FluxDir::X2, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws,
															       iperm);
						break;
					default:
						MHDSystem<problem_t>::template LaunchLevelEMF_FC2EC<NV, FluxDir::X3, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws,
															       iperm);
						break;
				}
			}
		}
		if (emf_ave_scheme == EMFAvgScheme::LondrilloDelZanna2004) {
			MHDSystem<problem_t>::template LaunchLevelEMF_Edge<CS, EMFAvgScheme::LondrilloDelZanna2004>(
			    ec_mf_emfs_wcomp[wcomp0], ws, reconstruct_dirs, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, dx_wcomp[reconstruct_dirs[0]],
			    dx_wcomp[reconstruct_dirs[1]], resistivity);
		} else if (emf_ave_scheme == EMFAvgScheme::Balsara2025b) {
			MHDSystem<problem_t>::template LaunchLevelEMF_Edge<CS, EMFAvgScheme::Balsara2025b>(
			    ec_mf_emfs_wcomp[wcomp0], ws, reconstruct_dirs, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, dx_wcomp[reconstruct_dirs[0]],
			    dx_wcomp[reconstruct_dirs[1]], resistivity);
		} else {
			amrex::Abort("Unknown EMF averaging type");
		}
	}
}

// compute emf components; Quokka (2026).
// uses fc Riemann v-field and fc b-field reconstructed to ec.
// note: Mignone21a: Mignone & Del Zanna (2021), JCP 424:109748; sec. 4.2/5 describes a similar
// single-reconstruction approach based on using the fc v-field from the Riemann solver.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_vs_wcomp,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						 SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
						 amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_Quokka2026()");
	// level-wide implementation (shared with FelkerStone2018 and Balsara2025a): every per-edge expression (and its operand order)
	// is the same as in the former per-box implementation, but the work runs as a few ParallelFor(MultiFab) kernels over
	// persistent per-level scratch arrays (ec v-field and b-field only). the workspace is keyed on the level's cell-centred
	// BoxArray, recovered from the fc b-field.
	const amrex::BoxArray ba_cc = amrex::convert(fcw_mf_cVars_wcomp[0].boxArray(), amrex::IntVect::TheCellVector());
	const amrex::MultiFab cc_mf_layout(ba_cc, fcw_mf_cVars_wcomp[0].DistributionMap(), 1, 0, amrex::MFInfo().SetAlloc(false));
	MHDSystem<problem_t>::template ComputeEMF_LevelDispatch<EMFComputeScheme::Quokka2026>(ec_mf_emfs_wcomp, cc_mf_layout, &fcw_mf_vs_wcomp,
											      fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
											      plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
}

// compute emf components; Balsara2025a.
// uses cc v-field and fc b-field averaged to cc to compute the emf, then reconstructs it cc->ec.
// note Balsara2025b sec. 4 (steps 2-3) describes a similar but more elaborate AFD-WENO version of this
// b-field fc->cc, then cc->ec procedure; the following instead follows Balsara2025a's simpler approach.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Balsara2025a(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						   SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
						   amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_Balsara2025a()");
	// level-wide implementation (shared with FelkerStone2018): every per-edge expression (and its operand order) is the same as in
	// the former per-box implementation, but the work runs as a few ParallelFor(MultiFab) kernels over persistent per-level
	// scratch arrays. the cc EMF is reconstructed cc->fc->ec; the fc b-field is also moved fc->ec for the averaging schemes
	// (this reconstruction is not used by the Balsara2025a EMF formula itself, which only needs the cc b-field).
	MHDSystem<problem_t>::template ComputeEMF_LevelDispatch<EMFComputeScheme::Balsara2025a>(ec_mf_emfs_wcomp, cc_mf_cVars, nullptr, fcw_mf_cVars_wcomp,
												fcw_mf_fspds_wcomp, reconstruction_order, plm_limiter,
												emf_ave_scheme, dx_wcomp, resistivity);
}

// average emf components; LondrilloDelZanna2004, eqn. 56.
// uses fast MHD wave speeds to weight the quadrant average.
// note FelkerStone2018 implements this as their eqn. 41

// per-edge body, called from the level-wide EMF kernel (LaunchLevelEMF_Edge).
// EA/BA are accessors called as a(i, j, k); the level-wide kernel passes register values (EdgeValue).
template <typename problem_t>
template <typename EA, typename BA>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void MHDSystem<problem_t>::EMFAverageEdge_LondrilloDelZanna2004(
    int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2, EA const &ec_a4_emf_iquad1_wcomp2,
    EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2, BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p, BA const &ec_a4_b_wcomp1_m,
    BA const &ec_a4_b_wcomp1_p, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1,
    std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	{
		// LondrilloDelZanna2004 eqn. 56, alpha{1/2}^{m/p}: note that unlike in Balsara2025b, these are
		// absolute wave-speed magnitudes, so no negation is needed.
		const double max_fspd_wcomp0_m =
		    std::max(fc_a4_fspds_wcomp0(i, j, k, 0), fc_a4_fspds_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 0));
		const double max_fspd_wcomp0_p =
		    std::max(fc_a4_fspds_wcomp0(i, j, k, 1), fc_a4_fspds_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 1));
		const double max_fspd_wcomp1_m =
		    std::max(fc_a4_fspds_wcomp1(i, j, k, 0), fc_a4_fspds_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 0));
		const double max_fspd_wcomp1_p =
		    std::max(fc_a4_fspds_wcomp1(i, j, k, 1), fc_a4_fspds_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 1));

		const double emf_iquad0_wcomp2 = ec_a4_emf_iquad0_wcomp2(i, j, k);
		const double emf_iquad1_wcomp2 = ec_a4_emf_iquad1_wcomp2(i, j, k);
		const double emf_iquad2_wcomp2 = ec_a4_emf_iquad2_wcomp2(i, j, k);
		const double emf_iquad3_wcomp2 = ec_a4_emf_iquad3_wcomp2(i, j, k);

		const double b_T_wcomp0 = ec_a4_b_wcomp0_p(i, j, k);
		const double b_B_wcomp0 = ec_a4_b_wcomp0_m(i, j, k);
		const double b_R_wcomp1 = ec_a4_b_wcomp1_p(i, j, k);
		const double b_L_wcomp1 = ec_a4_b_wcomp1_m(i, j, k);
		// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
		// (-,+) | (+,+)
		//   1   |   2
		// ------+------
		//   0   |   3
		// (-,-) | (+,-)

		// LondrilloDelZanna2004 eqn. 56, numerator: a weighted sum of two different ways to group and average
		// the four corner EMFs. num1 and num2 compute the same value, but change the order of the summed
		// elements, so that averaging them gives exact floating-point symmetry.
		const double num1 =
		    ((max_fspd_wcomp0_p * max_fspd_wcomp1_p) * emf_iquad0_wcomp2 + (max_fspd_wcomp0_m * max_fspd_wcomp1_p) * emf_iquad3_wcomp2) +
		    ((max_fspd_wcomp0_p * max_fspd_wcomp1_m) * emf_iquad1_wcomp2 + (max_fspd_wcomp0_m * max_fspd_wcomp1_m) * emf_iquad2_wcomp2);
		const double num2 =
		    ((max_fspd_wcomp0_p * max_fspd_wcomp1_p) * emf_iquad0_wcomp2 + (max_fspd_wcomp0_p * max_fspd_wcomp1_m) * emf_iquad1_wcomp2) +
		    ((max_fspd_wcomp0_m * max_fspd_wcomp1_p) * emf_iquad3_wcomp2 + (max_fspd_wcomp0_m * max_fspd_wcomp1_m) * emf_iquad2_wcomp2);

		// averaged for exact floating-point symmetry
		const double numerator = 0.5 * (num1 + num2);
		// LondrilloDelZanna2004 eqn. 56, denominator
		const double denominator = (max_fspd_wcomp0_m + max_fspd_wcomp0_p) * (max_fspd_wcomp1_m + max_fspd_wcomp1_p);

		// LondrilloDelZanna2004 eqn. 56, dissipative correction term: both terms below have the opposite sign
		// to the paper's own formula, consistent with this module's emf sign convention.
		const double term2 = ((max_fspd_wcomp1_m * max_fspd_wcomp1_p) / (max_fspd_wcomp1_m + max_fspd_wcomp1_p)) * (b_T_wcomp0 - b_B_wcomp0) +
				     ((max_fspd_wcomp0_m * max_fspd_wcomp0_p) / (max_fspd_wcomp0_m + max_fspd_wcomp0_p)) * (b_L_wcomp1 - b_R_wcomp1);

		// LondrilloDelZanna2004 eqn. 56 (FelkerStone2018 eqn. 41)
		ec_a4_emf_ave_wcomp2(i, j, k) = (numerator / denominator) + term2;
		MHDSystem<problem_t>::ApplyResistiveCorrection(ec_a4_emf_ave_wcomp2, i, j, k, fc_a4_b_wcomp0, fc_a4_b_wcomp1, delta_wcomp0, delta_wcomp1,
							       dx_wcomp0, dx_wcomp1, resistivity);
	}
}

// average emf components; Balsara2025b.
// uses a 2D Riemann solver to average the EMF quadrants.
// note Balsara2025a sec. 3 recounts the same derivation (eqns. 3.2-3.10).

// per-edge body, called from the level-wide EMF kernel (LaunchLevelEMF_Edge); see EMFAverageEdge_LondrilloDelZanna2004.
template <typename problem_t>
template <typename EA, typename BA>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void MHDSystem<problem_t>::EMFAverageEdge_Balsara2025b(
    int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2, EA const &ec_a4_emf_iquad1_wcomp2,
    EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2, BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p, BA const &ec_a4_b_wcomp1_m,
    BA const &ec_a4_b_wcomp1_p, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1,
    std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	{
		// Balsara2025b sec. 3.1 (Balsara2025a sec. 3); unlike in FelkerStone2018, these wave speeds are signed,
		// so note the negation
		const double max_fspd_wcomp0_m =
		    -std::max(fc_a4_fspds_wcomp0(i, j, k, 0), fc_a4_fspds_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 0));
		const double max_fspd_wcomp0_p =
		    std::max(fc_a4_fspds_wcomp0(i, j, k, 1), fc_a4_fspds_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 1));
		const double max_fspd_wcomp1_m =
		    -std::max(fc_a4_fspds_wcomp1(i, j, k, 0), fc_a4_fspds_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 0));
		const double max_fspd_wcomp1_p =
		    std::max(fc_a4_fspds_wcomp1(i, j, k, 1), fc_a4_fspds_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 1));

		const auto emf_LB_wcomp2 = ec_a4_emf_iquad0_wcomp2(i, j, k);
		const auto emf_LT_wcomp2 = ec_a4_emf_iquad1_wcomp2(i, j, k);
		const auto emf_RT_wcomp2 = ec_a4_emf_iquad2_wcomp2(i, j, k);
		const auto emf_RB_wcomp2 = ec_a4_emf_iquad3_wcomp2(i, j, k);

		// b_T/b_B and b_R/b_L are assigned opposite to Balsara2025b's geometric convention (sec. 3.1; also in Balsara2025a
		// sec. 3), which is required to stay consistent with this module's emf sign convention.
		const auto b_T_wcomp0 = ec_a4_b_wcomp0_m(i, j, k);
		const auto b_B_wcomp0 = ec_a4_b_wcomp0_p(i, j, k);
		const auto b_R_wcomp1 = ec_a4_b_wcomp1_m(i, j, k);
		const auto b_L_wcomp1 = ec_a4_b_wcomp1_p(i, j, k);

		const auto max_fspd = std::max({max_fspd_wcomp0_m, max_fspd_wcomp0_p, max_fspd_wcomp1_m, max_fspd_wcomp1_p});

		double emf_T_star_wcomp2 = 0.0;
		double emf_B_star_wcomp2 = 0.0;
		double b_dstar_wcomp1 = 0.0;
		double emf_R_star_wcomp2 = 0.0;
		double emf_L_star_wcomp2 = 0.0;
		double b_dstar_wcomp0 = 0.0;
		double emf_dstar_wcomp2 = 0.0;

		if (max_fspd_wcomp0_m != max_fspd_wcomp0_p && max_fspd_wcomp1_m != max_fspd_wcomp1_p) {
			// Balsara2025b eqn. 7HLL (Balsara2025a eqn. 3.2): dir-0 HLL star states.
			emf_T_star_wcomp2 = (max_fspd_wcomp0_p * emf_LT_wcomp2 - max_fspd_wcomp0_m * emf_RT_wcomp2) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m) -
					    (max_fspd_wcomp0_p * max_fspd_wcomp0_m) * (b_R_wcomp1 - b_L_wcomp1) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m);
			emf_B_star_wcomp2 = (max_fspd_wcomp0_p * emf_LB_wcomp2 - max_fspd_wcomp0_m * emf_RB_wcomp2) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m) -
					    (max_fspd_wcomp0_p * max_fspd_wcomp0_m) * (b_R_wcomp1 - b_L_wcomp1) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m);
			// Balsara2025b eqn. 8HLL (Balsara2025a eqn. 3.4): dir-1 HLL star states.
			emf_R_star_wcomp2 = (max_fspd_wcomp1_p * emf_RB_wcomp2 - max_fspd_wcomp1_m * emf_RT_wcomp2) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m) +
					    (max_fspd_wcomp1_p * max_fspd_wcomp1_m) * (b_T_wcomp0 - b_B_wcomp0) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m);
			emf_L_star_wcomp2 = (max_fspd_wcomp1_p * emf_LB_wcomp2 - max_fspd_wcomp1_m * emf_LT_wcomp2) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m) +
					    (max_fspd_wcomp1_p * max_fspd_wcomp1_m) * (b_T_wcomp0 - b_B_wcomp0) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m);
			// Balsara2025b eqns. 16-17 (Balsara2025a eqn. 3.6): double-star b-field states; the four-corner sum
			// pairs rot180 partners (LB,RT)/(LT,RB), for floating-point associativity symmetry (cf. ld04 num1/num2).
			b_dstar_wcomp0 = (max_fspd_wcomp1_p * b_T_wcomp0 - max_fspd_wcomp1_m * b_B_wcomp0) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m) +
					 ((emf_LB_wcomp2 - emf_RT_wcomp2) + (emf_RB_wcomp2 - emf_LT_wcomp2)) / (2.0 * (max_fspd_wcomp1_p - max_fspd_wcomp1_m));
			b_dstar_wcomp1 = (max_fspd_wcomp0_p * b_R_wcomp1 - max_fspd_wcomp0_m * b_L_wcomp1) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m) +
					 ((emf_RT_wcomp2 - emf_LB_wcomp2) + (emf_RB_wcomp2 - emf_LT_wcomp2)) / (2.0 * (max_fspd_wcomp0_p - max_fspd_wcomp0_m));
			// Balsara2025b eqns. 18 and 19 (Balsara2025a eqns. 3.7 for dir-0 flux and 3.8 for dir-1 flux); emf_dstar_wcomp2 = average of both.
			const auto emf_dstar_1_wcomp2 =
			    -(max_fspd_wcomp0_p + max_fspd_wcomp0_m) * b_dstar_wcomp1 / 2.0 +
			    (max_fspd_wcomp1_p * (emf_LB_wcomp2 + emf_RB_wcomp2) - max_fspd_wcomp1_m * (emf_LT_wcomp2 + emf_RT_wcomp2)) /
				(2.0 * (max_fspd_wcomp1_p - max_fspd_wcomp1_m)) -
			    max_fspd_wcomp1_p * max_fspd_wcomp1_m * (b_B_wcomp0 - b_T_wcomp0) / (max_fspd_wcomp1_p - max_fspd_wcomp1_m) +
			    (max_fspd_wcomp0_p * b_R_wcomp1 + max_fspd_wcomp0_m * b_L_wcomp1) / 2.0;
			const auto emf_dstar_2_wcomp2 =
			    (max_fspd_wcomp1_p + max_fspd_wcomp1_m) * b_dstar_wcomp0 / 2.0 +
			    (max_fspd_wcomp0_p * (emf_LB_wcomp2 + emf_LT_wcomp2) - max_fspd_wcomp0_m * (emf_RB_wcomp2 + emf_RT_wcomp2)) /
				(2.0 * (max_fspd_wcomp0_p - max_fspd_wcomp0_m)) -
			    (max_fspd_wcomp1_p * b_T_wcomp0 + max_fspd_wcomp1_m * b_B_wcomp0) / 2.0 -
			    max_fspd_wcomp0_p * max_fspd_wcomp0_m * (b_R_wcomp1 - b_L_wcomp1) / (max_fspd_wcomp0_p - max_fspd_wcomp0_m);
			emf_dstar_wcomp2 = 0.5 * (emf_dstar_1_wcomp2 + emf_dstar_2_wcomp2);
		} else {
			// LLF fallback: used when max_fspd_wcomp0_m=max_fspd_wcomp0_p or max_fspd_wcomp1_m=max_fspd_wcomp1_p (HLL denominator vanishes).
			// Balsara2025b eqns. 7LLF and 8LLF (Balsara2025a eqns. 3.3 for dir-0 and 3.5 for dir-1): LLF star states.
			emf_T_star_wcomp2 = 0.5 * ((emf_LT_wcomp2 + emf_RT_wcomp2) + max_fspd * (b_R_wcomp1 - b_L_wcomp1));
			emf_B_star_wcomp2 = 0.5 * ((emf_LB_wcomp2 + emf_RB_wcomp2) + max_fspd * (b_R_wcomp1 - b_L_wcomp1));
			emf_R_star_wcomp2 = 0.5 * ((emf_RB_wcomp2 + emf_RT_wcomp2) - max_fspd * (b_T_wcomp0 - b_B_wcomp0));
			emf_L_star_wcomp2 = 0.5 * ((emf_LB_wcomp2 + emf_LT_wcomp2) - max_fspd * (b_T_wcomp0 - b_B_wcomp0));
			// Balsara2025b eqn. 12 (Balsara2025a eqn. 3.9): LLF double-star emf.
			emf_dstar_wcomp2 = 0.5 * ((emf_RT_wcomp2 + emf_LT_wcomp2 + emf_LB_wcomp2 + emf_RB_wcomp2) / 2.0 +
						  max_fspd * (b_B_wcomp0 - b_T_wcomp0 + b_R_wcomp1 - b_L_wcomp1));
		}

		// Balsara2025b eqn. 20 (Balsara2025a fig. 4): select state at the dir-2 edge based on which speeds are zero
		if (max_fspd_wcomp0_m == 0.0 && max_fspd_wcomp1_m == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_LB_wcomp2;
		} else if (max_fspd_wcomp0_p == 0.0 && max_fspd_wcomp1_m == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_RB_wcomp2;
		} else if (max_fspd_wcomp0_p == 0.0 && max_fspd_wcomp1_p == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_RT_wcomp2;
		} else if (max_fspd_wcomp0_m == 0.0 && max_fspd_wcomp1_p == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_LT_wcomp2;
		} else if (max_fspd_wcomp0_m == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_L_star_wcomp2;
		} else if (max_fspd_wcomp0_p == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_R_star_wcomp2;
		} else if (max_fspd_wcomp1_p == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_T_star_wcomp2;
		} else if (max_fspd_wcomp1_m == 0.0) {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_B_star_wcomp2;
		} else {
			ec_a4_emf_ave_wcomp2(i, j, k) = emf_dstar_wcomp2;
		}

		MHDSystem<problem_t>::ApplyResistiveCorrection(ec_a4_emf_ave_wcomp2, i, j, k, fc_a4_b_wcomp0, fc_a4_b_wcomp1, delta_wcomp0, delta_wcomp1,
							       dx_wcomp0, dx_wcomp1, resistivity);
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_mf_cVars_old_wcomp,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_cVars_new_wcomp,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_mf_emfs_wcomp, double dt,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp)
{
	const BL_PROFILE("MHDSystem::SolveInductionEqn()");
	// compute the total right-hand-side for the MOL integration

	// flux sign convention: flux_(i) is into zone i from the left; -flux_(i+1) is into zone i from the right

	// loop over faces with wcomp0-normal
	for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
		const int wcomp1 = (wcomp0 + 1) % 3;
		const int wcomp2 = (wcomp0 + 2) % 3;

		// indexing: delta_wcomp{0/1}[3: spatial dimension]
		std::array<int, 3> delta_wcomp1 = {0, 0, 0};
		std::array<int, 3> delta_wcomp2 = {0, 0, 0};
		if (wcomp0 == 0) {
			delta_wcomp1[1] = 1;
			delta_wcomp2[2] = 1;
		} else if (wcomp0 == 1) {
			delta_wcomp1[2] = 1;
			delta_wcomp2[0] = 1;
		} else if (wcomp0 == 2) {
			delta_wcomp1[0] = 1;
			delta_wcomp2[1] = 1;
		}

		auto const dx_wcomp1 = dx_wcomp[wcomp1];
		auto const dx_wcomp2 = dx_wcomp[wcomp2];
		auto const ec_emf_wcomp1 = ec_mf_emfs_wcomp[wcomp1].const_arrays();
		auto const ec_emf_wcomp2 = ec_mf_emfs_wcomp[wcomp2].const_arrays();
		auto const fc_a4_cVars_old = fc_mf_cVars_old_wcomp[wcomp0].const_arrays();
		auto fc_a4_cVars_new = fc_mf_cVars_new_wcomp[wcomp0].arrays();

		amrex::ParallelFor(fc_mf_cVars_new_wcomp[wcomp0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// the ec emfs sit in the opposite fc directions relative to the face
			const double emf_m_wcomp1 = ec_emf_wcomp1[bx](i, j, k);
			const double emf_m_wcomp2 = ec_emf_wcomp2[bx](i, j, k);
			const double emf_p_wcomp1 = ec_emf_wcomp1[bx](i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2]);
			const double emf_p_wcomp2 = ec_emf_wcomp2[bx](i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2]);
			const double db_dt = (dx_wcomp1 * (emf_m_wcomp1 - emf_p_wcomp1) + dx_wcomp2 * (emf_p_wcomp2 - emf_m_wcomp2)) / (dx_wcomp1 * dx_wcomp2);

			fc_a4_cVars_new[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) =
			    fc_a4_cVars_old[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
MHDSystem<problem_t>::computeResistiveEMF(amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, int i,
					  int j, int k, std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1, amrex::Real dx_wcomp0,
					  amrex::Real dx_wcomp1, amrex::Real resistivity) -> amrex::Real
{
	const amrex::Real ec_j = (fc_a4_b_wcomp1(i, j, k) - fc_a4_b_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2])) / dx_wcomp0 -
				 (fc_a4_b_wcomp0(i, j, k) - fc_a4_b_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2])) / dx_wcomp1;
	return resistivity * ec_j;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
MHDSystem<problem_t>::ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, int i, int j, int k,
					       amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
					       std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1, amrex::Real dx_wcomp0,
					       amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
		ec_a4_emf_ave_wcomp2(i, j, k) -=
		    computeResistiveEMF(fc_a4_b_wcomp0, fc_a4_b_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
		const amrex::Real eta = computeResistivity<problem_t>(i, j, k, fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1);
		ec_a4_emf_ave_wcomp2(i, j, k) -=
		    computeResistiveEMF(fc_a4_b_wcomp0, fc_a4_b_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, eta);
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::AddResistiveEnergyFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fcw_mf_fluxes_wcomp,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
		return;
	}
	if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::none) {
		return;
	}

	const BL_PROFILE("MHDSystem::AddResistiveEnergyFlux()");

	for (int wcomp0 = 0; wcomp0 < AMREX_SPACEDIM; ++wcomp0) {
		const int wcomp1 = (wcomp0 + 1) % 3;
		const int wcomp2 = (wcomp0 + 2) % 3;

		// indexing: delta_wcomp{0/1/2}[3: spatial dimension]
		std::array<int, 3> delta_wcomp0 = {0, 0, 0};
		std::array<int, 3> delta_wcomp1 = {0, 0, 0};
		std::array<int, 3> delta_wcomp2 = {0, 0, 0};
		delta_wcomp0[wcomp0] = 1;
		delta_wcomp1[wcomp1] = 1;
		delta_wcomp2[wcomp2] = 1;

		const amrex::Real dx_wcomp0 = dx_wcomp[wcomp0];
		const amrex::Real dx_wcomp1 = dx_wcomp[wcomp1];
		const amrex::Real dx_wcomp2 = dx_wcomp[wcomp2];
		const int energy_idx = HydroSystem<problem_t>::energy_index;

		for (amrex::MFIter mfi(fcw_mf_fluxes_wcomp[wcomp0]); mfi.isValid(); ++mfi) {
			const amrex::Box &box_face = mfi.validbox();

			// b-field Array4 aliases (no copy) for the wcomp{0/1/2}-faces
			const auto fc_a4_b_wcomp_wcomp1 = fcw_mf_cVars_wcomp[wcomp1][mfi].const_array(bfield_index);
			const auto fc_a4_b_wcomp_wcomp2 = fcw_mf_cVars_wcomp[wcomp2][mfi].const_array(bfield_index);
			const auto fc_a4_b_wcomp_wcomp0 = fcw_mf_cVars_wcomp[wcomp0][mfi].const_array(bfield_index);
			auto fc_a4_flux = fcw_mf_fluxes_wcomp[wcomp0][mfi].array();

			amrex::ParallelFor(box_face, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// odr-use every captured variable used only in the constexpr-if below, forcing nvcc to capture
				// them here: an extended __device__ lambda cannot first-capture a variable in a constexpr-if context.
				amrex::ignore_unused(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, fc_a4_b_wcomp_wcomp2, delta_wcomp0, delta_wcomp1, delta_wcomp2,
						     dx_wcomp0, dx_wcomp1, dx_wcomp2, resistivity);
				amrex::Real eta_j_wcomp1_lo = 0.0;
				amrex::Real eta_j_wcomp1_hi = 0.0;
				amrex::Real eta_j_wcomp2_lo = 0.0;
				amrex::Real eta_j_wcomp2_hi = 0.0;
				if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
					// NOLINTBEGIN(readability-suspicious-call-argument): wcomp1's current uses its transverse pair
					// (wcomp2, wcomp0), passed into these functions' generic (wcomp0, wcomp1) slots, not a swap.
					eta_j_wcomp1_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i, j, k, delta_wcomp2, delta_wcomp0,
									      dx_wcomp2, dx_wcomp0, resistivity);
					eta_j_wcomp1_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i + delta_wcomp2[0], j + delta_wcomp2[1],
								k + delta_wcomp2[2], delta_wcomp2, delta_wcomp0, dx_wcomp2, dx_wcomp0, resistivity);
					// NOLINTEND(readability-suspicious-call-argument)
					eta_j_wcomp2_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1,
									      dx_wcomp0, dx_wcomp1, resistivity);
					eta_j_wcomp2_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i + delta_wcomp1[0], j + delta_wcomp1[1],
								k + delta_wcomp1[2], delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
				} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
					// NOLINTBEGIN(readability-suspicious-call-argument): wcomp1's current uses its transverse pair
					// (wcomp2, wcomp0), passed into these functions' generic (wcomp0, wcomp1) slots, not a swap.
					const amrex::Real eta_wcomp1_lo =
					    computeResistivity<problem_t>(i, j, k, fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, dx_wcomp2, dx_wcomp0);
					eta_j_wcomp1_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i, j, k, delta_wcomp2, delta_wcomp0,
									      dx_wcomp2, dx_wcomp0, eta_wcomp1_lo);
					const amrex::Real eta_wcomp1_hi =
					    computeResistivity<problem_t>(i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2], fc_a4_b_wcomp_wcomp2,
									  fc_a4_b_wcomp_wcomp0, dx_wcomp2, dx_wcomp0);
					eta_j_wcomp1_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i + delta_wcomp2[0], j + delta_wcomp2[1],
								k + delta_wcomp2[2], delta_wcomp2, delta_wcomp0, dx_wcomp2, dx_wcomp0, eta_wcomp1_hi);
					// NOLINTEND(readability-suspicious-call-argument)
					const amrex::Real eta_wcomp2_lo =
					    computeResistivity<problem_t>(i, j, k, fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, dx_wcomp0, dx_wcomp1);
					eta_j_wcomp2_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1,
									      dx_wcomp0, dx_wcomp1, eta_wcomp2_lo);
					const amrex::Real eta_wcomp2_hi =
					    computeResistivity<problem_t>(i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2], fc_a4_b_wcomp_wcomp0,
									  fc_a4_b_wcomp_wcomp1, dx_wcomp0, dx_wcomp1);
					eta_j_wcomp2_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i + delta_wcomp1[0], j + delta_wcomp1[1],
								k + delta_wcomp1[2], delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, eta_wcomp2_hi);
				}

				// average fc b-fields across wcomp0 to ec
				const amrex::Real ave_b_wcomp2_lo =
				    0.5 * (fc_a4_b_wcomp_wcomp2(i, j, k) + fc_a4_b_wcomp_wcomp2(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2]));
				const amrex::Real ave_b_wcomp2_hi =
				    0.5 * (fc_a4_b_wcomp_wcomp2(i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2]) +
					   fc_a4_b_wcomp_wcomp2(i + delta_wcomp2[0] - delta_wcomp0[0], j + delta_wcomp2[1] - delta_wcomp0[1],
								k + delta_wcomp2[2] - delta_wcomp0[2]));
				const amrex::Real ave_b_wcomp1_lo =
				    0.5 * (fc_a4_b_wcomp_wcomp1(i, j, k) + fc_a4_b_wcomp_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2]));
				const amrex::Real ave_b_wcomp1_hi =
				    0.5 * (fc_a4_b_wcomp_wcomp1(i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2]) +
					   fc_a4_b_wcomp_wcomp1(i + delta_wcomp1[0] - delta_wcomp0[0], j + delta_wcomp1[1] - delta_wcomp0[1],
								k + delta_wcomp1[2] - delta_wcomp0[2]));

				// flux_eta is the wcomp0-component of cross(eta_j, b): each cross term (eta_j_wcomp1*b_wcomp2 and
				// eta_j_wcomp2*b_wcomp1) is independently averaged over its own two bounding edges (0.5 weight
				// each), then the two averaged terms are subtracted.
				const amrex::Real flux_eta = 0.5 * (eta_j_wcomp1_lo * ave_b_wcomp2_lo + eta_j_wcomp1_hi * ave_b_wcomp2_hi -
								    eta_j_wcomp2_lo * ave_b_wcomp1_lo - eta_j_wcomp2_hi * ave_b_wcomp1_hi);
				fc_a4_flux(i, j, k, energy_idx) += flux_eta;
			});
		}
	}
}

#endif // MHD_SYSTEM_HPP_
