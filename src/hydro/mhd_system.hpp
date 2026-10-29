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

	static void AverageEMF(amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
			       amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
			       std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
			       std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside, EMFAvgScheme emf_ave_scheme,
			       amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
			       amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

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

	static void ReconstructTo(FluxDir dir, arrayconst_t &in_state_middle, array_t &out_state_left, array_t &out_state_right,
				  const amrex::Box &box_valid_range, int reconstruction_order, SlopeLimiter plm_limiter);

	// EMF averaging schemes
	static void EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
						     amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
						     std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
						     std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside,
						     amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
						     amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
						     amrex::Real resistivity);

	static void EMFAverage_Balsara2025b(amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
					    amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
					    std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
					    std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside,
					    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
					    amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

	// per-edge bodies of the EMF averaging schemes
	template <typename EA, typename BA>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void EMFAverageEdge_LondrilloDelZanna2004(
	    int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2, EA const &ec_a4_emf_iquad1_wcomp2,
	    EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2, BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p,
	    BA const &ec_a4_b_wcomp1_m, BA const &ec_a4_b_wcomp1_p, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0,
	    amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1, std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
	    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0,
	    amrex::Real dx_wcomp1, amrex::Real resistivity);

	template <typename EA, typename BA>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void EMFAverageEdge_Balsara2025b(
	    int i, int j, int k, amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, EA const &ec_a4_emf_iquad0_wcomp2, EA const &ec_a4_emf_iquad1_wcomp2,
	    EA const &ec_a4_emf_iquad2_wcomp2, EA const &ec_a4_emf_iquad3_wcomp2, BA const &ec_a4_b_wcomp0_m, BA const &ec_a4_b_wcomp0_p,
	    BA const &ec_a4_b_wcomp1_m, BA const &ec_a4_b_wcomp1_p, amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp0,
	    amrex::Array4<const amrex::Real> const &fc_a4_fspds_wcomp1, std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
	    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0,
	    amrex::Real dx_wcomp1, amrex::Real resistivity);

	// level-wide FelkerStone2018 machinery
	// a register value read through the same a(i, j, k) interface as an Array4
	struct EdgeValue {
		amrex::Real v;
		AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int /*i*/, int /*j*/, int /*k*/) const noexcept -> amrex::Real { return v; }
	};

	// persistent per-level scratch arrays (cell-centred BoxArray, 4 ghost cells: the integer index range [lo-4, hi+4] holds every
	// cc, fc and ec range used below for all three edge orientations, so the same arrays are reused for each orientation).
	struct EMFWorkspace {
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
		amrex::BoxArray ba;
		amrex::DistributionMapping dm;
		amrex::MultiFab cc_vs; // 3 comps: cc v-field
		amrex::MultiFab fc_vs; // 4 comps: [icomp][side] fc v-field (cc->fc), reused for each (wcomp0, iperm)
		amrex::MultiFab ec_vs; // 16 comps: [iperm][icomp][iface][side] ec v-field (fc->ec)
		amrex::MultiFab ec_bs; // 4 comps: [icomp][side] ec b-field (fc->ec)
		bool defined = false;
	};
	static auto GetEMFWorkspace(amrex::MultiFab const &cc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> EMFWorkspace &;

	template <FluxDir DIR, int ORD, SlopeLimiter LIM>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void ReconstructCellFS18(amrex::Array4<const amrex::Real> const &in, amrex::Array4<amrex::Real> const &out_left,
									   amrex::Array4<amrex::Real> const &out_right, int i, int j, int k);

	template <FluxDir DIR, int ORD, SlopeLimiter LIM>
	static void LaunchFS18_CC2FC(amrex::MultiFab const &cc_mf_iter, amrex::IntVect const &ng, EMFWorkspace &ws, int wcompA, int wcompB);

	template <FluxDir DIR, int ORD, SlopeLimiter LIM>
	static void LaunchFS18_FC2EC(amrex::MultiFab const &fc_mf_cVars_face, amrex::IntVect const &ng, EMFWorkspace &ws, int iperm);

	template <EMFAvgScheme AVG>
	static void LaunchFS18_EdgeEMF(amrex::MultiFab &ec_mf_emf, EMFWorkspace &ws, std::array<int, 2> const &reconstruct_dirs,
				       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
				       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
				       amrex::Real resistivity);

	template <int ORD, SlopeLimiter LIM>
	static void ComputeEMF_FelkerStone2018_Level(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, EMFAvgScheme emf_ave_scheme,
						     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity);

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

template <typename problem_t>
void MHDSystem<problem_t>::AverageEMF(amrex::Array4<amrex::Real> const &ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
				      amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
				      std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
				      std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside, EMFAvgScheme emf_ave_scheme,
				      amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
				      amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	if (emf_ave_scheme == EMFAvgScheme::LondrilloDelZanna2004) {
		EMFAverage_LondrilloDelZanna2004(ec_a4_emf_ave_wcomp2, ec_fabs_emfs_iquad, box_ec, reconstruct_dirs, fcw_fspds_wcomp, ec_fabs_bs_icomp_jeside,
						 fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else if (emf_ave_scheme == EMFAvgScheme::Balsara2025b) {
		EMFAverage_Balsara2025b(ec_a4_emf_ave_wcomp2, ec_fabs_emfs_iquad, box_ec, reconstruct_dirs, fcw_fspds_wcomp, ec_fabs_bs_icomp_jeside,
					fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else {
		amrex::Abort("Unknown EMF averaging type");
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
	if (reconstruction_order == 5) {
		MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<5, SlopeLimiter::minmod>(
		    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (reconstruction_order == 3) {
		MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<3, SlopeLimiter::minmod>(
		    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (reconstruction_order == 2) {
		switch (plm_limiter) {
			case SlopeLimiter::minmod:
				MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<2, SlopeLimiter::minmod>(
				    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
				break;
			case SlopeLimiter::sweby:
				MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<2, SlopeLimiter::sweby>(
				    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
				break;
			case SlopeLimiter::mc:
				MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<2, SlopeLimiter::mc>(
				    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
				break;
			default:
				amrex::Abort("Invalid PLM slope limiter.");
		}
	} else if (reconstruction_order == 1) {
		MHDSystem<problem_t>::template ComputeEMF_FelkerStone2018_Level<1, SlopeLimiter::minmod>(
		    ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, emf_ave_scheme, dx_wcomp, resistivity);
	} else {
		amrex::Abort("Invalid reconstruction order specified! Supported orders: 1 (constant), 2 (PLM), 3 (PPM), 5 (xPPM).");
	}
}

template <typename problem_t>
auto MHDSystem<problem_t>::GetEMFWorkspace(amrex::MultiFab const &cc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> EMFWorkspace &
{
	// one workspace per level (identified by its cell size); it is reallocated whenever the level's BoxArray or
	// DistributionMapping changes (i.e., on regrid). every array is fully overwritten (over the ranges read) before each use.
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

	if (!ws->defined || !(ws->ba == cc_mf.boxArray()) || !(ws->dm == cc_mf.DistributionMap())) {
		// kernels queued by a previous call may still read the old arrays
		amrex::Gpu::streamSynchronize();
		const int nghost_ws = 4;
		ws->ba = cc_mf.boxArray();
		ws->dm = cc_mf.DistributionMap();
		ws->cc_vs = amrex::MultiFab(ws->ba, ws->dm, 3, nghost_ws);
		ws->fc_vs = amrex::MultiFab(ws->ba, ws->dm, 4, nghost_ws);
		ws->ec_vs = amrex::MultiFab(ws->ba, ws->dm, 16, nghost_ws);
		ws->ec_bs = amrex::MultiFab(ws->ba, ws->dm, 4, nghost_ws);
		ws->defined = true;
	}
	return *ws;
}

template <typename problem_t>
template <FluxDir DIR, int ORD, SlopeLimiter LIM>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void MHDSystem<problem_t>::ReconstructCellFS18(amrex::Array4<const amrex::Real> const &in,
										  amrex::Array4<amrex::Real> const &out_left,
										  amrex::Array4<amrex::Real> const &out_right, int i, int j, int k)
{
	// the same per-cell call that MHDSystem::ReconstructTo makes for a single component (n = 0, no component offsets)
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

// FelkerStone2018 sec. 4.1.1 (step 3), first half: cc->fc[DIR] for both v-field components that are required at the edge.
// outputs: ws.fc_vs comps [2 * icomp + side-of-face].
template <typename problem_t>
template <FluxDir DIR, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::LaunchFS18_CC2FC(amrex::MultiFab const &cc_mf_iter, amrex::IntVect const &ng, EMFWorkspace &ws, const int wcompA,
					     const int wcompB)
{
	auto const cc_a4s_vs = ws.cc_vs.const_arrays();
	auto const fc_a4s_vs = ws.fc_vs.arrays();
	amrex::ParallelFor(cc_mf_iter, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// icomp = 0: v-field component wcompA
		MHDSystem<problem_t>::template ReconstructCellFS18<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(cc_a4s_vs[bx], wcompA, 1),
										  amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 0, 1),
										  amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 1, 1), i, j, k);
		// icomp = 1: v-field component wcompB
		MHDSystem<problem_t>::template ReconstructCellFS18<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(cc_a4s_vs[bx], wcompB, 1),
										  amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 2, 1),
										  amrex::Array4<amrex::Real>(fc_a4s_vs[bx], 3, 1), i, j, k);
	});
}

// FelkerStone2018 sec. 4.1.1 (step 3), second half: fc->ec along DIR for both v-field components and both face sides, plus
// (steps 1 and 2) the b-field component normal to the face, reconstructed along DIR to the edge.
// outputs: ws.ec_vs comps [((iperm * 2 + icomp) * 2 + iface) * 2 + side-of-edge], ws.ec_bs comps [2 * iperm + side-of-edge].
template <typename problem_t>
template <FluxDir DIR, int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::LaunchFS18_FC2EC(amrex::MultiFab const &fc_mf_cVars_face, amrex::IntVect const &ng, EMFWorkspace &ws, const int iperm)
{
	constexpr int b_idx = bfield_index;
	auto const fc_a4s_vs = ws.fc_vs.const_arrays();
	auto const fc_a4s_cVars = fc_mf_cVars_face.const_arrays();
	auto const ec_a4s_vs = ws.ec_vs.arrays();
	auto const ec_a4s_bs = ws.ec_bs.arrays();
	amrex::ParallelFor(fc_mf_cVars_face, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int icomp = 0; icomp < 2; ++icomp) {
			for (int iface = 0; iface < 2; ++iface) {
				const int c = ((iperm * 2 + icomp) * 2 + iface) * 2;
				MHDSystem<problem_t>::template ReconstructCellFS18<DIR, ORD, LIM>(
				    amrex::Array4<const amrex::Real>(fc_a4s_vs[bx], 2 * icomp + iface, 1), amrex::Array4<amrex::Real>(ec_a4s_vs[bx], c, 1),
				    amrex::Array4<amrex::Real>(ec_a4s_vs[bx], c + 1, 1), i, j, k);
			}
		}
		// b-field: icomp = iperm (component normal to this face), reconstructed along DIR
		MHDSystem<problem_t>::template ReconstructCellFS18<DIR, ORD, LIM>(amrex::Array4<const amrex::Real>(fc_a4s_cVars[bx], b_idx, 1),
										  amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * iperm, 1),
										  amrex::Array4<amrex::Real>(ec_a4s_bs[bx], 2 * iperm + 1, 1), i, j, k);
	});
}

// per-edge: average the ec v-fields over the two cc->fc->ec permutations, compute cross(v, b) in all four quadrants, and
// average the quadrant EMFs (+ resistive correction) into the edge EMF.
template <typename problem_t>
template <EMFAvgScheme AVG>
void MHDSystem<problem_t>::LaunchFS18_EdgeEMF(amrex::MultiFab &ec_mf_emf, EMFWorkspace &ws, std::array<int, 2> const &reconstruct_dirs,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, const amrex::Real dx_wcomp0,
					       const amrex::Real dx_wcomp1, const amrex::Real resistivity)
{
	constexpr int b_idx = bfield_index;
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

		// average the two ways of reconstructing v-fields cc->fc->ec, in the former setVal(0)/plus(perm 0)/plus(perm 1)/mult(0.5) order.
		// quadrants (dir-0, dir-1): 0 = (-,-), 1 = (-,+), 2 = (+,+), 3 = (+,-).
		// perm 0 feeds quadrants 0, 1, 3, 2 from (iface, side) = (0,0), (0,1), (1,0), (1,1);
		// perm 1 feeds quadrants 0, 3, 1, 2 from (iface, side) = (0,0), (0,1), (1,0), (1,1).
		amrex::Real ec_vs_icomp_iquad[2][4];
		for (int icomp = 0; icomp < 2; ++icomp) {
			const int p0 = (0 * 2 + icomp) * 4; // perm 0, + (iface * 2 + side)
			const int p1 = (1 * 2 + icomp) * 4; // perm 1, + (iface * 2 + side)
			ec_vs_icomp_iquad[icomp][0] = ((0.0 + ec_a4_vs(i, j, k, p0 + 0)) + ec_a4_vs(i, j, k, p1 + 0)) * 0.5;
			ec_vs_icomp_iquad[icomp][1] = ((0.0 + ec_a4_vs(i, j, k, p0 + 1)) + ec_a4_vs(i, j, k, p1 + 2)) * 0.5;
			ec_vs_icomp_iquad[icomp][2] = ((0.0 + ec_a4_vs(i, j, k, p0 + 3)) + ec_a4_vs(i, j, k, p1 + 3)) * 0.5;
			ec_vs_icomp_iquad[icomp][3] = ((0.0 + ec_a4_vs(i, j, k, p0 + 2)) + ec_a4_vs(i, j, k, p1 + 1)) * 0.5;
		}

		// FelkerStone2018 eqns. 36-37: cross(v, b) at each corner
		amrex::Real emf_iquad[4];
		for (int iquad = 0; iquad < 4; ++iquad) {
			const int idx0 = (iquad == 0 || iquad == 3) ? 0 : 1; // choose from B/T for dir-0
			const int idx1 = (iquad < 2) ? 0 : 1;		     // choose from L/R for dir-1
			const amrex::Real v_wcomp0 = ec_vs_icomp_iquad[0][iquad];
			const amrex::Real v_wcomp1 = ec_vs_icomp_iquad[1][iquad];
			const amrex::Real b_wcomp0 = ec_a4_bs(i, j, k, 0 + idx0);
			const amrex::Real b_wcomp1 = ec_a4_bs(i, j, k, 2 + idx1);
			emf_iquad[iquad] = v_wcomp0 * b_wcomp1 - v_wcomp1 * b_wcomp0;
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
			MHDSystem<problem_t>::EMFAverageEdge_LondrilloDelZanna2004(
			    i, j, k, ec_a4_emf_ave_wcomp2, emf_iquad0, emf_iquad1, emf_iquad2, emf_iquad3, b_wcomp0_m, b_wcomp0_p, b_wcomp1_m, b_wcomp1_p,
			    fc_a4_fspds_wcomp0, fc_a4_fspds_wcomp1, delta_wcomp0, delta_wcomp1, fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
		} else {
			MHDSystem<problem_t>::EMFAverageEdge_Balsara2025b(i, j, k, ec_a4_emf_ave_wcomp2, emf_iquad0, emf_iquad1, emf_iquad2, emf_iquad3, b_wcomp0_m,
									  b_wcomp0_p, b_wcomp1_m, b_wcomp1_p, fc_a4_fspds_wcomp0, fc_a4_fspds_wcomp1, delta_wcomp0,
									  delta_wcomp1, fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
		}
	});
}

template <typename problem_t>
template <int ORD, SlopeLimiter LIM>
void MHDSystem<problem_t>::ComputeEMF_FelkerStone2018_Level(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
							    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
							    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, EMFAvgScheme emf_ave_scheme,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	const int nghost_cc = 4; // 4 cc ghost cells needed for cc->fc->ec PPM reconstruction
	// note: all centerings share the same distribution mapping and box ordering, so one box index addresses all of them
	EMFWorkspace &ws = MHDSystem<problem_t>::GetEMFWorkspace(cc_mf_cVars, dx_wcomp);

	// extract cc v-fields over the cc box grown by nghost_cc
	{
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
	}

	// loop (implicitly) over the 3 edge orientations (indexed by wcomp0); the scratch arrays are reused for each orientation
	// (kernels on one stream run in order).
	for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
		// right-hand-rule: dirs perpendicular to wcomp0.
		const std::array<int, 2> reconstruct_dirs = {(wcomp0 + 1) % 3, (wcomp0 + 2) % 3};

		// FelkerStone2018 sec. 4.1.1: two permutations for reconstructing cc->fc->ec; iperm selects the face direction.
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
					MHDSystem<problem_t>::template LaunchFS18_CC2FC<FluxDir::X1, ORD, LIM>(cc_mf_cVars, ng_cc, ws, reconstruct_dirs[0],
													      reconstruct_dirs[1]);
					break;
				case 1:
					MHDSystem<problem_t>::template LaunchFS18_CC2FC<FluxDir::X2, ORD, LIM>(cc_mf_cVars, ng_cc, ws, reconstruct_dirs[0],
													      reconstruct_dirs[1]);
					break;
				default:
					MHDSystem<problem_t>::template LaunchFS18_CC2FC<FluxDir::X3, ORD, LIM>(cc_mf_cVars, ng_cc, ws, reconstruct_dirs[0],
													      reconstruct_dirs[1]);
					break;
			}

			// fc->ec: cell range = fc[dir2face] box grown by 1 along dir2edge
			amrex::IntVect ng_fc(0);
			ng_fc[reconstruct_dir2edge] = 1;
			amrex::MultiFab const &fc_mf_cVars_face = fcw_mf_cVars_wcomp[reconstruct_dir2face];
			switch (reconstruct_dir2edge) {
				case 0:
					MHDSystem<problem_t>::template LaunchFS18_FC2EC<FluxDir::X1, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws, iperm);
					break;
				case 1:
					MHDSystem<problem_t>::template LaunchFS18_FC2EC<FluxDir::X2, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws, iperm);
					break;
				default:
					MHDSystem<problem_t>::template LaunchFS18_FC2EC<FluxDir::X3, ORD, LIM>(fc_mf_cVars_face, ng_fc, ws, iperm);
					break;
			}
		}

		if (emf_ave_scheme == EMFAvgScheme::LondrilloDelZanna2004) {
			MHDSystem<problem_t>::template LaunchFS18_EdgeEMF<EMFAvgScheme::LondrilloDelZanna2004>(
			    ec_mf_emfs_wcomp[wcomp0], ws, reconstruct_dirs, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, dx_wcomp[reconstruct_dirs[0]],
			    dx_wcomp[reconstruct_dirs[1]], resistivity);
		} else if (emf_ave_scheme == EMFAvgScheme::Balsara2025b) {
			MHDSystem<problem_t>::template LaunchFS18_EdgeEMF<EMFAvgScheme::Balsara2025b>(
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
	// note: all centerings share the same distribution mapping so looping over cc MFIter is valid.
	// note: cc, fc, and ec data have different cell counts

	// loop over each box-array on the level
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(fcw_mf_cVars_wcomp[0], amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// create a view of all the data (+ghost cells; do not make another copy of the data)
		// indexing: field[3: fc-normal direction]
		std::array<amrex::FArrayBox, 3> fcw_fabs_vs_wcomp = {
		    amrex::FArrayBox(fcw_mf_vs_wcomp[0][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcw_mf_vs_wcomp[1][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcw_mf_vs_wcomp[2][mfi], amrex::make_alias, 0, 1),
		};
		// indexing: field[3: fc-normal direction = field component]
		std::array<amrex::FArrayBox, 3> fcw_fabs_bs_wcomp = {
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};
		// compute the ec emf components
		for (int wcomp0 = 0; wcomp0 < AMREX_SPACEDIM; ++wcomp0) {
			// define the two reconstruction directions needed to get cc v-fields to ec;
			// right-hand-rule: dirs perpendicular to wcomp0.
			// indexing: reconstruct_dirs[2: reconstruction direction]
			std::array<int, 2> reconstruct_dirs = {(wcomp0 + 1) % 3, (wcomp0 + 2) % 3};
			const amrex::Box box_ec = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(reconstruct_dirs[0]) +
									     amrex::IntVect::TheDimensionVector(reconstruct_dirs[1]));
			const amrex::Box box_ec_plus1 = amrex::grow(box_ec, 1);

			// FArrayBoxes for storing the ec fields produced by reconstructing fc->ec.
			// indexing: field[2: i-component][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_vs_icomp_jeside;
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bs_icomp_jeside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jeside = 0; jeside < 2; ++jeside) {
					ec_fabs_vs_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena());
					ec_fabs_bs_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena());
				}
			}

			// reconstruct the field components that are normal to the cell-face: fc->ec
			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(reconstruct_dirs[(icomp + 1) % 2]);
				const int wcomp = reconstruct_dirs[icomp];
				// reconstruct components fc->ec
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fcw_fabs_bs_wcomp[wcomp].array(), ec_fabs_bs_icomp_jeside[icomp][0].array(),
								    ec_fabs_bs_icomp_jeside[icomp][1].array(), box_ec, reconstruction_order, plm_limiter);
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fcw_fabs_vs_wcomp[wcomp].array(), ec_fabs_vs_icomp_jeside[icomp][0].array(),
								    ec_fabs_vs_icomp_jeside[icomp][1].array(), box_ec, reconstruction_order, plm_limiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emfs_iquad;
			// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
			// |----------------------------------------------------------------------------------------|
			// |            q2                                                                          |
			// |         {v/b}0^T                 |                                                     |
			// |        \       /         q1 + q2 | q2 + q3                                             |
			// |         \     /             TL   |  TR             emf^BL = v0^B * b1^L - v1^L * b0^B  |
			// |          \   /             (-,+) | (+,+)                                               |
			// |      q1   \ /   q3               |                 emf^TL = v0^T * b1^L - v1^L * b0^T  |
			// |  {v/b}1^L . {v/b}1^R  ->  ---------------  where:                                      |
			// |           / \                    |                 emf^TR = v0^T * b1^R - v1^R * b0^T  |
			// |          /   \             (-,-) | (+,-)                                               |
			// |         /     \             BL   |  BR             emf^BR = v0^B * b1^R - v1^R * b0^B  |
			// |        /       \         q0 + q1 | q3 + q0                                             |
			// |         {v/b}0^B                 |                                                     |
			// |            q0                                                                          |
			// |----------------------------------------------------------------------------------------|
			// compute the EMF along ec using a single kernel (all quadrants inside)
			{
				// bind read/write Array4 views on the host (required for GPU lambda capture)
				// indexing: field[4: quadrant around edge]
				std::array<amrex::Array4<const amrex::Real>, 4> ec_vs_wcomp0_iquad;
				std::array<amrex::Array4<const amrex::Real>, 4> ec_vs_wcomp1_iquad;
				std::array<amrex::Array4<const amrex::Real>, 4> ec_bs_wcomp0_iquad;
				std::array<amrex::Array4<const amrex::Real>, 4> ec_bs_wcomp1_iquad;
				std::array<amrex::Array4<amrex::Real>, 4> ec_emfs_wcomp2_iquad;

				for (int iquad = 0; iquad < 4; ++iquad) {
					const int idx0 = (iquad == 0 || iquad == 3) ? 0 : 1; // B/T selector for dir-0
					const int idx1 = (iquad < 2) ? 0 : 1;		     // L/R selector for dir-1

					// define EMF FArrayBox for each quadrant (must be allocated outside the kernel)
					ec_fabs_emfs_iquad[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());

					// extract relevant v-field and b-field components (host: get Array4 views)
					ec_vs_wcomp0_iquad[iquad] = ec_fabs_vs_icomp_jeside[0][idx0].const_array(); // B/T
					ec_bs_wcomp0_iquad[iquad] = ec_fabs_bs_icomp_jeside[0][idx0].const_array(); // B/T
					ec_vs_wcomp1_iquad[iquad] = ec_fabs_vs_icomp_jeside[1][idx1].const_array(); // L/R
					ec_bs_wcomp1_iquad[iquad] = ec_fabs_bs_icomp_jeside[1][idx1].const_array(); // L/R
					ec_emfs_wcomp2_iquad[iquad] = ec_fabs_emfs_iquad[iquad].array();	    // output EMF view
				}

				// single kernel over the ec box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int iquad = 0; iquad < 4; ++iquad) {
						const amrex::Real v_wcomp0 = ec_vs_wcomp0_iquad[iquad](i, j, k);
						const amrex::Real v_wcomp1 = ec_vs_wcomp1_iquad[iquad](i, j, k);
						const amrex::Real b_wcomp0 = ec_bs_wcomp0_iquad[iquad](i, j, k);
						const amrex::Real b_wcomp1 = ec_bs_wcomp1_iquad[iquad](i, j, k);
						ec_emfs_wcomp2_iquad[iquad](i, j, k) = v_wcomp0 * b_wcomp1 - v_wcomp1 * b_wcomp0;
					}
				});
			}

			const auto &ec_a4_emf_ave_wcomp2 = ec_mf_emfs_wcomp[wcomp0][mfi].array();

			std::array<amrex::Array4<const amrex::Real>, 3> const fcw_fspds_wcomp = {
			    fcw_mf_fspds_wcomp[0].const_array(mfi), fcw_mf_fspds_wcomp[1].const_array(mfi), fcw_mf_fspds_wcomp[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(ec_a4_emf_ave_wcomp2, ec_fabs_emfs_iquad, box_ec, reconstruct_dirs, fcw_fspds_wcomp,
							 ec_fabs_bs_icomp_jeside, emf_ave_scheme,
							 fcw_mf_cVars_wcomp[reconstruct_dirs[0]][mfi].const_array(bfield_index),
							 fcw_mf_cVars_wcomp[reconstruct_dirs[1]][mfi].const_array(bfield_index), dx_wcomp[reconstruct_dirs[0]],
							 dx_wcomp[reconstruct_dirs[1]], resistivity);
		}
	}
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
	const int nghost_cc = 4;
	// note: all centerings share the same distribution mapping; looping over cc MFIter is valid
	// note: cc, fc, and ec data have different cell counts

	const auto &ba = cc_mf_cVars.boxArray();
	const auto &dm = cc_mf_cVars.DistributionMap();
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	amrex::MultiFab cc_mf_emf(ba, dm, 3, nghost_cc);
	cc_mf_emf.setVal(0.0, 0, 3, nghost_cc); // initialize everything to zero (including ghost zones)

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc_emf = mfi.growntilebox(nghost_cc); // ensure enough ghost cells for EMF computation

		// emf Array4 views for this tile
		const auto &cc_a4_emf_wcomp0 = cc_mf_emf[mfi].array(0);
		const auto &cc_a4_emf_wcomp1 = cc_mf_emf[mfi].array(1);
		const auto &cc_a4_emf_wcomp2 = cc_mf_emf[mfi].array(2);

		const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
		// indexing: field[3: field component]
		std::array<amrex::Array4<amrex::Real>, 3> const cc_a4_emfs_wcomp = {cc_a4_emf_wcomp0, cc_a4_emf_wcomp1, cc_a4_emf_wcomp2};
		// indexing: field[3: fc-normal direction = field component]
		std::array<amrex::FArrayBox, 3> fcw_fabs_bs_wcomp = {
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// fc b-field Array4 views; indexing: field[3: fc-normal direction = field component]
		std::array<amrex::Array4<amrex::Real const>, 3> fc_a4_bs_wcomp = {fcw_mf_cVars_wcomp[0][mfi].const_array(MHDSystem<problem_t>::bfield_index),
										  fcw_mf_cVars_wcomp[1][mfi].const_array(MHDSystem<problem_t>::bfield_index),
										  fcw_mf_cVars_wcomp[2][mfi].const_array(MHDSystem<problem_t>::bfield_index)};

		// compute cross(v, b) for all three dimensions
		amrex::ParallelFor(box_cc_emf, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
				cc_a4_emfs_wcomp[wcomp0](i, j, k) = vs_wcomp[wcomp1] * b_ave_wcomp2 - vs_wcomp[wcomp2] * b_ave_wcomp1;
			}
		});
	}

	cc_mf_emf.FillBoundary(); // fill ghost cells
	amrex::Gpu::streamSynchronize();

	// reconstruct emf cc->ec; also move b-field fc->ec for averaging solvers.

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// indexing: field[3: fc-normal direction = field component]
		std::array<amrex::FArrayBox, 3> fcw_fabs_bs_wcomp = {
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {

			// define the two reconstruction directions needed to get cc v-fields to ec;
			// right-hand-rule: dirs perpendicular to wcomp0.
			// indexing: reconstruct_dirs[2: reconstruction direction]
			std::array<int, 2> reconstruct_dirs = {(wcomp0 + 1) % 3, (wcomp0 + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(reconstruct_dirs[0]),
								    amrex::IntVect::TheDimensionVector(reconstruct_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_plus1 = amrex::grow(box_ec, 1);

			// initializing array to hold EMF at cell edge
			const auto &ec_a4_emf_ave_wcomp2 = ec_mf_emfs_wcomp[wcomp0][mfi].array();
			// indexing: field[2: i-side of edge]
			std::array<amrex::FArrayBox, 2> ec_fabs_emfs_ieside = {amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena()),
									       amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena())};

			ec_fabs_emfs_ieside[0].setVal<amrex::RunOn::Device>(0.0);
			ec_fabs_emfs_ieside[1].setVal<amrex::RunOn::Device>(0.0);
			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emfs_iquad;

			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emfs_iquad[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
				ec_fabs_emfs_iquad[iquad].setVal<amrex::RunOn::Device>(0.0);
			}

			// Balsara2025a sec. 3: reconstruct the cc EMF to ec.
			// there are two possible permutations for doing this:
			//   1. cc->fc[dir-0]->ec
			//   2. cc->fc[dir-1]->ec
			// note that reconstruction does not commute, so the two estimated emfs are weighted equally, and averaged below.
			for (int iperm = 0; iperm < 2; ++iperm) {
				// for each permutation of reconstructing cc->fc->ec

				const int reconstruct_dir2face = reconstruct_dirs[iperm];
				const int reconstruct_dir2edge = reconstruct_dirs[(iperm + 1) % 2];
				const auto dir2face = static_cast<FluxDir>(reconstruct_dir2face);
				const auto dir2edge = static_cast<FluxDir>(reconstruct_dir2edge);
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(reconstruct_dir2face);
				const amrex::IntVect vec_fc2ec = amrex::IntVect::TheDimensionVector(reconstruct_dir2edge);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// only keep the fc strip needed by the follow-up fc->ec reconstruction
				const amrex::Box box_fc_emf = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec);
				// PPM writes one interface outside the requested range in the reconstruction direction.
				const amrex::Box box_fc_emf_scratch = amrex::grow(box_fc_emf, vec_cc2fc);

				// reconstruct both required EMF components cc->fc->ec

				// create temporary FArrayBox for storing the fc EMF reconstructed from cc
				// indexing: field[2: i-side of face]
				std::array<amrex::FArrayBox, 2> fc_fabs_emfs_ifside = {amrex::FArrayBox(box_fc_emf_scratch, 1, amrex::The_Async_Arena()),
										       amrex::FArrayBox(box_fc_emf_scratch, 1, amrex::The_Async_Arena())};
				// reset values in temporary FArrayBox
				fc_fabs_emfs_ifside[0].setVal<amrex::RunOn::Device>(0.0);
				fc_fabs_emfs_ifside[1].setVal<amrex::RunOn::Device>(0.0);

				// reconstruct emf components cc->fc
				MHDSystem<problem_t>::ReconstructTo(dir2face, cc_mf_emf[mfi].array(wcomp0), fc_fabs_emfs_ifside[0].array(),
								    fc_fabs_emfs_ifside[1].array(), box_fc_emf, reconstruction_order, plm_limiter);

				// reconstruct emf components fc->ec
				for (int iface = 0; iface < 2; ++iface) {
					// reset values in temporary FArrayBox
					ec_fabs_emfs_ieside[0].setVal<amrex::RunOn::Device>(0.0);
					ec_fabs_emfs_ieside[1].setVal<amrex::RunOn::Device>(0.0);

					MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_emfs_ifside[iface].array(), ec_fabs_emfs_ieside[0].array(),
									    ec_fabs_emfs_ieside[1].array(), box_ec, reconstruction_order, plm_limiter);

					// figure out which ec quadrant this reconstructed emf component corresponds with
					int iquad0 = -1;
					int iquad1 = -1;

					// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
					// (-,+) | (+,+)
					//   1   |   2
					// ------+------
					//   0   |   3
					// (-,-) | (+,-)
					if (iperm == 0) {
						iquad0 = (iface == 0) ? 0 : 3;
						iquad1 = (iface == 0) ? 1 : 2;
					} else {
						iquad0 = (iface == 0) ? 0 : 1;
						iquad1 = (iface == 0) ? 3 : 2;
					}

					ec_fabs_emfs_iquad[iquad0].plus<amrex::RunOn::Device>(ec_fabs_emfs_ieside[0], 0, 0, 1);
					ec_fabs_emfs_iquad[iquad1].plus<amrex::RunOn::Device>(ec_fabs_emfs_ieside[1], 0, 0, 1);
				}
			}
			// finish averaging the two different ways for reconstructing emf: cc->fc->ec
			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emfs_iquad[iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
			}

			// indexing: field[2: i-component][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bs_icomp_jeside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jeside = 0; jeside < 2; ++jeside) {
					ec_fabs_bs_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena());
				}
			}

			// Balsara2025a sec. 3: this reconstruction is not used by the Balsara2025a EMF formula itself (which only
			// needs cc b-field); it is done here because the EMF averaging schemes need ec b-field values.
			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(reconstruct_dirs[(icomp + 1) % 2]);
				const int wcomp = reconstruct_dirs[icomp];
				// reconstruct components fc->ec
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fcw_fabs_bs_wcomp[wcomp].array(), ec_fabs_bs_icomp_jeside[icomp][0].array(),
								    ec_fabs_bs_icomp_jeside[icomp][1].array(), box_ec, reconstruction_order, plm_limiter);
			}
			std::array<amrex::Array4<const amrex::Real>, 3> const fcw_fspds_wcomp = {
			    fcw_mf_fspds_wcomp[0].const_array(mfi), fcw_mf_fspds_wcomp[1].const_array(mfi), fcw_mf_fspds_wcomp[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(ec_a4_emf_ave_wcomp2, ec_fabs_emfs_iquad, box_ec, reconstruct_dirs, fcw_fspds_wcomp,
							 ec_fabs_bs_icomp_jeside, emf_ave_scheme,
							 fcw_mf_cVars_wcomp[reconstruct_dirs[0]][mfi].const_array(bfield_index),
							 fcw_mf_cVars_wcomp[reconstruct_dirs[1]][mfi].const_array(bfield_index), dx_wcomp[reconstruct_dirs[0]],
							 dx_wcomp[reconstruct_dirs[1]], resistivity);
		}
	}
}

// average emf components; LondrilloDelZanna2004, eqn. 56.
// uses fast MHD wave speeds to weight the quadrant average.
// note FelkerStone2018 implements this as their eqn. 41

template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_LondrilloDelZanna2004(
    amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad, amrex::Box const &box_ec,
    std::array<int, 2> const &reconstruct_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
    std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
    amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_LondrilloDelZanna2004()");
	const auto &ec_a4_emf_iquad0_wcomp2 = ec_fabs_emfs_iquad[0].const_array();
	const auto &ec_a4_emf_iquad1_wcomp2 = ec_fabs_emfs_iquad[1].const_array();
	const auto &ec_a4_emf_iquad2_wcomp2 = ec_fabs_emfs_iquad[2].const_array();
	const auto &ec_a4_emf_iquad3_wcomp2 = ec_fabs_emfs_iquad[3].const_array();

	const auto &ec_a4_b_wcomp0_m = ec_fabs_bs_icomp_jeside[0][0].const_array();
	const auto &ec_a4_b_wcomp0_p = ec_fabs_bs_icomp_jeside[0][1].const_array();
	const auto &ec_a4_b_wcomp1_m = ec_fabs_bs_icomp_jeside[1][0].const_array();
	const auto &ec_a4_b_wcomp1_p = ec_fabs_bs_icomp_jeside[1][1].const_array();

	int const wcomp0_comp = reconstruct_dirs[0];
	int const wcomp1_comp = reconstruct_dirs[1];
	// indexing: delta_wcomp{0/1}[3: spatial dimension]
	std::array<int, 3> delta_wcomp0 = {0, 0, 0};
	std::array<int, 3> delta_wcomp1 = {0, 0, 0};

	delta_wcomp0[wcomp0_comp] = 1;
	delta_wcomp1[wcomp1_comp] = 1;

	const auto &fc_a4_fspds_wcomp0 = fcw_fspds_wcomp[wcomp0_comp];
	const auto &fc_a4_fspds_wcomp1 = fcw_fspds_wcomp[wcomp1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		MHDSystem<problem_t>::EMFAverageEdge_LondrilloDelZanna2004(
		    i, j, k, ec_a4_emf_ave_wcomp2, ec_a4_emf_iquad0_wcomp2, ec_a4_emf_iquad1_wcomp2, ec_a4_emf_iquad2_wcomp2, ec_a4_emf_iquad3_wcomp2,
		    ec_a4_b_wcomp0_m, ec_a4_b_wcomp0_p, ec_a4_b_wcomp1_m, ec_a4_b_wcomp1_p, fc_a4_fspds_wcomp0, fc_a4_fspds_wcomp1, delta_wcomp0, delta_wcomp1,
		    fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	});
}

// per-edge body of EMFAverage_LondrilloDelZanna2004 (shared with the level-wide FelkerStone2018 kernel).
// EA/BA are accessors called as a(i, j, k): Array4 in the per-box path, register values in the level-wide path.
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

template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_Balsara2025b(amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
						   amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
						   std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
						   std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside,
						   amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
						   amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
						   amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_Balsara2025b()");
	const auto &ec_a4_emf_iquad0_wcomp2 = ec_fabs_emfs_iquad[0].const_array();
	const auto &ec_a4_emf_iquad1_wcomp2 = ec_fabs_emfs_iquad[1].const_array();
	const auto &ec_a4_emf_iquad2_wcomp2 = ec_fabs_emfs_iquad[2].const_array();
	const auto &ec_a4_emf_iquad3_wcomp2 = ec_fabs_emfs_iquad[3].const_array();

	const auto &ec_a4_b_wcomp0_m = ec_fabs_bs_icomp_jeside[0][0].const_array();
	const auto &ec_a4_b_wcomp0_p = ec_fabs_bs_icomp_jeside[0][1].const_array();
	const auto &ec_a4_b_wcomp1_m = ec_fabs_bs_icomp_jeside[1][0].const_array();
	const auto &ec_a4_b_wcomp1_p = ec_fabs_bs_icomp_jeside[1][1].const_array();

	int const wcomp0_comp = reconstruct_dirs[0];
	int const wcomp1_comp = reconstruct_dirs[1];
	// indexing: delta_wcomp{0/1}[3: spatial dimension]
	std::array<int, 3> delta_wcomp0 = {0, 0, 0};
	std::array<int, 3> delta_wcomp1 = {0, 0, 0};

	delta_wcomp0[wcomp0_comp] = 1;
	delta_wcomp1[wcomp1_comp] = 1;

	const auto &fc_a4_fspds_wcomp0 = fcw_fspds_wcomp[wcomp0_comp];
	const auto &fc_a4_fspds_wcomp1 = fcw_fspds_wcomp[wcomp1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		MHDSystem<problem_t>::EMFAverageEdge_Balsara2025b(i, j, k, ec_a4_emf_ave_wcomp2, ec_a4_emf_iquad0_wcomp2, ec_a4_emf_iquad1_wcomp2,
								  ec_a4_emf_iquad2_wcomp2, ec_a4_emf_iquad3_wcomp2, ec_a4_b_wcomp0_m, ec_a4_b_wcomp0_p,
								  ec_a4_b_wcomp1_m, ec_a4_b_wcomp1_p, fc_a4_fspds_wcomp0, fc_a4_fspds_wcomp1, delta_wcomp0,
								  delta_wcomp1, fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	});
}

// per-edge body of EMFAverage_Balsara2025b (shared with the level-wide FelkerStone2018 kernel); see EMFAverageEdge_LondrilloDelZanna2004.
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
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &in_state_middle, array_t &out_state_left, array_t &out_state_right,
					 const amrex::Box &box_valid_range, int reconstruction_order, SlopeLimiter plm_limiter)
{
	const BL_PROFILE("MHDSystem::ReconstructTo()");
	const amrex::IntVect dir_vec = amrex::IntVect::TheDimensionVector(static_cast<int>(dir));
	// PPM kernels loop over cells and fill left(i+1) and right(i); include one extra cell in the reconstruction direction
	const amrex::Box box_cell_range = amrex::grow(amrex::enclosedCells(box_valid_range, static_cast<int>(dir)), dir_vec);
	const amrex::Box box_interface_range = amrex::surroundingNodes(box_cell_range, static_cast<int>(dir));
	if (reconstruction_order == 5) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X1>(in_state_middle, out_state_left, out_state_right,
												    box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X2>(in_state_middle, out_state_left, out_state_right,
												    box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X3>(in_state_middle, out_state_left, out_state_right,
												    box_cell_range, box_interface_range, 1);
				break;
		}
	} else if (reconstruction_order == 3) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1);
				break;
		}
	} else if (reconstruction_order == 2) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1, plm_limiter);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1, plm_limiter);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3>(in_state_middle, out_state_left, out_state_right,
												 box_cell_range, box_interface_range, 1, plm_limiter);
				break;
		}
	} else if (reconstruction_order == 1) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(in_state_middle, out_state_left, out_state_right,
												      box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(in_state_middle, out_state_left, out_state_right,
												      box_cell_range, box_interface_range, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(in_state_middle, out_state_left, out_state_right,
												      box_cell_range, box_interface_range, 1);
				break;
		}
	} else {
		amrex::Abort("Invalid reconstruction order specified! Supported orders: 1 (constant), 2 (PLM), 3 (PPM), 5 (xPPM).");
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


		{
			// level-wide launch over all boxes of the flux MultiFab
			const int b_idx = bfield_index;
			const auto b_arrs_wcomp1 = fcw_mf_cVars_wcomp[wcomp1].const_arrays();
			const auto b_arrs_wcomp2 = fcw_mf_cVars_wcomp[wcomp2].const_arrays();
			const auto b_arrs_wcomp0 = fcw_mf_cVars_wcomp[wcomp0].const_arrays();
			const auto flux_arrs = fcw_mf_fluxes_wcomp[wcomp0].arrays();

			amrex::ParallelFor(fcw_mf_fluxes_wcomp[wcomp0], amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
				// b-field Array4 aliases (no copy) for the wcomp{0/1/2}-faces, starting at component bfield_index
				const amrex::Array4<const amrex::Real> fc_a4_b_wcomp_wcomp1(b_arrs_wcomp1[bx], b_idx);
				const amrex::Array4<const amrex::Real> fc_a4_b_wcomp_wcomp2(b_arrs_wcomp2[bx], b_idx);
				const amrex::Array4<const amrex::Real> fc_a4_b_wcomp_wcomp0(b_arrs_wcomp0[bx], b_idx);
				const amrex::Array4<amrex::Real> fc_a4_flux = flux_arrs[bx];
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
