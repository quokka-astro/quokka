#ifndef MHD_SYSTEM_HPP_ // NOLINT
#define MHD_SYSTEM_HPP_

//==============================================================================
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file mhd_system.hpp
/// \brief A class for solving the resistive MHD induction equation, including Ohmic heating for non-isothermal plasmas.
///

// library headers
#include "AMReX_BLProfiler.H"
#include "AMReX_GpuControl.H"
#include "AMReX_MFIter.H"
#include "AMReX_ParmParse.H"

// internal headers
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"

AMREX_ENUM(EMFComputeScheme, FelkerStone2017, Balsara2025, Quokka2026); // NOLINT
// FelkerStone2017: Felker & Stone (2018), JCP 375:1365; uses cc v-field.
// Balsara2025a: Balsara et al. (2025a), ApJ 988:134; EMF reconstructed from cc->ec.
// Quokka2026: work in preparation; variant of Mignone21a: Mignone & Del Zanna (2021), JCP 424:109748.

AMREX_ENUM(EMFAvgScheme, LondrilloDelZanna2004, Balsara2025); // NOLINT
// LondrilloDelZanna2004: Londrillo & Del Zanna (2004), JCP 195:17; wave-speed-weighted quadrant average.
// Balsara2025b: Balsara et al. (2025b), CAMC 7; higher-order averaging via 2D Riemann solver.

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
				case EMFAvgScheme::Balsara2025:
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
	static void ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
					       SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
					       amrex::Real resistivity = 0.0);

	static void ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
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

	static void EMFAverage_Balsara2025(amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
					   amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
					   std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
					   std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside,
					   amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0, amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1,
					   amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

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
	if (emf_compute_scheme == EMFComputeScheme::FelkerStone2017) {
		MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
								 plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Balsara2025) {
		MHDSystem<problem_t>::ComputeEMF_Balsara2025(ec_mf_emfs_wcomp, cc_mf_cVars, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
							     plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
		MHDSystem<problem_t>::ComputeEMF_Quokka2026(ec_mf_emfs_wcomp, fcw_mf_vs_wcomp, fcw_mf_cVars_wcomp, fcw_mf_fspds_wcomp, reconstruction_order,
							    plm_limiter, emf_ave_scheme, dx_wcomp, resistivity);
	} else {
		throw std::runtime_error("Unsupported EMF-scheme. Expected either FelkerStone2017, Balsara2025, or Quokka2026.");
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
	} else if (emf_ave_scheme == EMFAvgScheme::Balsara2025) {
		EMFAverage_Balsara2025(ec_a4_emf_ave_wcomp2, ec_fabs_emfs_iquad, box_ec, reconstruct_dirs, fcw_fspds_wcomp, ec_fabs_bs_icomp_jeside,
				       fc_a4_b_wcomp0, fc_a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else {
		amrex::Abort("Unknown EMF averaging type");
	}
}

// compute emf components; FelkerStone2017.
// uses cc v-field and fc b-field reconstructed to ec.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						      SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme,
						      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FelkerStone2017()");
	const int nghost_cc = 4; // 4 cc ghost cells needed for cc->fc->ec PPM reconstruction
	// note: all centerings share the same distribution mapping; looping over cc MFIter is valid
	// note: cc, fc, and ec data have different cell counts

	// loop over each box-array on this level
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// extract cc v-fields
		// indexing: field[3: field component]
		const amrex::Box &box_cc_u = amrex::grow(box_cc, nghost_cc);
		std::array<amrex::FArrayBox, 3> cc_fabs_vs_wcomp = {amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
								    amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
								    amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena())};
		{
			const auto &cc_a4_v_wcomp0 = cc_fabs_vs_wcomp[0].array();
			const auto &cc_a4_v_wcomp1 = cc_fabs_vs_wcomp[1].array();
			const auto &cc_a4_v_wcomp2 = cc_fabs_vs_wcomp[2].array();
			const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();

			amrex::ParallelFor(box_cc_u, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
				const auto p_wcomp0 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				const auto p_wcomp1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				const auto p_wcomp2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				cc_a4_v_wcomp0(i, j, k) = p_wcomp0 / rho;
				cc_a4_v_wcomp1(i, j, k) = p_wcomp1 / rho;
				cc_a4_v_wcomp2(i, j, k) = p_wcomp2 / rho;
			});
		}

		// indexing: field[3: fc-normal direction = field component]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fcw_fabs_bs_wcomp = {
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcw_mf_cVars_wcomp[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// compute the b-field flux through each cell-face
		for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
			// each cell-edge is shared by two adjacent cell-faces, so a one-to-one mapping exists between face-pairs and edges,
			// therefore looping (implicitly) over the 3 edge orientations (indexed by wcomp0), rather than iterating per
			// cell-face and revisiting each face's edges, avoids redundant compute.

			// define the two reconstruction directions needed to get cc v-fields to ec;
			// right-hand-rule: dirs perpendicular to wcomp0.
			// indexing: reconstruct_dirs[2: reconstruction direction]
			std::array<int, 2> reconstruct_dirs = {(wcomp0 + 1) % 3, (wcomp0 + 2) % 3};
			// indexing: vecs_cc2ec[2: unit vector to reach edge]
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(reconstruct_dirs[0]),
								    amrex::IntVect::TheDimensionVector(reconstruct_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_plus1 = amrex::grow(box_ec, 1);

			// initialise a FArrayBox for storing the temporary v-fields created in each permutation of reconstructing fc->ec.
			// indexing: ec_fabs_vs_ieside[2: i-side of edge]
			std::array<amrex::FArrayBox, 2> ec_fabs_vs_ieside = {amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena()),
									     amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena())};

			// indexing: field[2: i-component][2: i-side of edge]
			// note: b-field components cannot be discontinuous along themselves (i.e., either side of the face where they are
			// stored), so there are only two possible values (sides of the interface), rather than four (quadrants of) possible
			// reconstructed values.
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bs_icomp_jeside;

			// initialise FArrayBox for storing the ec v-fields averaged across the two reconstruction permutations.
			// indexing: field[2: i-component][4: quadrant around edge]
			std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_vs_icomp_jquad;

			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_bs_icomp_jeside[icomp][0] = amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena());
				ec_fabs_bs_icomp_jeside[icomp][1] = amrex::FArrayBox(box_ec_plus1, 1, amrex::The_Async_Arena());
				for (int jquad = 0; jquad < 4; ++jquad) {
					ec_fabs_vs_icomp_jquad[icomp][jquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					ec_fabs_vs_icomp_jquad[icomp][jquad].setVal<amrex::RunOn::Device>(0.0);
				}
			}

			// FelkerStone2017 sec. 4.1.1 (step 3): reconstruct the two cc v-field components that are required
			// (to compute the emf at the edge) to ec. there are two possible permutations for doing this:
			//   1. cc->fc[dir-0]->ec
			//   2. cc->fc[dir-1]->ec
			// note that reconstruction does not commute, so the two estimated emfs are weighted equally, and averaged below.
			for (int iperm = 0; iperm < 2; ++iperm) {
				// for each permutation of reconstructing cc->fc->ec

				// define quantities
				const int reconstruct_dir2face = reconstruct_dirs[iperm];
				const int reconstruct_dir2edge = reconstruct_dirs[(iperm + 1) % 2];
				const auto dir2face = static_cast<FluxDir>(reconstruct_dir2face);
				const auto dir2edge = static_cast<FluxDir>(reconstruct_dir2edge);
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(reconstruct_dir2face);
				const amrex::IntVect vec_fc2ec = amrex::IntVect::TheDimensionVector(reconstruct_dir2edge);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// only keep the fc strip needed by the follow-up fc->ec reconstruction
				const amrex::Box box_fc_u = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec);
				// PPM writes one interface outside the requested range in the reconstruction direction.
				const amrex::Box box_fc_u_scratch = amrex::grow(box_fc_u, vec_cc2fc);

				// reconstruct both required v-fields cc->fc->ec
				for (int icomp = 0; icomp < 2; ++icomp) {
					// create temporary FArrayBox for storing the fc v-field reconstructed from cc
					// indexing: field[2: i-side of face]
					const int wcomp = reconstruct_dirs[icomp];
					std::array<amrex::FArrayBox, 2> fc_fabs_vs_ifside = {amrex::FArrayBox(box_fc_u_scratch, 1, amrex::The_Async_Arena()),
											     amrex::FArrayBox(box_fc_u_scratch, 1, amrex::The_Async_Arena())};

					// reconstruct v-field components cc->fc
					MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_vs_wcomp[wcomp].array(), fc_fabs_vs_ifside[0].array(),
									    fc_fabs_vs_ifside[1].array(), box_fc_u, reconstruction_order, plm_limiter);

					// reconstruct v-field components fc->ec
					for (int iface = 0; iface < 2; ++iface) {
						// reset values in temporary FArrayBox
						ec_fabs_vs_ieside[0].setVal<amrex::RunOn::Device>(0.0);
						ec_fabs_vs_ieside[1].setVal<amrex::RunOn::Device>(0.0);

						// reconstruct v-field component fc->ec
						MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_vs_ifside[iface].array(), ec_fabs_vs_ieside[0].array(),
										    ec_fabs_vs_ieside[1].array(), box_ec, reconstruction_order, plm_limiter);

						// figure out which ec quadrant this reconstructed v-field component corresponds with
						int jquad0 = -1;
						int jquad1 = -1;

						// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
						// (-,+) | (+,+)
						//   1   |   2
						// ------+------
						//   0   |   3
						// (-,-) | (+,-)
						if (iperm == 0) {
							jquad0 = (iface == 0) ? 0 : 3;
							jquad1 = (iface == 0) ? 1 : 2;
						} else {
							jquad0 = (iface == 0) ? 0 : 1;
							jquad1 = (iface == 0) ? 3 : 2;
						}

						ec_fabs_vs_icomp_jquad[icomp][jquad0].plus<amrex::RunOn::Device>(ec_fabs_vs_ieside[0], 0, 0, 1);
						ec_fabs_vs_icomp_jquad[icomp][jquad1].plus<amrex::RunOn::Device>(ec_fabs_vs_ieside[1], 0, 0, 1);
					}
				}
			}

			// finish averaging the two different ways for reconstructing v-fields: cc->fc->ec
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jquad = 0; jquad < 4; ++jquad) {
					ec_fabs_vs_icomp_jquad[icomp][jquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
				}
			}

			// FelkerStone2017 sec. 4.1.1 (steps 1 and 2): reconstruct the two required fc b-field components to ec.
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int reconstruct_dir2edge = reconstruct_dirs[(icomp + 1) % 2];
				const auto dir2edge = static_cast<FluxDir>(reconstruct_dir2edge);
				const int wcomp = reconstruct_dirs[icomp];
				// reconstruct b-field components fc->ec
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fcw_fabs_bs_wcomp[wcomp].array(), ec_fabs_bs_icomp_jeside[icomp][0].array(),
								    ec_fabs_bs_icomp_jeside[icomp][1].array(), box_ec, reconstruction_order, plm_limiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emfs_iquad;

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
					// extract relevant v-field and b-field components (host: get Array4 views)
					const int idx0 = (iquad == 0 || iquad == 3) ? 0 : 1;			    // choose from B/T for dir-0
					const int idx1 = (iquad < 2) ? 0 : 1;					    // choose from L/R for dir-1
					ec_vs_wcomp0_iquad[iquad] = ec_fabs_vs_icomp_jquad[0][iquad].const_array(); // comp=0, index jquad
					ec_vs_wcomp1_iquad[iquad] = ec_fabs_vs_icomp_jquad[1][iquad].const_array(); // comp=1, index jquad
					ec_bs_wcomp0_iquad[iquad] = ec_fabs_bs_icomp_jeside[0][idx0].const_array(); // comp=0, index idx0
					ec_bs_wcomp1_iquad[iquad] = ec_fabs_bs_icomp_jeside[1][idx1].const_array(); // comp=1, index idx1

					// define EMF FArrayBox for each quadrant (must be allocated outside the kernel)
					ec_fabs_emfs_iquad[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					ec_emfs_wcomp2_iquad[iquad] = ec_fabs_emfs_iquad[iquad].array();
				}

				// single kernel over the ec box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int iquad = 0; iquad < 4; ++iquad) {
						const amrex::Real v_wcomp0 = ec_vs_wcomp0_iquad[iquad](i, j, k);
						const amrex::Real v_wcomp1 = ec_vs_wcomp1_iquad[iquad](i, j, k);
						const amrex::Real b_wcomp0 = ec_bs_wcomp0_iquad[iquad](i, j, k);
						const amrex::Real b_wcomp1 = ec_bs_wcomp1_iquad[iquad](i, j, k);
						// FelkerStone2017 eqns. 36-37: cross(v, b) at each corner
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
void MHDSystem<problem_t>::ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emfs_wcomp, amrex::MultiFab const &cc_mf_cVars,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_cVars_wcomp,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcw_mf_fspds_wcomp, int reconstruction_order,
						  SlopeLimiter plm_limiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_wcomp,
						  amrex::Real resistivity)
{

	const BL_PROFILE("MHDSystem::ComputeEMF_Balsara2025()");
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
// note FelkerStone2017 implements this as their eqn. 41

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

		// LondrilloDelZanna2004 eqn. 56 (FelkerStone2017 eqn. 41)
		ec_a4_emf_ave_wcomp2(i, j, k) = (numerator / denominator) + term2;
		MHDSystem<problem_t>::ApplyResistiveCorrection(ec_a4_emf_ave_wcomp2, i, j, k, fc_a4_b_wcomp0, fc_a4_b_wcomp1, delta_wcomp0, delta_wcomp1,
							       dx_wcomp0, dx_wcomp1, resistivity);
	});
}

// average emf components; Balsara2025b.
// uses a 2D Riemann solver to average the EMF quadrants.
// note Balsara2025a sec. 3 recounts the same derivation (eqns. 3.2-3.10).

template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_Balsara2025(amrex::Array4<amrex::Real> ec_a4_emf_ave_wcomp2, std::array<amrex::FArrayBox, 4> const &ec_fabs_emfs_iquad,
						  amrex::Box const &box_ec, std::array<int, 2> const &reconstruct_dirs,
						  std::array<amrex::Array4<const amrex::Real>, 3> const &fcw_fspds_wcomp,
						  std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bs_icomp_jeside,
						  amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp0,
						  amrex::Array4<const amrex::Real> const &fc_a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
						  amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_Balsara2025()");
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
		// Balsara2025b sec. 3.1 (Balsara2025a sec. 3); unlike in FelkerStone2017, these wave speeds are signed,
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
			// Balsara2025b eqns. 16-17 (Balsara2025a eqn. 3.6): double-star b-field states.
			// the four-corner EMF sums pair rot180 partners (LB,RT) and (LT,RB) before combining, so the
			// numerator is bit-exact anti-covariant under the point reflection (cf. ld04 num1/num2).
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
	});
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

				// flux_eta is the wcomp0-component of cross(eta_j, b), averaged over the lo/hi bounding edges
				const amrex::Real flux_eta = 0.25 * (eta_j_wcomp1_lo * ave_b_wcomp2_lo + eta_j_wcomp1_hi * ave_b_wcomp2_hi -
								     eta_j_wcomp2_lo * ave_b_wcomp1_lo - eta_j_wcomp2_hi * ave_b_wcomp1_hi);
				fc_a4_flux(i, j, k, energy_idx) += flux_eta;
			});
		}
	}
}

#endif // MHD_SYSTEM_HPP_
