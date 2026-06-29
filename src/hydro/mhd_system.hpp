#ifndef MHD_SYSTEM_HPP_ // NOLINT
#define MHD_SYSTEM_HPP_
//==============================================================================
// ...
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file mhd_system.hpp
/// \brief Defines a class for solving the MHD equations.
///

// c++ headers

// library headers

// internal headers
#include "AMReX_BLProfiler.H"
#include "AMReX_GpuControl.H"
#include "AMReX_MFIter.H"
#include "AMReX_ParmParse.H"
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include <iostream>

// Felker + Stone (2017): uses cell-centered velocity
// Balsara (2025): EMF interpolation from cc->ec
// Quokka variant of FS17: uses face-centered Riemann velocity
AMREX_ENUM(EMFComputeScheme, FelkerStone2017, Balsara2025, Quokka2026); // NOLINT

// Londrillo + Del Zanna (2004)
// Balsara (2025): Higher-order averaging
AMREX_ENUM(EMFAvgScheme, LondrilloDelZanna2004, Balsara2025); // NOLINT

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

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	static constexpr int bfield_index = Physics_Indices<problem_t>::mhdFirstIndex;

	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_ave_scheme,
			       SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
			       amrex::Real resistivity = 0.0);

	static void AverageEMF(amrex::Array4<amrex::Real> const &a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad, amrex::Box const &box_ec,
			       std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
			       std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside, EMFAvgScheme emf_ave_scheme,
			       amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1,
			       amrex::Real resistivity);

	static void ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
					       SlopeLimiter plmLimiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					       amrex::Real resistivity = 0.0);

	static void ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter,
					   EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real resistivity = 0.0);

	static void ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter,
					  EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real resistivity = 0.0);

	static void EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad,
						     amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
						     std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
						     std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside,
						     amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1,
						     amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

	static void EMFAverage_Balsara2025(amrex::Array4<amrex::Real> a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad, amrex::Box const &box_ec,
					   std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
					   std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside,
					   amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1, amrex::Real dx_wcomp0,
					   amrex::Real dx_wcomp1, amrex::Real resistivity);

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto computeResistiveEMF(amrex::Array4<const amrex::Real> const &a4_b_wcomp0,
									    amrex::Array4<const amrex::Real> const &a4_b_wcomp1, int i, int j, int k,
									    std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
									    amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity) -> amrex::Real;

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &a4_emf_wcomp2_ave, int i, int j, int k,
										 amrex::Array4<const amrex::Real> const &a4_b_wcomp0,
										 amrex::Array4<const amrex::Real> const &a4_b_wcomp1,
										 std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
										 amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity);

	static void AddResistiveEnergyFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					   amrex::Real resistivity);

	static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_iValid, int reconstructionOrder,
				  SlopeLimiter plmLimiter);

	static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);
};

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeResistivity(int /*i*/, int /*j*/, int /*k*/, amrex::Array4<const amrex::Real> const & /*a4_b_wcomp0*/,
							    amrex::Array4<const amrex::Real> const & /*a4_b_wcomp1*/, amrex::Real /*dx_wcomp0*/, amrex::Real /*dx_wcomp1*/)
    -> amrex::Real
{
	static_assert(sizeof(problem_t) == 0, "computeResistivity must be specialized in the problem file when using ResistivityModel::problem_defined");
	return 0.0;
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_ave_scheme,
				      SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
				      amrex::Real resistivity)
{
	if (emf_compute_scheme == EMFComputeScheme::FelkerStone2017) {
		MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(ec_mf_emf_comps, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
								 emf_ave_scheme, dx, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Balsara2025) {
		MHDSystem<problem_t>::ComputeEMF_Balsara2025(ec_mf_emf_comps, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
							     emf_ave_scheme, dx, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
		MHDSystem<problem_t>::ComputeEMF_Quokka2026(ec_mf_emf_comps, fcx_mf_vel, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
							    emf_ave_scheme, dx, resistivity);
	} else {
		throw std::runtime_error("Unsupported EMF-scheme. Expected either FelkerStone2017, Balsara2025, or Quokka2026.");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::AverageEMF(amrex::Array4<amrex::Real> const &a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad, amrex::Box const &box_ec,
				      std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
				      std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside, EMFAvgScheme emf_ave_scheme,
				      amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1, amrex::Real dx_wcomp0,
				      amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	if (emf_ave_scheme == EMFAvgScheme::LondrilloDelZanna2004) {
		EMFAverage_LondrilloDelZanna2004(a4_emf_wcomp2_ave, ec_fabs_emf_iquad, box_ec, extrap_dirs, fspds, ec_fabs_b_icomp_jeside, a4_b_wcomp0, a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else if (emf_ave_scheme == EMFAvgScheme::Balsara2025) {
		EMFAverage_Balsara2025(a4_emf_wcomp2_ave, ec_fabs_emf_iquad, box_ec, extrap_dirs, fspds, ec_fabs_b_icomp_jeside, a4_b_wcomp0, a4_b_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else {
		amrex::Abort("Unknown EMF averaging type");
	}
}

// emf compute solver; Felker18a (Felker & Stone 2018, ApJS 237:24).
// uses cell-centered velocity and face-centered magnetic fields extrapolated to the cell-edge.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						      SlopeLimiter plmLimiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						      amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FelkerStone2017()");
	const int nghost_cc = 4; // 4 cc ghost cells needed for cc->fc->ec PPM reconstruction
	// note: all centerings share the same distribution mapping; looping over cc MFIter is valid
	// note: cc, fc, and ec data have different cell counts

	// loop over each box-array on this level
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// extract cell-centered velocity fields
		// indexing: field[3: x-component]
		const amrex::Box &box_cc_u = amrex::grow(box_cc, nghost_cc);
		std::array<amrex::FArrayBox, 3> cc_fabs_u_wcomp = {amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena())};
		{
			const auto &cc_a4_u_wcomp0 = cc_fabs_u_wcomp[0].array();
			const auto &cc_a4_u_wcomp1 = cc_fabs_u_wcomp[1].array();
			const auto &cc_a4_u_wcomp2 = cc_fabs_u_wcomp[2].array();
			const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();

			amrex::ParallelFor(box_cc_u, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
				const auto p_wcomp0 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				const auto p_wcomp1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				const auto p_wcomp2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				cc_a4_u_wcomp0(i, j, k) = p_wcomp0 / rho;
				cc_a4_u_wcomp1(i, j, k) = p_wcomp1 / rho;
				cc_a4_u_wcomp2(i, j, k) = p_wcomp2 / rho;
			});
		}

		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_b_wcomp = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// compute the magnetic flux through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {
			// for each of the two cell-edges on the cell-face
			// we are doing redundant compute. only need to look at one edge for each face: there is a one-to-one mapping.

			// define the two directions we need to extrapolate cell-centered velocity fields to get them to the cell-edge
			// we will want to compute E2 = (u_wcomp0 * b_wcomp1 - u_wcomp1 * b_wcomp0) along the cell-edge
			std::array<int, 2> extrap_dirs = {(iedge + 1) % 3, (iedge + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(extrap_dirs[0]),
								    amrex::IntVect::TheDimensionVector(extrap_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// initialise FArrayBox for storing the temporary edge-centered velocity fields created in each permutation of reconstructing from the
			// cell-face indexing: field[2: i-side of edge]
			std::array<amrex::FArrayBox, 2> ec_fabs_u_ieside = {amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena()),
									    amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena())};

			// indexing: field[2: i-compnent][2: i-side of edge]
			// note: magnetic field components cannot be discontinuous along themselves (i.e., either side of the face where they are
			// stored), so there are only two possible values (sides), rather than four (quadrants of) possible reconstructed values
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_b_icomp_jeside;

			// initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
			// indexing: field[2: i-compnent][4: quadrant around edge]
			std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_u_icomp_jquad;

			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_b_icomp_jeside[icomp][0] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				ec_fabs_b_icomp_jeside[icomp][1] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				for (int jquad = 0; jquad < 4; ++jquad) {
					ec_fabs_u_icomp_jquad[icomp][jquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					ec_fabs_u_icomp_jquad[icomp][jquad].setVal<amrex::RunOn::Device>(0.0);
				}
			}

			// extrapolate the two required cell-centered velocity field components to the cell-edge
			// there are two possible permutations for doing this: getting cell-centered quanties to a cell-edge
			// first is cc->fc[dir-0]->ec and second is cc->fc[dir-1]->ec
			for (int iperm = 0; iperm < 2; ++iperm) {
				// for each permutation of extrapolating cc->fc->ec

				// define quantities
				const int extrap_dir2face = extrap_dirs[iperm];
				const int extrap_dir2edge = extrap_dirs[(iperm + 1) % 2];
				const auto dir2face = static_cast<FluxDir>(extrap_dir2face);
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(extrap_dir2face);
				const amrex::IntVect vec_fc2ec = amrex::IntVect::TheDimensionVector(extrap_dir2edge);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// only keep the face-centered strip needed by the follow-up fc->ec reconstruction
				const amrex::Box box_fc_u = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec);
				// PPM writes one interface outside the requested range in the reconstruction direction.
				const amrex::Box box_fc_u_scratch = amrex::grow(box_fc_u, vec_cc2fc);

				// extrapolate both required cell-centered velocity fields to the cell-edge
				for (int icomp = 0; icomp < 2; ++icomp) {
					// create temporary FArrayBox for storing the face-centered velocity field reconstructed from the cell-center
					// indexing: field[2: i-side of face]
					const int wcomp = extrap_dirs[icomp];
					std::array<amrex::FArrayBox, 2> fc_fabs_u_ifside = {amrex::FArrayBox(box_fc_u_scratch, 1, amrex::The_Async_Arena()),
											    amrex::FArrayBox(box_fc_u_scratch, 1, amrex::The_Async_Arena())};

					// extrapolate cell-centered velocity components to the cell-face
					MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_u_wcomp[wcomp].array(), fc_fabs_u_ifside[0].array(),
									    fc_fabs_u_ifside[1].array(), box_fc_u, reconstructionOrder, plmLimiter);

					// extrapolate face-centered velocity components to the cell-edge
					for (int iface = 0; iface < 2; ++iface) {
						// reset values in temporary FArrayBox
						ec_fabs_u_ieside[0].setVal<amrex::RunOn::Device>(0.0);
						ec_fabs_u_ieside[1].setVal<amrex::RunOn::Device>(0.0);

						// extrapolate face-centered velocity component to the cell-edge
						MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_u_ifside[iface].array(), ec_fabs_u_ieside[0].array(),
										    ec_fabs_u_ieside[1].array(), box_ec, reconstructionOrder, plmLimiter);

						// figure out which quadrant of the cell-edge this extrapolated velocity component corresponds with
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

						ec_fabs_u_icomp_jquad[icomp][jquad0].plus<amrex::RunOn::Device>(ec_fabs_u_ieside[0], 0, 0, 1);
						ec_fabs_u_icomp_jquad[icomp][jquad1].plus<amrex::RunOn::Device>(ec_fabs_u_ieside[1], 0, 0, 1);
					}
				}
			}

			// finish averaging the two different ways for extrapolating velocity fields: cc->fc->ec
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jquad = 0; jquad < 4; ++jquad) {
					ec_fabs_u_icomp_jquad[icomp][jquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
				}
			}

			// extrapolate the two required face-centered magnetic field components to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int extrap_dir2edge = extrap_dirs[(icomp + 1) % 2];
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const int wcomp = extrap_dirs[icomp];
				// extrapolate face-centered magnetic components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_b_wcomp[wcomp].array(), ec_fabs_b_icomp_jeside[icomp][0].array(),
								    ec_fabs_b_icomp_jeside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_iquad;

			// compute the EMF along the cell-edge using a single kernel (all quadrants inside)
			{
				// bind read/write Array4 views on the host (required for GPU lambda capture)
				std::array<amrex::Array4<const amrex::Real>, 4> us_wcomp0;
				std::array<amrex::Array4<const amrex::Real>, 4> us_wcomp1;
				std::array<amrex::Array4<const amrex::Real>, 4> bs_wcomp0;
				std::array<amrex::Array4<const amrex::Real>, 4> bs_wcomp1;
				std::array<amrex::Array4<amrex::Real>, 4> emfs_wcomp2;

				for (int qi = 0; qi < 4; ++qi) {
					// extract relevant velocity and magnetic field components (host: get Array4 views)
					const int idx0 = (qi == 0 || qi == 3) ? 0 : 1;	    // B/T selector for dir-0
					const int idx1 = (qi < 2) ? 0 : 1;		    // L/R selector for dir-1
					us_wcomp0[qi] = ec_fabs_u_icomp_jquad[0][qi].const_array();	    // component 0, index jquad
					us_wcomp1[qi] = ec_fabs_u_icomp_jquad[1][qi].const_array();	    // component 1, index jquad
					bs_wcomp0[qi] = ec_fabs_b_icomp_jeside[0][idx0].const_array(); // component 0, index idx0
					bs_wcomp1[qi] = ec_fabs_b_icomp_jeside[1][idx1].const_array(); // component 1, index idx1

					// define EMF FArrayBox for each quadrant (we need to allocate outside the kernel)
					ec_fabs_emf_iquad[qi] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					emfs_wcomp2[qi] = ec_fabs_emf_iquad[qi].array();
				}

				// single kernel over the edge-centered box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int qi = 0; qi < 4; ++qi) {
						const amrex::Real u_wcomp0 = us_wcomp0[qi](i, j, k);
						const amrex::Real u_wcomp1 = us_wcomp1[qi](i, j, k);
						const amrex::Real b_wcomp0 = bs_wcomp0[qi](i, j, k);
						const amrex::Real b_wcomp1 = bs_wcomp1[qi](i, j, k);
						emfs_wcomp2[qi](i, j, k) = u_wcomp0 * b_wcomp1 - u_wcomp1 * b_wcomp0; // cross product at the edge
					}
				});
			}

			// compute electric field on the cell-edge
			const auto &a4_emf_wcomp2_ave = ec_mf_emf_comps[iedge][mfi].array();

			// selected averaging method for EMF:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf_wcomp2_ave, ec_fabs_emf_iquad, box_ec, extrap_dirs, fspds, ec_fabs_b_icomp_jeside, emf_ave_scheme,
							 fcx_mf_cVars[extrap_dirs[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[extrap_dirs[1]][mfi].const_array(bfield_index), dx[extrap_dirs[0]], dx[extrap_dirs[1]],
							 resistivity);
		}
	}
}

// emf compute solver; Quokka (2026) variant of Felker18a.
// uses face-centered Riemann velocity and face-centered magnetic fields extrapolated to the cell-edge.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						 SlopeLimiter plmLimiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						 amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_Quokka2026()");

	// loop over each box-array on the level
	// note: all centerings share the same distribution mapping; looping over cc MFIter is valid
	// note: cc, fc, and ec data have different cell counts
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(fcx_mf_cVars[0], amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// indexing: field[3: x-component/x-face]
		// create a view of all the u-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_u_wcomp = {
		    amrex::FArrayBox(fcx_mf_vel[0][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[1][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[2][mfi], amrex::make_alias, 0, 1),
		};
		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_b_wcomp = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};
		// compute the emf components on the cell-edge to inform how much magnetic flux travels through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {

			// define the two face-centered velocity/magnetic field components we need at the cell-edge
			// we will want to compute E2 = (u_wcomp0 * b_wcomp1 - u_wcomp1 * b_wcomp0) along the cell-edge
			std::array<int, 2> field_w_indices = {(iedge + 1) % 3, (iedge + 2) % 3};
			const amrex::Box box_ec = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(field_w_indices[0]) +
									     amrex::IntVect::TheDimensionVector(field_w_indices[1]));
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// FArrayBoxes for storing the edge-centered fields produced by reconstructing from the cell-face to the cell-edge
			// indexing: field[2: i-component][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_u_icomp_jeside;
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_b_icomp_jeside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jeside = 0; jeside < 2; ++jeside) {
					ec_fabs_u_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
					ec_fabs_b_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
			}

			// extrapolate the face-centered fields (normal to the cell-face) to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(field_w_indices[(icomp + 1) % 2]);
				const int wcomp = field_w_indices[icomp];
				// extrapolate face-centered components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_b_wcomp[wcomp].array(), ec_fabs_b_icomp_jeside[icomp][0].array(),
								    ec_fabs_b_icomp_jeside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_u_wcomp[wcomp].array(), ec_fabs_u_icomp_jeside[icomp][0].array(),
								    ec_fabs_u_icomp_jeside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_iquad;
			// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
			// |---------------------------------------------------------------------------------------------|
			// |          q_2                                                                                |
			// |       u,b_{0,T}                 |                                                           |
			// |       \       /       q_1 + q_2 | q_2 + q_3                                                 |
			// |        \     /             Q_1  |  Q_2          Q_0 = u_{0,B} * b_{1,L} - u_{1,L} * b_{0,B} |
			// |         \   /             (-,+) | (+,+)                                                     |
			// |    q_1   \ /   q_3              |               Q_1 = u_{0,T} * b_{1,L} - u_{1,L} * b_{0,T} |
			// | u,b_{1,L} . u,b_{1,R} -> --------------- where:                                             |
			// |          / \                    |               Q_2 = u_{0,T} * b_{1,R} - u_{1,R} * b_{0,T} |
			// |         /   \             (-,-) | (+,-)                                                     |
			// |        /     \             Q_0  |  Q_3          Q_3 = u_{0,B} * b_{1,R} - u_{1,R} * b_{0,B} |
			// |       /       \       q_0 + q_1 | q_3 + q_0                                                 |
			// |       u,b_{0,B}                 |                                                           |
			// |          q_0                                                                                |
			// |---------------------------------------------------------------------------------------------|
			// compute the EMF along the cell-edge using a single kernel (all quadrants inside)
			{
				// bind read/write Array4 views on the host (required for GPU lambda capture)
				std::array<amrex::Array4<const amrex::Real>, 4> us_wcomp0;
				std::array<amrex::Array4<const amrex::Real>, 4> us_wcomp1;
				std::array<amrex::Array4<const amrex::Real>, 4> bs_wcomp0;
				std::array<amrex::Array4<const amrex::Real>, 4> bs_wcomp1;
				std::array<amrex::Array4<amrex::Real>, 4> emfs_wcomp2;

				for (int qi = 0; qi < 4; ++qi) {
					const int idx0 = (qi == 0 || qi == 3) ? 0 : 1; // B/T selector for dir-0
					const int idx1 = (qi < 2) ? 0 : 1;	       // L/R selector for dir-1

					// define EMF FArrayBox for each quadrant (we need to allocate outside the kernel)
					ec_fabs_emf_iquad[qi] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());

					// extract relevant velocity and magnetic field components (host: get Array4 views)
					us_wcomp0[qi] = ec_fabs_u_icomp_jeside[0][idx0].const_array(); // B/T
					bs_wcomp0[qi] = ec_fabs_b_icomp_jeside[0][idx0].const_array(); // B/T
					us_wcomp1[qi] = ec_fabs_u_icomp_jeside[1][idx1].const_array(); // L/R
					bs_wcomp1[qi] = ec_fabs_b_icomp_jeside[1][idx1].const_array(); // L/R
					emfs_wcomp2[qi] = ec_fabs_emf_iquad[qi].array();		    // output EMF view
				}

				// single kernel over the edge-centered box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int qi = 0; qi < 4; ++qi) {
						const amrex::Real u_wcomp0 = us_wcomp0[qi](i, j, k);
						const amrex::Real u_wcomp1 = us_wcomp1[qi](i, j, k);
						const amrex::Real b_wcomp0 = bs_wcomp0[qi](i, j, k);
						const amrex::Real b_wcomp1 = bs_wcomp1[qi](i, j, k);
						emfs_wcomp2[qi](i, j, k) = u_wcomp0 * b_wcomp1 - u_wcomp1 * b_wcomp0; // cross product at the edge
					}
				});
			}

			const auto &a4_emf_wcomp2_ave = ec_mf_emf_comps[iedge][mfi].array();

			// selected averaging method for the emf:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf_wcomp2_ave, ec_fabs_emf_iquad, box_ec, field_w_indices, fspds, ec_fabs_b_icomp_jeside, emf_ave_scheme,
							 fcx_mf_cVars[field_w_indices[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[field_w_indices[1]][mfi].const_array(bfield_index), dx[field_w_indices[0]],
							 dx[field_w_indices[1]], resistivity);
		}
	}
}

// emf compute solver; Balsara25a (Balsara et al. 2025, ApJ 988:134b).
// uses cell-centered velocity and face-centered magnetic fields averaged to cell-center to compute the emf,
// then extrapolates to the cell-edge.

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						  SlopeLimiter plmLimiter, EMFAvgScheme emf_ave_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						  amrex::Real resistivity)
{
	// v x b at cell center; v is already cc, b averaged from fc

	const BL_PROFILE("MHDSystem::ComputeEMF_Balsara2025()");
	const int nghost_cc = 4;
	// note: cc, fc, and ec data have different cell counts

	const auto &ba = cc_mf_cVars.boxArray();
	const auto &dm = cc_mf_cVars.DistributionMap();
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	amrex::MultiFab cc_mf_emf(ba, dm, 3, nghost_cc);
	cc_mf_emf.setVal(0.0, 0, 3, nghost_cc); // initialize to zero everywhere including ghost zones

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc_emf = mfi.growntilebox(nghost_cc); // ensure enough ghost cells for EMF computation

		// emf Array4 views for this tile
		const auto &cc_a4_emf_wcomp0 = cc_mf_emf[mfi].array(0);
		const auto &cc_a4_emf_wcomp1 = cc_mf_emf[mfi].array(1);
		const auto &cc_a4_emf_wcomp2 = cc_mf_emf[mfi].array(2);

		const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
		std::array<amrex::Array4<amrex::Real>, 3> const cc_a4_emf_wcomp = {cc_a4_emf_wcomp0, cc_a4_emf_wcomp1, cc_a4_emf_wcomp2};
		std::array<amrex::FArrayBox, 3> fc_fabs_b_wcomp = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// face-centered b-field Array4 views
		std::array<amrex::Array4<amrex::Real const>, 3> fc_a4_b_wcomp = {fcx_mf_cVars[0][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[1][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[2][mfi].const_array(MHDSystem<problem_t>::bfield_index)};

		// compute v x b for all three dimensions
		amrex::ParallelFor(box_cc_emf, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
			std::array<amrex::Real, 3> v = {cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho,
							cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho,
							cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho};
			// emf component for each dimension
			for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
				int const wcomp1 = (wcomp0 + 1) % 3;
				int const wcomp2 = (wcomp0 + 2) % 3;

				std::array<int, 3> delta_wcomp1 = {0, 0, 0};
				std::array<int, 3> delta_wcomp2 = {0, 0, 0};
				delta_wcomp1[wcomp1] = 1;
				delta_wcomp2[wcomp2] = 1;

				// average face-centered b to cell center
				amrex::Real const b_wcomp1_ave =
				    0.5 * (fc_a4_b_wcomp[wcomp1](i, j, k) + fc_a4_b_wcomp[wcomp1](i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2]));
				amrex::Real const b_wcomp2_ave =
				    0.5 * (fc_a4_b_wcomp[wcomp2](i, j, k) + fc_a4_b_wcomp[wcomp2](i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2]));

				// v x b computation
				cc_a4_emf_wcomp[wcomp0](i, j, k) = v[wcomp1] * b_wcomp2_ave - v[wcomp2] * b_wcomp1_ave;
			}
		});
	}

	cc_mf_emf.FillBoundary(); // fill ghost cells
	amrex::Gpu::streamSynchronize();

	// interpolate emf from cell-center to cell-edge; also move b-field from face to cell-edge for averaging solvers.

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) { // keep
		const amrex::Box &box_cc = mfi.validbox();

		std::array<amrex::FArrayBox, 3> fc_fabs_b_wcomp = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		for (int iedge = 0; iedge < 3; ++iedge) {

			std::array<int, 2> extrap_dirs = {(iedge + 1) % 3, (iedge + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(extrap_dirs[0]),
								    amrex::IntVect::TheDimensionVector(extrap_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// initializing array to hold EMF at cell edge
			const auto &a4_emf_wcomp2_ave = ec_mf_emf_comps[iedge][mfi].array();
			std::array<amrex::FArrayBox, 2> ec_fabs_emf_ieside = {amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena()),
									      amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena())};

			ec_fabs_emf_ieside[0].setVal<amrex::RunOn::Device>(0.0);
			ec_fabs_emf_ieside[1].setVal<amrex::RunOn::Device>(0.0);
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_iquad;

			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emf_iquad[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
				ec_fabs_emf_iquad[iquad].setVal<amrex::RunOn::Device>(0.0);
			}

			// interpolate the cell-centered EMF to the cell-edge
			// there are two possible permutations for doing this: getting cell-centered quanties to a cell-edge
			// first is cc->fc[dir-0]->ec and second is cc->fc[dir-1]->ec
			for (int iperm = 0; iperm < 2; ++iperm) {
				// for each permutation of extrapolating cc->fc->ec

				const int extrap_dir2face = extrap_dirs[iperm];
				const int extrap_dir2edge = extrap_dirs[(iperm + 1) % 2];
				const auto dir2face = static_cast<FluxDir>(extrap_dir2face);
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(extrap_dir2face);
				const amrex::IntVect vec_fc2ec = amrex::IntVect::TheDimensionVector(extrap_dir2edge);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// only keep the face-centered strip needed by the follow-up fc->ec reconstruction
				const amrex::Box box_fc_emf = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec);
				// PPM writes one interface outside the requested range in the reconstruction direction.
				const amrex::Box box_fc_emf_scratch = amrex::grow(box_fc_emf, vec_cc2fc);

				// extrapolate both required cell-centered EMF to the cell-edge

				// create temporary FArrayBox for storing the face-centered EMF reconstructed from the cell-center
				// indexing: field[2: i-side of face]
				std::array<amrex::FArrayBox, 2> fc_fabs_emf_ifside = {amrex::FArrayBox(box_fc_emf_scratch, 1, amrex::The_Async_Arena()),
										      amrex::FArrayBox(box_fc_emf_scratch, 1, amrex::The_Async_Arena())};
				// reset values in temporary FArrayBox
				fc_fabs_emf_ifside[0].setVal<amrex::RunOn::Device>(0.0);
				fc_fabs_emf_ifside[1].setVal<amrex::RunOn::Device>(0.0);

				// extrapolate cell-centered velocity components to the cell-face
				MHDSystem<problem_t>::ReconstructTo(dir2face, cc_mf_emf[mfi].array(iedge), fc_fabs_emf_ifside[0].array(),
								    fc_fabs_emf_ifside[1].array(), box_fc_emf, reconstructionOrder, plmLimiter);

				// extrapolate face-centered emf components to the cell-edge
				for (int iface = 0; iface < 2; ++iface) {
					// reset values in temporary FArrayBox
					ec_fabs_emf_ieside[0].setVal<amrex::RunOn::Device>(0.0);
					ec_fabs_emf_ieside[1].setVal<amrex::RunOn::Device>(0.0);

					MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_emf_ifside[iface].array(), ec_fabs_emf_ieside[0].array(),
									    ec_fabs_emf_ieside[1].array(), box_ec, reconstructionOrder, plmLimiter);

					// figure out which quadrant of the cell-edge this extrapolated emf component corresponds with
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

					ec_fabs_emf_iquad[iquad0].plus<amrex::RunOn::Device>(ec_fabs_emf_ieside[0], 0, 0, 1);
					ec_fabs_emf_iquad[iquad1].plus<amrex::RunOn::Device>(ec_fabs_emf_ieside[1], 0, 0, 1);
				}
			}
			// finish averaging the two different ways for extrapolating emf: cc->fc->ec
			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emf_iquad[iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
			}

			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_b_icomp_jeside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int jeside = 0; jeside < 2; ++jeside) {
					ec_fabs_b_icomp_jeside[icomp][jeside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
			}

			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(extrap_dirs[(icomp + 1) % 2]);
				const int wcomp = extrap_dirs[icomp];
				// extrapolate face-centered components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_b_wcomp[wcomp].array(), ec_fabs_b_icomp_jeside[icomp][0].array(),
								    ec_fabs_b_icomp_jeside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}
			// selected averaging method for the emf:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf_wcomp2_ave, ec_fabs_emf_iquad, box_ec, extrap_dirs, fspds, ec_fabs_b_icomp_jeside, emf_ave_scheme,
							 fcx_mf_cVars[extrap_dirs[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[extrap_dirs[1]][mfi].const_array(bfield_index), dx[extrap_dirs[0]], dx[extrap_dirs[1]],
							 resistivity);
		}
	}
}

// emf averaging solver; LD2004 (Londrillo & Del Zanna 2004, JCP 195). uses fast wave speeds to weight the quadrant average.
template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad,
							    amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
							    std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
							    std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside,
							    amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1,
							    amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_LondrilloDelZanna2004()");

	const auto &a4_emf_wcomp2_iquad0 = ec_fabs_emf_iquad[0].const_array();
	const auto &a4_emf_wcomp2_iquad1 = ec_fabs_emf_iquad[1].const_array();
	const auto &a4_emf_wcomp2_iquad2 = ec_fabs_emf_iquad[2].const_array();
	const auto &a4_emf_wcomp2_iquad3 = ec_fabs_emf_iquad[3].const_array();

	const auto &a4_b_wcomp0_m = ec_fabs_b_icomp_jeside[0][0].const_array();
	const auto &a4_b_wcomp0_p = ec_fabs_b_icomp_jeside[0][1].const_array();
	const auto &a4_b_wcomp1_m = ec_fabs_b_icomp_jeside[1][0].const_array();
	const auto &a4_b_wcomp1_p = ec_fabs_b_icomp_jeside[1][1].const_array();

	int const wcomp0_comp = extrap_dirs[0];
	int const wcomp1_comp = extrap_dirs[1];
	std::array<int, 3> delta_wcomp0 = {0, 0, 0};
	std::array<int, 3> delta_wcomp1 = {0, 0, 0};

	delta_wcomp0[wcomp0_comp] = 1;
	delta_wcomp1[wcomp1_comp] = 1;

	const auto &a4_fspd_wcomp0 = fspds[wcomp0_comp];
	const auto &a4_fspd_wcomp1 = fspds[wcomp1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double a0_m = std::max(a4_fspd_wcomp0(i, j, k, 0), a4_fspd_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 0));
		const double a0_p = std::max(a4_fspd_wcomp0(i, j, k, 1), a4_fspd_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 1));
		const double a1_m = std::max(a4_fspd_wcomp1(i, j, k, 0), a4_fspd_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 0));
		const double a1_p = std::max(a4_fspd_wcomp1(i, j, k, 1), a4_fspd_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 1));

		const double emf_wcomp2_iquad0 = a4_emf_wcomp2_iquad0(i, j, k);
		const double emf_wcomp2_iquad1 = a4_emf_wcomp2_iquad1(i, j, k);
		const double emf_wcomp2_iquad2 = a4_emf_wcomp2_iquad2(i, j, k);
		const double emf_wcomp2_iquad3 = a4_emf_wcomp2_iquad3(i, j, k);

		const double b_wcomp0_T = a4_b_wcomp0_p(i, j, k);
		const double b_wcomp0_B = a4_b_wcomp0_m(i, j, k);
		const double b_wcomp1_R = a4_b_wcomp1_p(i, j, k);
		const double b_wcomp1_L = a4_b_wcomp1_m(i, j, k);
		// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
		// (-,+) | (+,+)
		//   1   |   2
		// ------+------
		//   0   |   3
		// (-,-) | (+,-)

		const double num1 = ((a0_p * a1_p) * emf_wcomp2_iquad0 + (a0_m * a1_p) * emf_wcomp2_iquad3) + ((a0_p * a1_m) * emf_wcomp2_iquad1 + (a0_m * a1_m) * emf_wcomp2_iquad2);
		const double num2 = ((a0_p * a1_p) * emf_wcomp2_iquad0 + (a0_p * a1_m) * emf_wcomp2_iquad1) + ((a0_m * a1_p) * emf_wcomp2_iquad3 + (a0_m * a1_m) * emf_wcomp2_iquad2);

		// averaged for exact floating-point symmetry.
		const double numerator = 0.5 * (num1 + num2);
		const double denominator = (a0_m + a0_p) * (a1_m + a1_p);

		// Felker18a eq. 41 (= LD2004 eq. 56); a0_m<=0,a0_p>=0 are signed speeds (not negated like in Balsara25a).
		const double term2 = ((a1_m * a1_p) / (a1_m + a1_p)) * (b_wcomp0_T - b_wcomp0_B) + ((a0_m * a0_p) / (a0_m + a0_p)) * (b_wcomp1_L - b_wcomp1_R);

		a4_emf_wcomp2_ave(i, j, k) = (numerator / denominator) + term2;
		MHDSystem<problem_t>::ApplyResistiveCorrection(a4_emf_wcomp2_ave, i, j, k, a4_b_wcomp0, a4_b_wcomp1, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	});
}

// emf averaging via 2d riemann solver; Balsara25a (Balsara et al. 2025, ApJ 988:134b), sec. 3.

template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_Balsara2025(amrex::Array4<amrex::Real> a4_emf_wcomp2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_iquad,
						  amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
						  std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
						  std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_b_icomp_jeside,
						  amrex::Array4<const amrex::Real> const &a4_b_wcomp0, amrex::Array4<const amrex::Real> const &a4_b_wcomp1, amrex::Real dx_wcomp0,
						  amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_Balsara2025()");
	const auto &a4_emf_wcomp2_iquad0 = ec_fabs_emf_iquad[0].const_array();
	const auto &a4_emf_wcomp2_iquad1 = ec_fabs_emf_iquad[1].const_array();
	const auto &a4_emf_wcomp2_iquad2 = ec_fabs_emf_iquad[2].const_array();
	const auto &a4_emf_wcomp2_iquad3 = ec_fabs_emf_iquad[3].const_array();

	const auto &a4_b_wcomp0_m = ec_fabs_b_icomp_jeside[0][0].const_array();
	const auto &a4_b_wcomp0_p = ec_fabs_b_icomp_jeside[0][1].const_array();
	const auto &a4_b_wcomp1_m = ec_fabs_b_icomp_jeside[1][0].const_array();
	const auto &a4_b_wcomp1_p = ec_fabs_b_icomp_jeside[1][1].const_array();

	int const wcomp0_comp = extrap_dirs[0];
	int const wcomp1_comp = extrap_dirs[1];
	std::array<int, 3> delta_wcomp0 = {0, 0, 0};
	std::array<int, 3> delta_wcomp1 = {0, 0, 0};

	delta_wcomp0[wcomp0_comp] = 1;
	delta_wcomp1[wcomp1_comp] = 1;

	const auto &a4_fspd_wcomp0 = fspds[wcomp0_comp];
	const auto &a4_fspd_wcomp1 = fspds[wcomp1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		// signal speeds (Balsara25a sec. 3): s_L <= 0, s_B <= 0; s_R >= 0, s_T >= 0; max over two adjacent faces per dir (Felker18a app. A).
		const double s_L = -std::max(a4_fspd_wcomp0(i, j, k, 0), a4_fspd_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 0));
		const double s_R = std::max(a4_fspd_wcomp0(i, j, k, 1), a4_fspd_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2], 1));
		const double s_B = -std::max(a4_fspd_wcomp1(i, j, k, 0), a4_fspd_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 0));
		const double s_T = std::max(a4_fspd_wcomp1(i, j, k, 1), a4_fspd_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2], 1));

		// emf quadrants (Balsara25a fig. 2): LB=E_z(LD), LT=E_z(LU), RT=E_z(RU), RB=E_z(RD).
		const auto emf_wcomp2_LB = a4_emf_wcomp2_iquad0(i, j, k);
		const auto emf_wcomp2_LT = a4_emf_wcomp2_iquad1(i, j, k);
		const auto emf_wcomp2_RT = a4_emf_wcomp2_iquad2(i, j, k);
		const auto emf_wcomp2_RB = a4_emf_wcomp2_iquad3(i, j, k);

		// open: b-field slot assignment (b_wcomp0_T=m, b_wcomp0_B=p) is inverted relative to Balsara25a geometry (BxU/BxD);
		// reverting to the geometrically consistent assignment breaks BrioWuShockTube. root cause unresolved.
		const auto b_wcomp0_T = a4_b_wcomp0_m(i, j, k);
		const auto b_wcomp0_B = a4_b_wcomp0_p(i, j, k);
		const auto b_wcomp1_R = a4_b_wcomp1_m(i, j, k);
		const auto b_wcomp1_L = a4_b_wcomp1_p(i, j, k);

		const auto s_max = std::max({s_L, s_R, s_B, s_T});

		double emf_wcomp2_T_star = 0.0;
		double emf_wcomp2_B_star = 0.0;
		double b_wcomp1_dstar = 0.0;
		double emf_wcomp2_R_star = 0.0;
		double emf_wcomp2_L_star = 0.0;
		double b_wcomp0_dstar = 0.0;
		double emf_wcomp2_dstar = 0.0;

		if (s_L != s_R && s_B != s_T) {
			// Balsara25a eq. 3.2: x-direction HLL star states.
			emf_wcomp2_T_star = (s_R * emf_wcomp2_LT - s_L * emf_wcomp2_RT) / (s_R - s_L) - (s_R * s_L) * (b_wcomp1_R - b_wcomp1_L) / (s_R - s_L);
			emf_wcomp2_B_star = (s_R * emf_wcomp2_LB - s_L * emf_wcomp2_RB) / (s_R - s_L) - (s_R * s_L) * (b_wcomp1_R - b_wcomp1_L) / (s_R - s_L);
			// Balsara25a eq. 3.4: y-direction HLL star states.
			emf_wcomp2_R_star = (s_T * emf_wcomp2_RB - s_B * emf_wcomp2_RT) / (s_T - s_B) + (s_T * s_B) * (b_wcomp0_T - b_wcomp0_B) / (s_T - s_B);
			emf_wcomp2_L_star = (s_T * emf_wcomp2_LB - s_B * emf_wcomp2_LT) / (s_T - s_B) + (s_T * s_B) * (b_wcomp0_T - b_wcomp0_B) / (s_T - s_B);
			// Balsara25a eq. 3.6: double-star b-field states.
			b_wcomp0_dstar = (s_T * b_wcomp0_T - s_B * b_wcomp0_B) / (s_T - s_B) + (emf_wcomp2_LB - emf_wcomp2_LT + emf_wcomp2_RB - emf_wcomp2_RT) / (2.0 * (s_T - s_B));
			b_wcomp1_dstar = (s_R * b_wcomp1_R - s_L * b_wcomp1_L) / (s_R - s_L) + (-emf_wcomp2_LB - emf_wcomp2_LT + emf_wcomp2_RB + emf_wcomp2_RT) / (2.0 * (s_R - s_L));
			// Balsara25a eqs. 3.7 (x-flux) and 3.8 (y-flux); emf_wcomp2_dstar = average of both.
			const auto emf_wcomp2_dstar_1 = -(s_R + s_L) * b_wcomp1_dstar / 2.0 + (s_T * (emf_wcomp2_LB + emf_wcomp2_RB) - s_B * (emf_wcomp2_LT + emf_wcomp2_RT)) / (2.0 * (s_T - s_B)) -
						s_T * s_B * (b_wcomp0_B - b_wcomp0_T) / (s_T - s_B) + (s_R * b_wcomp1_R + s_L * b_wcomp1_L) / 2.0;
			const auto emf_wcomp2_dstar_2 = (s_T + s_B) * b_wcomp0_dstar / 2.0 + (s_R * (emf_wcomp2_LB + emf_wcomp2_LT) - s_L * (emf_wcomp2_RB + emf_wcomp2_RT)) / (2.0 * (s_R - s_L)) -
						(s_T * b_wcomp0_T + s_B * b_wcomp0_B) / 2.0 - s_R * s_L * (b_wcomp1_R - b_wcomp1_L) / (s_R - s_L);
			emf_wcomp2_dstar = 0.5 * (emf_wcomp2_dstar_1 + emf_wcomp2_dstar_2);
		} else {
			// LLF fallback: used when s_L==s_R or s_B==s_T (HLL denominator vanishes).
			// Balsara25a eqs. 3.3 (x) and 3.5 (y): LLF star states.
			emf_wcomp2_T_star = 0.5 * ((emf_wcomp2_LT + emf_wcomp2_RT) + s_max * (b_wcomp1_R - b_wcomp1_L));
			emf_wcomp2_B_star = 0.5 * ((emf_wcomp2_LB + emf_wcomp2_RB) + s_max * (b_wcomp1_R - b_wcomp1_L));
			emf_wcomp2_R_star = 0.5 * ((emf_wcomp2_RB + emf_wcomp2_RT) - s_max * (b_wcomp0_T - b_wcomp0_B));
			emf_wcomp2_L_star = 0.5 * ((emf_wcomp2_LB + emf_wcomp2_LT) - s_max * (b_wcomp0_T - b_wcomp0_B));
			// Balsara25a eq. 3.9: LLF double-star emf.
			emf_wcomp2_dstar = 0.5 * ((emf_wcomp2_RT + emf_wcomp2_LT + emf_wcomp2_LB + emf_wcomp2_RB) / 2.0 + s_max * (b_wcomp0_B - b_wcomp0_T + b_wcomp1_R - b_wcomp1_L));
		}

		// select state at the z-edge based on which speeds are zero (Balsara25a fig. 4).
		if (s_L == 0.0 && s_B == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_LB;
		} else if (s_R == 0.0 && s_B == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_RB;
		} else if (s_R == 0.0 && s_T == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_RT;
		} else if (s_L == 0.0 && s_T == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_LT;
		} else if (s_L == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_L_star;
		} else if (s_R == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_R_star;
		} else if (s_T == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_T_star;
		} else if (s_B == 0.0) {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_B_star;
		} else {
			a4_emf_wcomp2_ave(i, j, k) = emf_wcomp2_dstar;
		}

		MHDSystem<problem_t>::ApplyResistiveCorrection(a4_emf_wcomp2_ave, i, j, k, a4_b_wcomp0, a4_b_wcomp1, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	});
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_iValid,
					 int reconstructionOrder, SlopeLimiter plmLimiter)
{
	const BL_PROFILE("MHDSystem::ReconstructTo()");
	const amrex::IntVect dir_vec = amrex::IntVect::TheDimensionVector(static_cast<int>(dir));
	// PPM kernels loop over cells and fill left(i+1) and right(i); include one extra cell in the reconstruction direction
	const amrex::Box box_cell_range = amrex::grow(amrex::enclosedCells(box_iValid, static_cast<int>(dir)), dir_vec);
	const amrex::Box box_interface_range = amrex::surroundingNodes(box_cell_range, static_cast<int>(dir));
	if (reconstructionOrder == 5) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X1>(cState, lState, rState, box_cell_range, box_interface_range,
												    1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X2>(cState, lState, rState, box_cell_range, box_interface_range,
												    1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X3>(cState, lState, rState, box_cell_range, box_interface_range,
												    1);
				break;
		}
	} else if (reconstructionOrder == 3) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(cState, lState, rState, box_cell_range, box_interface_range,
												 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(cState, lState, rState, box_cell_range, box_interface_range,
												 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(cState, lState, rState, box_cell_range, box_interface_range,
												 1);
				break;
		}
	} else if (reconstructionOrder == 2) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1>(cState, lState, rState, box_cell_range, box_interface_range, 1,
												 plmLimiter);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2>(cState, lState, rState, box_cell_range, box_interface_range, 1,
												 plmLimiter);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3>(cState, lState, rState, box_cell_range, box_interface_range, 1,
												 plmLimiter);
				break;
		}
	} else if (reconstructionOrder == 1) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(cState, lState, rState, box_cell_range,
												      box_interface_range, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(cState, lState, rState, box_cell_range,
												      box_interface_range, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(cState, lState, rState, box_cell_range,
												      box_interface_range, 1);
				break;
		}
	} else {
		amrex::Abort("Invalid reconstruction order specified! Supported orders: 1 (constant), 2 (PLM), 3 (PPM), 5 (xPPM).");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, double dt,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	const BL_PROFILE("MHDSystem::SolveInductionEqn()");
	// compute the total right-hand-side for the MOL integration

	// flux sign convention: flux_(i) is into zone i from the left; -flux_(i+1) is into zone i from the right

	// loop over faces pointing in the wcomp0-direction
	for (int wcomp0 = 0; wcomp0 < 3; ++wcomp0) {
		// you have two edges on the perimeter of this face
		const int wcomp1 = (wcomp0 + 1) % 3; // vec_fc(wcomp0) + vec_fc(wcomp1)
		const int wcomp2 = (wcomp0 + 2) % 3; // vec_fc(wcomp0) + vec_fc(wcomp2)

		// direction to find the edges either side of the face. this depends on the direction the face points
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

		auto const dx1 = dx[wcomp1];
		auto const dx2 = dx[wcomp2];
		auto const ec_emf_wcomp1 = ec_emf_mf[wcomp1].const_arrays();
		auto const ec_emf_wcomp2 = ec_emf_mf[wcomp2].const_arrays();
		auto const fc_consVarOld = fc_consVarOld_mf[wcomp0].const_arrays();
		auto fc_consVarNew = fc_consVarNew_mf[wcomp0].arrays();

		amrex::ParallelFor(fc_consVarNew_mf[wcomp0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// the ec emfs sit in the opposite fc directions relative to the face
			const double emf_wcomp1_m = ec_emf_wcomp1[bx](i, j, k);
			const double emf_wcomp2_m = ec_emf_wcomp2[bx](i, j, k);
			const double emf_wcomp1_p = ec_emf_wcomp1[bx](i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2]);
			const double emf_wcomp2_p = ec_emf_wcomp2[bx](i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2]);
			const double db_dt = (dx1 * (emf_wcomp1_m - emf_wcomp1_p) + dx2 * (emf_wcomp2_p - emf_wcomp2_m)) / (dx1 * dx2);

			fc_consVarNew[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) =
			    fc_consVarOld[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto MHDSystem<problem_t>::computeResistiveEMF(amrex::Array4<const amrex::Real> const &a4_b_wcomp0,
										   amrex::Array4<const amrex::Real> const &a4_b_wcomp1, int i, int j, int k,
										   std::array<int, 3> const &delta_wcomp0, std::array<int, 3> const &delta_wcomp1,
										   amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity) -> amrex::Real
{
	const amrex::Real j_edge = (a4_b_wcomp1(i, j, k) - a4_b_wcomp1(i - delta_wcomp0[0], j - delta_wcomp0[1], k - delta_wcomp0[2])) / dx_wcomp0 -
				   (a4_b_wcomp0(i, j, k) - a4_b_wcomp0(i - delta_wcomp1[0], j - delta_wcomp1[1], k - delta_wcomp1[2])) / dx_wcomp1;
	return resistivity * j_edge;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
MHDSystem<problem_t>::ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &a4_emf_wcomp2_ave, int i, int j, int k, amrex::Array4<const amrex::Real> const &a4_b_wcomp0,
					       amrex::Array4<const amrex::Real> const &a4_b_wcomp1, std::array<int, 3> const &delta_wcomp0,
					       std::array<int, 3> const &delta_wcomp1, amrex::Real dx_wcomp0, amrex::Real dx_wcomp1, amrex::Real resistivity)
{
	if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
		a4_emf_wcomp2_ave(i, j, k) -= computeResistiveEMF(a4_b_wcomp0, a4_b_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
	} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
		const amrex::Real eta = computeResistivity<problem_t>(i, j, k, a4_b_wcomp0, a4_b_wcomp1, dx_wcomp0, dx_wcomp1);
		a4_emf_wcomp2_ave(i, j, k) -= computeResistiveEMF(a4_b_wcomp0, a4_b_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, eta);
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::AddResistiveEnergyFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real resistivity)
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

		std::array<int, 3> delta_wcomp0 = {0, 0, 0};
		std::array<int, 3> delta_wcomp1 = {0, 0, 0};
		std::array<int, 3> delta_wcomp2 = {0, 0, 0};
		delta_wcomp0[wcomp0] = 1;
		delta_wcomp1[wcomp1] = 1;
		delta_wcomp2[wcomp2] = 1;

		const amrex::Real dx_wcomp0 = dx[wcomp0];
		const amrex::Real dx_wcomp1 = dx[wcomp1];
		const amrex::Real dx_wcomp2 = dx[wcomp2];
		const int energy_idx = HydroSystem<problem_t>::energy_index;

		for (amrex::MFIter mfi(fluxArrays[wcomp0]); mfi.isValid(); ++mfi) {
			const amrex::Box &box_face = mfi.validbox();

			// fc_a4_b_wcomp_wcomp1 on wcomp1-faces, fc_a4_b_wcomp_wcomp2 on wcomp2-faces, fc_a4_b_wcomp_wcomp0 on wcomp0-faces (aliased, no copy)
			const auto fc_a4_b_wcomp_wcomp1 = fcx_mf_cVars[wcomp1][mfi].const_array(bfield_index);
			const auto fc_a4_b_wcomp_wcomp2 = fcx_mf_cVars[wcomp2][mfi].const_array(bfield_index);
			const auto fc_a4_b_wcomp_wcomp0 = fcx_mf_cVars[wcomp0][mfi].const_array(bfield_index);
			auto fc_a4_flux = fluxArrays[wcomp0][mfi].array();

			// wcomp1-edge: a4_b_wcomp0=fc_a4_b_wcomp_wcomp2, delta_wcomp0=delta_wcomp2, dx_wcomp0=dx_wcomp2; a4_b_wcomp1=fc_a4_b_wcomp_wcomp0, delta_wcomp1=delta_wcomp0, dx_wcomp1=dx_wcomp0
			// wcomp2-edge: a4_b_wcomp0=fc_a4_b_wcomp_wcomp0, delta_wcomp0=delta_wcomp0, dx_wcomp0=dx_wcomp0; a4_b_wcomp1=fc_a4_b_wcomp_wcomp1, delta_wcomp1=delta_wcomp1, dx_wcomp1=dx_wcomp1
			amrex::ParallelFor(box_face, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real eta_j_wcomp1_lo = 0.0;
				amrex::Real eta_j_wcomp1_hi = 0.0;
				amrex::Real eta_j_wcomp2_lo = 0.0;
				amrex::Real eta_j_wcomp2_hi = 0.0;
				if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
					eta_j_wcomp1_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i, j, k, delta_wcomp2, delta_wcomp0, dx_wcomp2,
									  dx_wcomp0, resistivity);
					eta_j_wcomp1_hi = computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i + delta_wcomp2[0], j + delta_wcomp2[1],
									  k + delta_wcomp2[2], delta_wcomp2, delta_wcomp0, dx_wcomp2, dx_wcomp0, resistivity);
					eta_j_wcomp2_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0,
									  dx_wcomp1, resistivity);
					eta_j_wcomp2_hi = computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i + delta_wcomp1[0], j + delta_wcomp1[1],
									  k + delta_wcomp1[2], delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, resistivity);
				} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
					const amrex::Real eta_wcomp1_lo = computeResistivity<problem_t>(i, j, k, fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, dx_wcomp2, dx_wcomp0);
					eta_j_wcomp1_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i, j, k, delta_wcomp2, delta_wcomp0, dx_wcomp2,
									  dx_wcomp0, eta_wcomp1_lo);
					const amrex::Real eta_wcomp1_hi =
					    computeResistivity<problem_t>(i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2], fc_a4_b_wcomp_wcomp2,
									  fc_a4_b_wcomp_wcomp0, dx_wcomp2, dx_wcomp0);
					eta_j_wcomp1_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp2, fc_a4_b_wcomp_wcomp0, i + delta_wcomp2[0], j + delta_wcomp2[1], k + delta_wcomp2[2],
								delta_wcomp2, delta_wcomp0, dx_wcomp2, dx_wcomp0, eta_wcomp1_hi);
					const amrex::Real eta_wcomp2_lo = computeResistivity<problem_t>(i, j, k, fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, dx_wcomp0, dx_wcomp1);
					eta_j_wcomp2_lo = computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i, j, k, delta_wcomp0, delta_wcomp1, dx_wcomp0,
									  dx_wcomp1, eta_wcomp2_lo);
					const amrex::Real eta_wcomp2_hi =
					    computeResistivity<problem_t>(i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2], fc_a4_b_wcomp_wcomp0,
									  fc_a4_b_wcomp_wcomp1, dx_wcomp0, dx_wcomp1);
					eta_j_wcomp2_hi =
					    computeResistiveEMF(fc_a4_b_wcomp_wcomp0, fc_a4_b_wcomp_wcomp1, i + delta_wcomp1[0], j + delta_wcomp1[1], k + delta_wcomp1[2],
								delta_wcomp0, delta_wcomp1, dx_wcomp0, dx_wcomp1, eta_wcomp2_hi);
				}

				// average face-b to each edge position across the wcomp0-direction.
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

				// f_eta = (eta_j x b)_wcomp0 = eta_j_wcomp1 * b_wcomp2 - eta_j_wcomp2 * b_wcomp1, averaged over lo and hi bounding edges.
				const amrex::Real f_eta = 0.25 * (eta_j_wcomp1_lo * ave_b_wcomp2_lo + eta_j_wcomp1_hi * ave_b_wcomp2_hi - eta_j_wcomp2_lo * ave_b_wcomp1_lo - eta_j_wcomp2_hi * ave_b_wcomp1_hi);
				fc_a4_flux(i, j, k, energy_idx) += f_eta;
			});
		}
	}
}

#endif // HYDRO_SYSTEM_HPP_
