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

AMREX_ENUM(EMFComputeScheme, FelkerStone2017, Balsara2025, Quokka2026); // NOLINT

// Felker + Stone (2017): uses cell-centered velocity
// Balsara (2025): EMF interpolation from cc->ec
// Quokka variant of FS17: uses face-centered Riemann velocity

AMREX_ENUM(EMFAvgScheme, LondrilloDelZanna2004, Balsara2025); // NOLINT
// Londrillo + Del Zanna (2004)
// Balsara (2025): Higher-order averaging

AMREX_FORCE_INLINE constexpr auto MinimumHydroRiemannGhost(bool is_mhd_enabled, EMFComputeScheme emf_compute_scheme, EMFAvgScheme emf_avg_scheme,
							   bool require_tracer_ghosts = false) -> int
{
	int nghost = require_tracer_ghosts ? 2 : 0;
	if (is_mhd_enabled) {
		if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
			nghost = std::max(nghost, 3);
		} else {
			switch (emf_avg_scheme) {
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

	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_avg_scheme,
			       SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
			       amrex::Real resistivity = 0.0);

	static void AverageEMF(amrex::Array4<amrex::Real> const &a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q, amrex::Box const &box_ec,
			       std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
			       std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside, EMFAvgScheme emf_avg_scheme,
			       amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1, amrex::Real dx_w0, amrex::Real dx_w1,
			       amrex::Real resistivity);

	static void ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
					       SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					       amrex::Real resistivity = 0.0);

	static void ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter,
					   EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real resistivity = 0.0);

	static void ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter,
					  EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Real resistivity = 0.0);

	static void EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q,
						     amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
						     std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
						     std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside,
						     amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1,
						     amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity);

	static void EMFAverage_Balsara2025(amrex::Array4<amrex::Real> a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q, amrex::Box const &box_ec,
					   std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
					   std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside,
					   amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1, amrex::Real dx_w0,
					   amrex::Real dx_w1, amrex::Real resistivity);

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto computeResistiveEMF(amrex::Array4<const amrex::Real> const &a4_b_w0,
									    amrex::Array4<const amrex::Real> const &a4_b_w1, int i, int j, int k,
									    std::array<int, 3> const &delta_w0, std::array<int, 3> const &delta_w1,
									    amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity) -> amrex::Real;

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &a4_emf2_ave, int i, int j, int k,
										 amrex::Array4<const amrex::Real> const &a4_b_w0,
										 amrex::Array4<const amrex::Real> const &a4_b_w1,
										 std::array<int, 3> const &delta_w0, std::array<int, 3> const &delta_w1,
										 amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity);

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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeResistivity(int /*i*/, int /*j*/, int /*k*/, amrex::Array4<const amrex::Real> const & /*a4_b_w0*/,
							    amrex::Array4<const amrex::Real> const & /*a4_b_w1*/, amrex::Real /*dx_w0*/, amrex::Real /*dx_w1*/)
    -> amrex::Real
{
	static_assert(sizeof(problem_t) == 0, "computeResistivity must be specialized in the problem file when using ResistivityModel::problem_defined");
	return 0.0;
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_avg_scheme,
				      SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
				      amrex::Real resistivity)
{
	if (emf_compute_scheme == EMFComputeScheme::FelkerStone2017) {
		MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(ec_mf_emf_components, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
								 emf_avg_scheme, dx, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Balsara2025) {
		MHDSystem<problem_t>::ComputeEMF_Balsara2025(ec_mf_emf_components, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
							     emf_avg_scheme, dx, resistivity);
	} else if (emf_compute_scheme == EMFComputeScheme::Quokka2026) {
		MHDSystem<problem_t>::ComputeEMF_Quokka2026(ec_mf_emf_components, fcx_mf_vel, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, plmLimiter,
							    emf_avg_scheme, dx, resistivity);
	} else {
		throw std::runtime_error("Unsupported EMF-scheme. Expected either FelkerStone2017, Balsara2025, or Quokka2026.");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::AverageEMF(amrex::Array4<amrex::Real> const &a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q, amrex::Box const &box_ec,
				      std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
				      std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside, EMFAvgScheme emf_avg_scheme,
				      amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1, amrex::Real dx_w0,
				      amrex::Real dx_w1, amrex::Real resistivity)
{
	if (emf_avg_scheme == EMFAvgScheme::LondrilloDelZanna2004) {
		EMFAverage_LondrilloDelZanna2004(a4_emf2_ave, ec_fabs_emf_q, box_ec, extrap_dirs, fspds, ec_fabs_bi_ieside, a4_b_w0, a4_b_w1, dx_w0, dx_w1, resistivity);
	} else if (emf_avg_scheme == EMFAvgScheme::Balsara2025) {
		EMFAverage_Balsara2025(a4_emf2_ave, ec_fabs_emf_q, box_ec, extrap_dirs, fspds, ec_fabs_bi_ieside, a4_b_w0, a4_b_w1, dx_w0, dx_w1, resistivity);
	} else {
		amrex::Abort("Unknown EMF averaging type");
	}
}

// EMF solver from Felker & Stone (2017)
// uses cell-centered velocity and face-centered magnetic fields extrapolated to the cell-edge to compute the EMF

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						      SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						      amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FelkerStone2017()");
	const int nghost_cc = 4; // we only need 4 cc ghost cells when reconstructing cc->fc->ec using PPM
	// note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
	// note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells

	// In this function we distinguish between world (w:3), array (i:2), quandrant (qi:4), and component (x:3) index-ing by using prefixes. We will
	// use the prefix x- when the w- and i- indexes are the same. We also choose to minimise the storage footprint by only computing and holding
	// onto the quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but allows us to
	// significantly reduces the total memory used, which is a much bigger bottleneck.

	// loop over each box-array on this level
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// extract cell-centered velocity fields
		// indexing: field[3: x-component]
		const amrex::Box &box_cc_u = amrex::grow(box_cc, nghost_cc);
		std::array<amrex::FArrayBox, 3> cc_fabs_ux = {amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_u, 1, amrex::The_Async_Arena())};
		{
			const auto &cc_a4_ux0 = cc_fabs_ux[0].array();
			const auto &cc_a4_ux1 = cc_fabs_ux[1].array();
			const auto &cc_a4_ux2 = cc_fabs_ux[2].array();
			const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();

			amrex::ParallelFor(box_cc_u, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
				const auto px1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				const auto px2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				const auto px3 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				cc_a4_ux0(i, j, k) = px1 / rho;
				cc_a4_ux1(i, j, k) = px2 / rho;
				cc_a4_ux2(i, j, k) = px3 / rho;
			});
		}

		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// compute the magnetic flux through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {
			// for each of the two cell-edges on the cell-face
			// we are doing redundant compute. only need to look at one edge for each face: there is a one-to-one mapping.

			// define the two directions we need to extrapolate cell-centered velocity fields to get them to the cell-edge
			// we will want to compute E2 = (u0 * b1 - u1 * b0) along the cell-edge
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
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bi_ieside;

			// initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
			// indexing: field[2: i-compnent][4: quadrant around edge]
			std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_ui_q;

			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_bi_ieside[icomp][0] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				ec_fabs_bi_ieside[icomp][1] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				for (int iquad = 0; iquad < 4; ++iquad) {
					ec_fabs_ui_q[icomp][iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					ec_fabs_ui_q[icomp][iquad].setVal<amrex::RunOn::Device>(0.0);
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
					MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_ux[wcomp].array(), fc_fabs_u_ifside[0].array(),
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

						ec_fabs_ui_q[icomp][iquad0].plus<amrex::RunOn::Device>(ec_fabs_u_ieside[0], 0, 0, 1);
						ec_fabs_ui_q[icomp][iquad1].plus<amrex::RunOn::Device>(ec_fabs_u_ieside[1], 0, 0, 1);
					}
				}
			}

			// finish averaging the two different ways for extrapolating velocity fields: cc->fc->ec
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int iquad = 0; iquad < 4; ++iquad) {
					ec_fabs_ui_q[icomp][iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
				}
			}

			// extrapolate the two required face-centered magnetic field components to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int extrap_dir2edge = extrap_dirs[(icomp + 1) % 2];
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const int wcomp = extrap_dirs[icomp];
				// extrapolate face-centered magnetic components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_bx[wcomp].array(), ec_fabs_bi_ieside[icomp][0].array(),
								    ec_fabs_bi_ieside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_q;

			// compute the EMF along the cell-edge using a single kernel (all quadrants inside)
			{
				// bind read/write Array4 views on the host (required for GPU lambda capture)
				std::array<amrex::Array4<const amrex::Real>, 4> u0s;
				std::array<amrex::Array4<const amrex::Real>, 4> u1s;
				std::array<amrex::Array4<const amrex::Real>, 4> b0s;
				std::array<amrex::Array4<const amrex::Real>, 4> b1s;
				std::array<amrex::Array4<amrex::Real>, 4> E2s;

				for (int qi = 0; qi < 4; ++qi) {
					// extract relevant velocity and magnetic field components (host: get Array4 views)
					const int idx0 = (qi == 0 || qi == 3) ? 0 : 1;	    // B/T selector for dir-0
					const int idx1 = (qi < 2) ? 0 : 1;		    // L/R selector for dir-1
					u0s[qi] = ec_fabs_ui_q[0][qi].const_array();	    // component 0, index iquad
					u1s[qi] = ec_fabs_ui_q[1][qi].const_array();	    // component 1, index iquad
					b0s[qi] = ec_fabs_bi_ieside[0][idx0].const_array(); // component 0, index idx0
					b1s[qi] = ec_fabs_bi_ieside[1][idx1].const_array(); // component 1, index idx1

					// define EMF FArrayBox for each quadrant (we need to allocate outside the kernel)
					ec_fabs_emf_q[qi] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
					E2s[qi] = ec_fabs_emf_q[qi].array();
				}

				// single kernel over the edge-centered box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int qi = 0; qi < 4; ++qi) {
						const amrex::Real u0 = u0s[qi](i, j, k);
						const amrex::Real u1 = u1s[qi](i, j, k);
						const amrex::Real b0 = b0s[qi](i, j, k);
						const amrex::Real b1 = b1s[qi](i, j, k);
						E2s[qi](i, j, k) = u0 * b1 - u1 * b0; // cross product at the edge
					}
				});
			}

			// compute electric field on the cell-edge
			const auto &a4_emf2_ave = ec_mf_emf_components[iedge][mfi].array();

			// selected averaging method for EMF:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf2_ave, ec_fabs_emf_q, box_ec, extrap_dirs, fspds, ec_fabs_bi_ieside, emf_avg_scheme,
							 fcx_mf_cVars[extrap_dirs[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[extrap_dirs[1]][mfi].const_array(bfield_index), dx[extrap_dirs[0]], dx[extrap_dirs[1]],
							 resistivity);
		}
	}
}

// EMF solver Quokka variant (2026) of Felker & Stone (2017)
// uses face-centered Riemann velocity and face-centered magnetic fields extrapolated to the cell-edge to compute the EMF

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						 SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						 amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_Quokka2026()");

	// loop over each box-array on the level
	// note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
	// note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	for (amrex::MFIter mfi(fcx_mf_cVars[0], amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// In this function we distinguish between world (w:3), array (i:2), quandrant (qi:4), and component (x:3) index-ing by using prefixes. We will
		// use the prefix x- when the w- and i- indexes are the same. We also choose to minimise the storage footprint by only computing and holding
		// onto the quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but allows us to
		// significantly reduces the total memory used, which is a much bigger bottleneck.

		// indexing: field[3: x-component/x-face]
		// create a view of all the u-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_ux = {
		    amrex::FArrayBox(fcx_mf_vel[0][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[1][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[2][mfi], amrex::make_alias, 0, 1),
		};
		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};
		// compute the emf components on the cell-edge to inform how much magnetic flux travels through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {

			// define the two face-centered velocity/magnetic field components we need at the cell-edge
			// we will want to compute E2 = (u0 * b1 - u1 * b0) along the cell-edge
			std::array<int, 2> field_w_indices = {(iedge + 1) % 3, (iedge + 2) % 3};
			const amrex::Box box_ec = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(field_w_indices[0]) +
									     amrex::IntVect::TheDimensionVector(field_w_indices[1]));
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// FArrayBoxes for storing the edge-centered fields produced by reconstructing from the cell-face to the cell-edge
			// indexing: field[2: i-component][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_ui_ieside;
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bi_ieside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int ieside = 0; ieside < 2; ++ieside) {
					ec_fabs_ui_ieside[icomp][ieside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
					ec_fabs_bi_ieside[icomp][ieside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
			}

			// extrapolate the face-centered fields (normal to the cell-face) to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(field_w_indices[(icomp + 1) % 2]);
				const int wcomp = field_w_indices[icomp];
				// extrapolate face-centered components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_bx[wcomp].array(), ec_fabs_bi_ieside[icomp][0].array(),
								    ec_fabs_bi_ieside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_ux[wcomp].array(), ec_fabs_ui_ieside[icomp][0].array(),
								    ec_fabs_ui_ieside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_q;
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
			// |       /       \       q_1 + q_2 | q_3 + q_0                                                 |
			// |       u,b_{0,B}                 |                                                           |
			// |          q_0                                                                                |
			// |---------------------------------------------------------------------------------------------|
			// compute the EMF along the cell-edge using a single kernel (all quadrants inside)
			{
				// bind read/write Array4 views on the host (required for GPU lambda capture)
				std::array<amrex::Array4<const amrex::Real>, 4> u0s;
				std::array<amrex::Array4<const amrex::Real>, 4> u1s;
				std::array<amrex::Array4<const amrex::Real>, 4> b0s;
				std::array<amrex::Array4<const amrex::Real>, 4> b1s;
				std::array<amrex::Array4<amrex::Real>, 4> E2s;

				for (int qi = 0; qi < 4; ++qi) {
					const int idx0 = (qi == 0 || qi == 3) ? 0 : 1; // B/T selector for dir-0
					const int idx1 = (qi < 2) ? 0 : 1;	       // L/R selector for dir-1

					// define EMF FArrayBox for each quadrant (we need to allocate outside the kernel)
					ec_fabs_emf_q[qi] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());

					// extract relevant velocity and magnetic field components (host: get Array4 views)
					u0s[qi] = ec_fabs_ui_ieside[0][idx0].const_array(); // B/T
					b0s[qi] = ec_fabs_bi_ieside[0][idx0].const_array(); // B/T
					u1s[qi] = ec_fabs_ui_ieside[1][idx1].const_array(); // L/R
					b1s[qi] = ec_fabs_bi_ieside[1][idx1].const_array(); // L/R
					E2s[qi] = ec_fabs_emf_q[qi].array();		    // output EMF view
				}

				// single kernel over the edge-centered box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int qi = 0; qi < 4; ++qi) {
						const amrex::Real u0 = u0s[qi](i, j, k);
						const amrex::Real u1 = u1s[qi](i, j, k);
						const amrex::Real b0 = b0s[qi](i, j, k);
						const amrex::Real b1 = b1s[qi](i, j, k);
						E2s[qi](i, j, k) = u0 * b1 - u1 * b0; // cross product at the edge
					}
				});
			}

			const auto &a4_emf2_ave = ec_mf_emf_components[iedge][mfi].array();

			// selected averaging method for the emf:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf2_ave, ec_fabs_emf_q, box_ec, field_w_indices, fspds, ec_fabs_bi_ieside, emf_avg_scheme,
							 fcx_mf_cVars[field_w_indices[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[field_w_indices[1]][mfi].const_array(bfield_index), dx[field_w_indices[0]],
							 dx[field_w_indices[1]], resistivity);
		}
	}
}

// EMF solver Balsara (2025)
// uses cell-centered velocity and face-centered magnetic fields averaged to cell-center to compute the EMF at cell center,
// then extrapolate the EMF to the cell-edge

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
						  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
						  SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
						  amrex::Real resistivity)
{
	// calculating v x B at cell center, v already at cell center, B at face center

	const BL_PROFILE("MHDSystem::ComputeEMF_Balsara2025()");
	const int nghost_cc = 4;
	// note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells

	const auto &ba = cc_mf_cVars.boxArray();
	const auto &dm = cc_mf_cVars.DistributionMap();
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	amrex::MultiFab cc_mf_emf(ba, dm, 3, nghost_cc);
	cc_mf_emf.setVal(0.0, 0, 3, nghost_cc); // initialize to zero everywhere including ghost zones

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc_emf = mfi.growntilebox(nghost_cc); // Ensure enough ghost cells for EMF computation

		// Access the MultiFab components for this tile
		const auto &cc_a4_emfx0 = cc_mf_emf[mfi].array(0);
		const auto &cc_a4_emfx1 = cc_mf_emf[mfi].array(1);
		const auto &cc_a4_emfx2 = cc_mf_emf[mfi].array(2);

		const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
		std::array<amrex::Array4<amrex::Real>, 3> const cc_a4_emf_array = {cc_a4_emfx0, cc_a4_emfx1, cc_a4_emfx2};
		std::array<amrex::FArrayBox, 3> fc_fabs_bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// Get face-centered B field arrays
		std::array<amrex::Array4<amrex::Real const>, 3> fc_a4_bx = {fcx_mf_cVars[0][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[1][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[2][mfi].const_array(MHDSystem<problem_t>::bfield_index)};

		// Compute v x B for all three directions in parallel
		amrex::ParallelFor(box_cc_emf, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
			std::array<amrex::Real, 3> v = {cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho,
							cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho,
							cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho};
			// Loop over each dimension to compute the corresponding EMF component
			for (int idim = 0; idim < 3; ++idim) {
				int const x2ind = (idim + 1) % 3;
				int const x3ind = (idim + 2) % 3;

				std::array<int, 3> delta_x2 = {0, 0, 0};
				std::array<int, 3> delta_x3 = {0, 0, 0};
				delta_x2[x2ind] = 1;
				delta_x3[x3ind] = 1;

				// Average face-centered B to cell center
				amrex::Real const bx2_avg =
				    0.5 * (fc_a4_bx[x2ind](i, j, k) + fc_a4_bx[x2ind](i + delta_x2[0], j + delta_x2[1], k + delta_x2[2]));
				amrex::Real const bx3_avg =
				    0.5 * (fc_a4_bx[x3ind](i, j, k) + fc_a4_bx[x3ind](i + delta_x3[0], j + delta_x3[1], k + delta_x3[2]));

				// v x B computation
				cc_a4_emf_array[idim](i, j, k) = v[x2ind] * bx3_avg - v[x3ind] * bx2_avg; //
			}
		});
	}

	cc_mf_emf.FillBoundary(); // fill ghost cells
	amrex::Gpu::streamSynchronize();

	// now that EMF is calculated at cell center, we need to interpolate to cell edge
	// we also need to get the magnetic field from the face to cell edge for the Balsara2025 or LondrilloDelZanna2004 solvers

	// In this part we distinguish between world (w:3), array (i:2), quandrant (q:4), and component (x:3) index-ing by using prefixes. We will
	// use the prefix x- when the w- and i- indexes are the same.

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) { // keep
		const amrex::Box &box_cc = mfi.validbox();

		std::array<amrex::FArrayBox, 3> fc_fabs_bx = {
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
			const auto &a4_emf2_ave = ec_mf_emf_components[iedge][mfi].array();
			std::array<amrex::FArrayBox, 2> ec_fabs_emf_ieside = {amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena()),
									      amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena())};

			ec_fabs_emf_ieside[0].setVal<amrex::RunOn::Device>(0.0);
			ec_fabs_emf_ieside[1].setVal<amrex::RunOn::Device>(0.0);
			std::array<amrex::FArrayBox, 4> ec_fabs_emf_q;

			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emf_q[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
				ec_fabs_emf_q[iquad].setVal<amrex::RunOn::Device>(0.0);
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

					ec_fabs_emf_q[iquad0].plus<amrex::RunOn::Device>(ec_fabs_emf_ieside[0], 0, 0, 1);
					ec_fabs_emf_q[iquad1].plus<amrex::RunOn::Device>(ec_fabs_emf_ieside[1], 0, 0, 1);
				}
			}
			// finish averaging the two different ways for extrapolating emf: cc->fc->ec
			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_emf_q[iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
			}

			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_bi_ieside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int ieside = 0; ieside < 2; ++ieside) {
					ec_fabs_bi_ieside[icomp][ieside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
			}

			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(extrap_dirs[(icomp + 1) % 2]);
				const int wcomp = extrap_dirs[icomp];
				// extrapolate face-centered components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_bx[wcomp].array(), ec_fabs_bi_ieside[icomp][0].array(),
								    ec_fabs_bi_ieside[icomp][1].array(), box_ec, reconstructionOrder, plmLimiter);
			}
			// selected averaging method for the emf:
			std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
										       fcx_mf_fspds[2].const_array(mfi)};
			MHDSystem<problem_t>::AverageEMF(a4_emf2_ave, ec_fabs_emf_q, box_ec, extrap_dirs, fspds, ec_fabs_bi_ieside, emf_avg_scheme,
							 fcx_mf_cVars[extrap_dirs[0]][mfi].const_array(bfield_index),
							 fcx_mf_cVars[extrap_dirs[1]][mfi].const_array(bfield_index), dx[extrap_dirs[0]], dx[extrap_dirs[1]],
							 resistivity);
		}
	}
}

// more complex emf solver: uses information about the fast wave speeds to do a weighted average of the quadrants
// from: Londrillo & Del Zanna 2004, JCP, 195
template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q,
							    amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
							    std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
							    std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside,
							    amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1,
							    amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_LondrilloDelZanna2004()");

	const auto &a4_emf2_q0 = ec_fabs_emf_q[0].const_array();
	const auto &a4_emf2_q1 = ec_fabs_emf_q[1].const_array();
	const auto &a4_emf2_q2 = ec_fabs_emf_q[2].const_array();
	const auto &a4_emf2_q3 = ec_fabs_emf_q[3].const_array();

	const auto &a4_b0_m = ec_fabs_bi_ieside[0][0].const_array();
	const auto &a4_b0_p = ec_fabs_bi_ieside[0][1].const_array();
	const auto &a4_b1_m = ec_fabs_bi_ieside[1][0].const_array();
	const auto &a4_b1_p = ec_fabs_bi_ieside[1][1].const_array();

	int const w0_comp = extrap_dirs[0];
	int const w1_comp = extrap_dirs[1];
	std::array<int, 3> delta_w0 = {0, 0, 0};
	std::array<int, 3> delta_w1 = {0, 0, 0};

	delta_w0[w0_comp] = 1;
	delta_w1[w1_comp] = 1;

	const auto &a4_fspd_w0 = fspds[w0_comp];
	const auto &a4_fspd_w1 = fspds[w1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// 	// LondrilloDelZanna2004 scheme:
		const double a0_m = std::max(a4_fspd_w0(i, j, k, 0), a4_fspd_w0(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2], 0));
		const double a0_p = std::max(a4_fspd_w0(i, j, k, 1), a4_fspd_w0(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2], 1));
		const double a1_m = std::max(a4_fspd_w1(i, j, k, 0), a4_fspd_w1(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2], 0));
		const double a1_p = std::max(a4_fspd_w1(i, j, k, 1), a4_fspd_w1(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2], 1));

		const double emf2_q0 = a4_emf2_q0(i, j, k);
		const double emf2_q1 = a4_emf2_q1(i, j, k);
		const double emf2_q2 = a4_emf2_q2(i, j, k);
		const double emf2_q3 = a4_emf2_q3(i, j, k);

		const double b0_T = a4_b0_p(i, j, k);
		const double b0_B = a4_b0_m(i, j, k);
		const double b1_R = a4_b1_p(i, j, k);
		const double b1_L = a4_b1_m(i, j, k);
		// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
		// (-,+) | (+,+)
		//   1   |   2
		// ------+------
		//   0   |   3
		// (-,-) | (+,-)

		const double num1 = ((a0_p * a1_p) * emf2_q0 + (a0_m * a1_p) * emf2_q3) + ((a0_p * a1_m) * emf2_q1 + (a0_m * a1_m) * emf2_q2);
		const double num2 = ((a0_p * a1_p) * emf2_q0 + (a0_p * a1_m) * emf2_q1) + ((a0_m * a1_p) * emf2_q3 + (a0_m * a1_m) * emf2_q2);

		// 	// must be averaged for exact floating-point symmetry
		const double numerator = 0.5 * (num1 + num2);
		const double denominator = (a0_m + a0_p) * (a1_m + a1_p);

		// Felker18a eq. 41 (= LD2004 eq. 56); a0_m<=0,a0_p>=0 are signed speeds (not negated like in b25).
		const double term2 = ((a1_m * a1_p) / (a1_m + a1_p)) * (b0_T - b0_B) + ((a0_m * a0_p) / (a0_m + a0_p)) * (b1_L - b1_R);

		a4_emf2_ave(i, j, k) = (numerator / denominator) + term2;
		MHDSystem<problem_t>::ApplyResistiveCorrection(a4_emf2_ave, i, j, k, a4_b_w0, a4_b_w1, delta_w0, delta_w1, dx_w0, dx_w1, resistivity);
	});
}

// solver from Balsara et al. 2025, 988, 134B

template <typename problem_t>
void MHDSystem<problem_t>::EMFAverage_Balsara2025(amrex::Array4<amrex::Real> a4_emf2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_emf_q,
						  amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs,
						  std::array<amrex::Array4<const amrex::Real>, 3> const &fspds,
						  std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_bi_ieside,
						  amrex::Array4<const amrex::Real> const &a4_b_w0, amrex::Array4<const amrex::Real> const &a4_b_w1, amrex::Real dx_w0,
						  amrex::Real dx_w1, amrex::Real resistivity)
{
	const BL_PROFILE("MHDSystem::EMFAverage_Balsara2025()");
	const auto &a4_emf2_q0 = ec_fabs_emf_q[0].const_array();
	const auto &a4_emf2_q1 = ec_fabs_emf_q[1].const_array();
	const auto &a4_emf2_q2 = ec_fabs_emf_q[2].const_array();
	const auto &a4_emf2_q3 = ec_fabs_emf_q[3].const_array();

	const auto &a4_b0_m = ec_fabs_bi_ieside[0][0].const_array();
	const auto &a4_b0_p = ec_fabs_bi_ieside[0][1].const_array();
	const auto &a4_b1_m = ec_fabs_bi_ieside[1][0].const_array();
	const auto &a4_b1_p = ec_fabs_bi_ieside[1][1].const_array();

	int const w0_comp = extrap_dirs[0];
	int const w1_comp = extrap_dirs[1];
	std::array<int, 3> delta_w0 = {0, 0, 0};
	std::array<int, 3> delta_w1 = {0, 0, 0};

	delta_w0[w0_comp] = 1;
	delta_w1[w1_comp] = 1;

	const auto &a4_fspd_w0 = fspds[w0_comp];
	const auto &a4_fspd_w1 = fspds[w1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		// Wave speeds: max over two adjacent faces per direction (Felker18a App. A); s_L,s_R,s_T,s_B are non-negative magnitudes.
		const double s_L = -std::max(a4_fspd_w0(i, j, k, 0), a4_fspd_w0(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2], 0));
		const double s_R = std::max(a4_fspd_w0(i, j, k, 1), a4_fspd_w0(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2], 1));
		const double s_B = -std::max(a4_fspd_w1(i, j, k, 0), a4_fspd_w1(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2], 0));
		const double s_T = std::max(a4_fspd_w1(i, j, k, 1), a4_fspd_w1(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2], 1));

		// EMF quadrants
		const auto emf2_LB = a4_emf2_q0(i, j, k);
		const auto emf2_LT = a4_emf2_q1(i, j, k);
		const auto emf2_RT = a4_emf2_q2(i, j, k);
		const auto emf2_RB = a4_emf2_q3(i, j, k);

		// Magnetic field components
		const auto b0_T = a4_b0_m(i, j, k);
		const auto b0_B = a4_b0_p(i, j, k);
		const auto b1_R = a4_b1_m(i, j, k);
		const auto b1_L = a4_b1_p(i, j, k);

		const auto s_max = std::max({s_L, s_R, s_B, s_T});

		double emf2_T_star = 0.0;
		double emf2_B_star = 0.0;
		double b1_dstar = 0.0;
		double emf2_R_star = 0.0;
		double emf2_L_star = 0.0;
		double b0_dstar = 0.0;
		double emf2_dstar = 0.0;

		if (s_L != s_R && s_B != s_T) {
			// eq. 3.2: x-directional HLL star states (T = upper, B = lower)
			emf2_T_star = (s_R * emf2_LT - s_L * emf2_RT) / (s_R - s_L) - (s_R * s_L) * (b1_R - b1_L) / (s_R - s_L);
			emf2_B_star = (s_R * emf2_LB - s_L * emf2_RB) / (s_R - s_L) - (s_R * s_L) * (b1_R - b1_L) / (s_R - s_L);
			// eq. 3.4: y-directional HLL star states (R = right, L = left)
			emf2_R_star = (s_T * emf2_RB - s_B * emf2_RT) / (s_T - s_B) + (s_T * s_B) * (b0_T - b0_B) / (s_T - s_B);
			emf2_L_star = (s_T * emf2_LB - s_B * emf2_LT) / (s_T - s_B) + (s_T * s_B) * (b0_T - b0_B) / (s_T - s_B);
			// eq. 3.6: double-star B fields
			b0_dstar = (s_T * b0_T - s_B * b0_B) / (s_T - s_B) + (emf2_LB - emf2_LT + emf2_RB - emf2_RT) / (2.0 * (s_T - s_B));
			b1_dstar = (s_R * b1_R - s_L * b1_L) / (s_R - s_L) + (-emf2_LB - emf2_LT + emf2_RB + emf2_RT) / (2.0 * (s_R - s_L));
			// eq. 3.7: E_z** via x-flux; eq. 3.8: E_z** via y-flux; final = average (last line of sec. 3)
			const auto emf2_dstar_1 = -(s_R + s_L) * b1_dstar / 2.0 + (s_T * (emf2_LB + emf2_RB) - s_B * (emf2_LT + emf2_RT)) / (2.0 * (s_T - s_B)) -
						s_T * s_B * (b0_B - b0_T) / (s_T - s_B) + (s_R * b1_R + s_L * b1_L) / 2.0;
			const auto emf2_dstar_2 = (s_T + s_B) * b0_dstar / 2.0 + (s_R * (emf2_LB + emf2_LT) - s_L * (emf2_RB + emf2_RT)) / (2.0 * (s_R - s_L)) -
						(s_T * b0_T + s_B * b0_B) / 2.0 - s_R * s_L * (b1_R - b1_L) / (s_R - s_L);
			emf2_dstar = 0.5 * (emf2_dstar_1 + emf2_dstar_2);
		} else {
		    // LLF fallback when s_L==s_R or s_B==s_T (HLL denominator vanishes)
			// eqs. 3.3, 3.5: LLF star states
			emf2_T_star = 0.5 * ((emf2_LT + emf2_RT) + s_max * (b1_R - b1_L));
			emf2_B_star = 0.5 * ((emf2_LB + emf2_RB) + s_max * (b1_R - b1_L));
			emf2_R_star = 0.5 * ((emf2_RB + emf2_RT) - s_max * (b0_T - b0_B));
			emf2_L_star = 0.5 * ((emf2_LB + emf2_LT) - s_max * (b0_T - b0_B));
			// eq. 3.9: LLF double-star
			emf2_dstar = 0.5 * ((emf2_RT + emf2_LT + emf2_LB + emf2_RB) / 2.0 + s_max * (b0_B - b0_T + b1_R - b1_L));
		}

		// Select state overlying the z-axis based on which speeds are zero (Fig. 4 of Balsara25a).
		if (s_L == 0.0 && s_B == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_LB;
		} else if (s_R == 0.0 && s_B == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_RB;
		} else if (s_R == 0.0 && s_T == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_RT;
		} else if (s_L == 0.0 && s_T == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_LT;
		} else if (s_L == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_L_star;
		} else if (s_R == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_R_star;
		} else if (s_T == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_T_star;
		} else if (s_B == 0.0) {
			a4_emf2_ave(i, j, k) = emf2_B_star;
		} else {
			a4_emf2_ave(i, j, k) = emf2_dstar;
		}

		MHDSystem<problem_t>::ApplyResistiveCorrection(a4_emf2_ave, i, j, k, a4_b_w0, a4_b_w1, delta_w0, delta_w1, dx_w0, dx_w1, resistivity);
	});
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_iValid,
					 int reconstructionOrder, SlopeLimiter plmLimiter)
{
	const BL_PROFILE("MHDSystem::ReconstructTo()");
	const amrex::IntVect dir_vec = amrex::IntVect::TheDimensionVector(static_cast<int>(dir));
	// PPM kernels loop over cells and fill left(i+1) and right(i), so include one extra cell in the reconstruction direction.
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

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	// loop over faces pointing in the w0-direction
	for (int w0 = 0; w0 < 3; ++w0) {
		// you have two edges on the perimeter of this face
		const int w1 = (w0 + 1) % 3; // vec_fc(w0) + vec_fc(w1)
		const int w2 = (w0 + 2) % 3; // vec_fc(w0) + vec_fc(w2)

		// direction to find the edges either side of the face. this depends on the direction the face points
		std::array<int, 3> delta_w1 = {0, 0, 0};
		std::array<int, 3> delta_w2 = {0, 0, 0};
		if (w0 == 0) {
			delta_w1[1] = 1;
			delta_w2[2] = 1;
		} else if (w0 == 1) {
			delta_w1[2] = 1;
			delta_w2[0] = 1;
		} else if (w0 == 2) {
			delta_w1[0] = 1;
			delta_w2[1] = 1;
		}

		auto const dx1 = dx[w1];
		auto const dx2 = dx[w2];
		auto const ec_emf_w1 = ec_emf_mf[w1].const_arrays();
		auto const ec_emf_w2 = ec_emf_mf[w2].const_arrays();
		auto const fc_consVarOld = fc_consVarOld_mf[w0].const_arrays();
		auto fc_consVarNew = fc_consVarNew_mf[w0].arrays();

		amrex::ParallelFor(fc_consVarNew_mf[w0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// the ec emfs sit in the opposite fc directions relative to the face
			const double emf_w1_m = ec_emf_w1[bx](i, j, k);
			const double emf_w2_m = ec_emf_w2[bx](i, j, k);
			const double emf_w1_p = ec_emf_w1[bx](i + delta_w2[0], j + delta_w2[1], k + delta_w2[2]);
			const double emf_w2_p = ec_emf_w2[bx](i + delta_w1[0], j + delta_w1[1], k + delta_w1[2]);
			const double db_dt = (dx1 * (emf_w1_m - emf_w1_p) + dx2 * (emf_w2_p - emf_w2_m)) / (dx1 * dx2);

			fc_consVarNew[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) =
			    fc_consVarOld[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
		});
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto MHDSystem<problem_t>::computeResistiveEMF(amrex::Array4<const amrex::Real> const &a4_b_w0,
										   amrex::Array4<const amrex::Real> const &a4_b_w1, int i, int j, int k,
										   std::array<int, 3> const &delta_w0, std::array<int, 3> const &delta_w1,
										   amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity) -> amrex::Real
{
	const amrex::Real J_edge = (a4_b_w1(i, j, k) - a4_b_w1(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2])) / dx_w0 -
				   (a4_b_w0(i, j, k) - a4_b_w0(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2])) / dx_w1;
	return resistivity * J_edge;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
MHDSystem<problem_t>::ApplyResistiveCorrection(amrex::Array4<amrex::Real> const &a4_emf2_ave, int i, int j, int k, amrex::Array4<const amrex::Real> const &a4_b_w0,
					       amrex::Array4<const amrex::Real> const &a4_b_w1, std::array<int, 3> const &delta_w0,
					       std::array<int, 3> const &delta_w1, amrex::Real dx_w0, amrex::Real dx_w1, amrex::Real resistivity)
{
	if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
		a4_emf2_ave(i, j, k) -= computeResistiveEMF(a4_b_w0, a4_b_w1, i, j, k, delta_w0, delta_w1, dx_w0, dx_w1, resistivity);
	} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
		const amrex::Real eta = computeResistivity<problem_t>(i, j, k, a4_b_w0, a4_b_w1, dx_w0, dx_w1);
		a4_emf2_ave(i, j, k) -= computeResistiveEMF(a4_b_w0, a4_b_w1, i, j, k, delta_w0, delta_w1, dx_w0, dx_w1, eta);
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

	for (int iface = 0; iface < AMREX_SPACEDIM; ++iface) {
		const int a = (iface + 1) % 3;
		const int b = (iface + 2) % 3;

		std::array<int, 3> delta_iface = {0, 0, 0};
		std::array<int, 3> delta_a = {0, 0, 0};
		std::array<int, 3> delta_b = {0, 0, 0};
		delta_iface[iface] = 1;
		delta_a[a] = 1;
		delta_b[b] = 1;

		const amrex::Real dx_iface = dx[iface];
		const amrex::Real dx_a = dx[a];
		const amrex::Real dx_b = dx[b];
		const int energy_idx = HydroSystem<problem_t>::energy_index;

		for (amrex::MFIter mfi(fluxArrays[iface]); mfi.isValid(); ++mfi) {
			const amrex::Box &box_face = mfi.validbox();

			// B_a on a-faces, B_b on b-faces, B_iface on iface-faces (aliased, no copy)
			const auto B_a = fcx_mf_cVars[a][mfi].const_array(bfield_index);
			const auto B_b = fcx_mf_cVars[b][mfi].const_array(bfield_index);
			const auto B_iface = fcx_mf_cVars[iface][mfi].const_array(bfield_index);
			auto flux = fluxArrays[iface][mfi].array();

			// a-edge: a4_b_w0=B_b, delta_w0=delta_b, dx_w0=dx_b; a4_b_w1=B_iface, delta_w1=delta_iface, dx_w1=dx_iface
			// b-edge: a4_b_w0=B_iface, delta_w0=delta_iface, dx_w0=dx_iface; a4_b_w1=B_a, delta_w1=delta_a, dx_w1=dx_a
			amrex::ParallelFor(box_face, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real eta_J_a0 = 0.0;
				amrex::Real eta_J_a1 = 0.0;
				amrex::Real eta_J_b0 = 0.0;
				amrex::Real eta_J_b1 = 0.0;
				const amrex::Real eta = resistivity; // first-capture for CUDA
				// first-capture for CUDA
				const auto B_b_local = B_b;
				const auto B_iface_local = B_iface;
				const auto B_a_local = B_a;
				const auto delta_b_local = delta_b;
				const auto delta_iface_local = delta_iface;
				const auto delta_a_local = delta_a;
				const amrex::Real dx_b_local = dx_b;
				const amrex::Real dx_iface_local = dx_iface;
				const amrex::Real dx_a_local = dx_a;
				if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::constant) {
					eta_J_a0 = computeResistiveEMF(B_b_local, B_iface_local, i, j, k, delta_b_local, delta_iface_local, dx_b_local,
								       dx_iface_local, eta);
					eta_J_a1 = computeResistiveEMF(B_b_local, B_iface_local, i + delta_b_local[0], j + delta_b_local[1],
								       k + delta_b_local[2], delta_b_local, delta_iface_local, dx_b_local, dx_iface_local, eta);
					eta_J_b0 = computeResistiveEMF(B_iface_local, B_a_local, i, j, k, delta_iface_local, delta_a_local, dx_iface_local,
								       dx_a_local, eta);
					eta_J_b1 = computeResistiveEMF(B_iface_local, B_a_local, i + delta_a_local[0], j + delta_a_local[1],
								       k + delta_a_local[2], delta_iface_local, delta_a_local, dx_iface_local, dx_a_local, eta);
				} else if constexpr (Physics_Traits<problem_t>::resistivity_model == ResistivityModel::problem_defined) {
					const amrex::Real eta_a0 = computeResistivity<problem_t>(i, j, k, B_b_local, B_iface_local, dx_b_local, dx_iface_local);
					eta_J_a0 = computeResistiveEMF(B_b_local, B_iface_local, i, j, k, delta_b_local, delta_iface_local, dx_b_local,
								       dx_iface_local, eta_a0);
					const amrex::Real eta_a1 =
					    computeResistivity<problem_t>(i + delta_b_local[0], j + delta_b_local[1], k + delta_b_local[2], B_b_local,
									  B_iface_local, dx_b_local, dx_iface_local);
					eta_J_a1 =
					    computeResistiveEMF(B_b_local, B_iface_local, i + delta_b_local[0], j + delta_b_local[1], k + delta_b_local[2],
								delta_b_local, delta_iface_local, dx_b_local, dx_iface_local, eta_a1);
					const amrex::Real eta_b0 = computeResistivity<problem_t>(i, j, k, B_iface_local, B_a_local, dx_iface_local, dx_a_local);
					eta_J_b0 = computeResistiveEMF(B_iface_local, B_a_local, i, j, k, delta_iface_local, delta_a_local, dx_iface_local,
								       dx_a_local, eta_b0);
					const amrex::Real eta_b1 =
					    computeResistivity<problem_t>(i + delta_a_local[0], j + delta_a_local[1], k + delta_a_local[2], B_iface_local,
									  B_a_local, dx_iface_local, dx_a_local);
					eta_J_b1 =
					    computeResistiveEMF(B_iface_local, B_a_local, i + delta_a_local[0], j + delta_a_local[1], k + delta_a_local[2],
								delta_iface_local, delta_a_local, dx_iface_local, dx_a_local, eta_b1);
				}

				// Average face-B to each edge position across the face-normal direction
				const amrex::Real avg_Bb_0 =
				    0.5 * (B_b_local(i, j, k) + B_b_local(i - delta_iface_local[0], j - delta_iface_local[1], k - delta_iface_local[2]));
				const amrex::Real avg_Bb_1 =
				    0.5 * (B_b_local(i + delta_b_local[0], j + delta_b_local[1], k + delta_b_local[2]) +
					   B_b_local(i + delta_b_local[0] - delta_iface_local[0], j + delta_b_local[1] - delta_iface_local[1],
						     k + delta_b_local[2] - delta_iface_local[2]));
				const amrex::Real avg_Ba_0 =
				    0.5 * (B_a_local(i, j, k) + B_a_local(i - delta_iface_local[0], j - delta_iface_local[1], k - delta_iface_local[2]));
				const amrex::Real avg_Ba_1 =
				    0.5 * (B_a_local(i + delta_a_local[0], j + delta_a_local[1], k + delta_a_local[2]) +
					   B_a_local(i + delta_a_local[0] - delta_iface_local[0], j + delta_a_local[1] - delta_iface_local[1],
						     k + delta_a_local[2] - delta_iface_local[2]));

				// F_eta = (eta_J x B)_iface = eta_J_a * B_b - eta_J_b * B_a, averaged over the two bounding edges
				const amrex::Real F_eta = 0.25 * (eta_J_a0 * avg_Bb_0 + eta_J_a1 * avg_Bb_1 - eta_J_b0 * avg_Ba_0 - eta_J_b1 * avg_Ba_1);
				flux(i, j, k, energy_idx) += F_eta;
			});
		}
	}
}

#endif // HYDRO_SYSTEM_HPP_
