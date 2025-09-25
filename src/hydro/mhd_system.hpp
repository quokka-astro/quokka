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

AMREX_ENUM(EMFAvgType, BalsaraSpicer, LD04, Balsara2025_HLL); // NOLINT

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	enum varIndex_perDim {
		bfield_index = Physics_Indices<problem_t>::mhdFirstIndex,
	};

	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type,
			       int emf_scheme);

	static void ComputeEMF_FS(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
				  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				  std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type);

	static void ComputeEMF_FS_FCVel(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type);

	static void ComputeEMF_Balsara(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
				       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type);

	static void EMFSolver_BalsaraSpicer(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec);

	static void EMFSolver_LD04(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec,
				   std::array<int, 2> extrap_dirs, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fspds,
				   std::array<std::array<amrex::FArrayBox, 2>, 2> &ec_fabs_Bi_ieside);

	static void EMFSolver_Balsara2025_HLL(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec,
					      std::array<int, 2> extrap_dirs, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fspds,
					      std::array<std::array<amrex::FArrayBox, 2>, 2> &ec_fabs_Bi_ieside);

	static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid, int reconstructionOrder);

	static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);
};

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type,
				      int emf_scheme)
{
	if (emf_scheme == 0) {
		MHDSystem<problem_t>::ComputeEMF_FS(ec_mf_emf_components, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, emf_avg_type);
	} else if (emf_scheme == 1) {
		MHDSystem<problem_t>::ComputeEMF_FS_FCVel(ec_mf_emf_components, fcx_mf_vel, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, emf_avg_type);
	} else if (emf_scheme == 2) {
		MHDSystem<problem_t>::ComputeEMF_Balsara(ec_mf_emf_components, cc_mf_cVars, fcx_mf_cVars, fcx_mf_fspds, reconstructionOrder, emf_avg_type);
	} else {
		throw std::runtime_error("Unsupported EMF-scheme: " + std::to_string(emf_scheme) + ". Expected either 0 (FS) or 1 (FS_FCVel).");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FS(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FS()");
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
		const amrex::Box &box_cc_U = amrex::grow(box_cc, nghost_cc);
		std::array<amrex::FArrayBox, 3> cc_fabs_Ux = {amrex::FArrayBox(box_cc_U, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_U, 1, amrex::The_Async_Arena()),
							      amrex::FArrayBox(box_cc_U, 1, amrex::The_Async_Arena())};
		{
			const auto &cc_a4_Ux0 = cc_fabs_Ux[0].array();
			const auto &cc_a4_Ux1 = cc_fabs_Ux[1].array();
			const auto &cc_a4_Ux2 = cc_fabs_Ux[2].array();
			const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();

			amrex::ParallelFor(box_cc_U, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
				const auto px1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				const auto px2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				const auto px3 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				cc_a4_Ux0(i, j, k) = px1 / rho;
				cc_a4_Ux1(i, j, k) = px2 / rho;
				cc_a4_Ux2(i, j, k) = px3 / rho;
			});
		}

		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// compute the magnetic flux through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {
			// for each of the two cell-edges on the cell-face
			// we are doing redundant compute. only need to look at one edge for each face: there is a one-to-one mapping.

			// define the two directions we need to extrapolate cell-centered velocity fields to get them to the cell-edge
			// we will want to compute E2 = (U0 * B1 - U1 * B0) along the cell-edge
			std::array<int, 2> extrap_dirs = {(iedge + 1) % 3, (iedge + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(extrap_dirs[0]),
								    amrex::IntVect::TheDimensionVector(extrap_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// initialise FArrayBox for storing the temporary edge-centered velocity fields created in each permutation of reconstructing from the
			// cell-face indexing: field[2: i-side of edge]
			std::array<amrex::FArrayBox, 2> ec_fabs_U_ieside = {amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena()),
									    amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena())};

			// indexing: field[2: i-compnent][2: i-side of edge]
			// note: magnetic field components cannot be discontinuous along themselves (i.e., either side of the face where they are
			// stored), so there are only two possible values (sides), rather than four (quadrants of) possible reconstructed values
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;

			// initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
			// indexing: field[2: i-compnent][4: quadrant around edge]
			std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_Ui_q;

			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_Bi_ieside[icomp][0] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				ec_fabs_Bi_ieside[icomp][1] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				for (int iquad = 0; iquad < 4; ++iquad) {
					ec_fabs_Ui_q[icomp][iquad] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
					ec_fabs_Ui_q[icomp][iquad].setVal<amrex::RunOn::Device>(0.0);
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
				// we expand the domain of cc-data, so that when we reconstruct cc->fc (we include enough ghost cells in the fc->ec dimension),
				// we get as an output (from reconstructing fc->ec) data only in the valid domain
				const amrex::Box box_cc_U =
				    amrex::grow(box_cc, (nghost_cc - 1) * vec_fc2ec); // note, the reconstruct function will uniformly grow the bounds by 1
				const amrex::Box box_fc_U = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec + 1);

				// extrapolate both required cell-centered velocity fields to the cell-edge
				for (int icomp = 0; icomp < 2; ++icomp) {
					// create temporary FArrayBox for storing the face-centered velocity field reconstructed from the cell-center
					// indexing: field[2: i-side of face]
					const int wcomp = extrap_dirs[icomp];
					std::array<amrex::FArrayBox, 2> fc_fabs_U_ifside = {amrex::FArrayBox(box_fc_U, 1, amrex::The_Async_Arena()),
											    amrex::FArrayBox(box_fc_U, 1, amrex::The_Async_Arena())};

					// extrapolate cell-centered velocity components to the cell-face
					MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_Ux[wcomp].array(), fc_fabs_U_ifside[0].array(),
									    fc_fabs_U_ifside[1].array(), box_cc_U, reconstructionOrder);

					// extrapolate face-centered velocity components to the cell-edge
					for (int iface = 0; iface < 2; ++iface) {
						// reset values in temporary FArrayBox
						ec_fabs_U_ieside[0].setVal<amrex::RunOn::Device>(0.0);
						ec_fabs_U_ieside[1].setVal<amrex::RunOn::Device>(0.0);

						// extrapolate face-centered velocity component to the cell-edge
						MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_U_ifside[iface].array(), ec_fabs_U_ieside[0].array(),
										    ec_fabs_U_ieside[1].array(), box_fc, reconstructionOrder);

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

						ec_fabs_Ui_q[icomp][iquad0].plus<amrex::RunOn::Device>(ec_fabs_U_ieside[0], 0, 0, 1);
						ec_fabs_Ui_q[icomp][iquad1].plus<amrex::RunOn::Device>(ec_fabs_U_ieside[1], 0, 0, 1);
					}
				}
			}

			// finish averaging the two different ways for extrapolating velocity fields: cc->fc->ec
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int iquad = 0; iquad < 4; ++iquad) {
					ec_fabs_Ui_q[icomp][iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
				}
			}

			// extrapolate the two required face-centered magnetic field components to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const int extrap_dir2edge = extrap_dirs[(icomp + 1) % 2];
				const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
				const int wcomp = extrap_dirs[icomp];
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(wcomp);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// extrapolate face-centered magnetic components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Bx[wcomp].array(), ec_fabs_Bi_ieside[icomp][0].array(),
								    ec_fabs_Bi_ieside[icomp][1].array(), box_fc, reconstructionOrder);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_E_q;

			// compute the EMF along the cell-edge
			for (int iquad = 0; iquad < 4; ++iquad) {
				// extract relevant velocity and magnetic field components
				const auto &U0_qi = ec_fabs_Ui_q[0][iquad].const_array();
				const auto &U1_qi = ec_fabs_Ui_q[1][iquad].const_array();
				const auto &B0_qi = ec_fabs_Bi_ieside[0][(iquad == 0 || iquad == 3) ? 0 : 1].const_array();
				const auto &B1_qi = ec_fabs_Bi_ieside[1][(iquad < 2) ? 0 : 1].const_array();

				// compute electric field in the quadrant about the cell-edge: cross product between velocity and magnetic field in that
				// define EMF FArrayBox
				ec_fabs_E_q[iquad] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());
				const auto &E2_qi = ec_fabs_E_q[iquad].array();

				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double u0 = U0_qi(i, j, k);
					const double u1 = U1_qi(i, j, k);
					const double b0 = B0_qi(i, j, k);
					const double b1 = B1_qi(i, j, k);
					const double uxb = u0 * b1 - u1 * b0;
					E2_qi(i, j, k) = uxb;
				});
			}

			// extract wavespeeds
			int const w0_index = extrap_dirs[0];
			int const w1_index = extrap_dirs[1];
			std::array<int, 3> delta_w0 = {0, 0, 0};
			std::array<int, 3> delta_w1 = {0, 0, 0};
			delta_w0[w0_index] = 1;
			delta_w1[w1_index] = 1;
			const auto &fspd_x0 = fcx_mf_fspds[w0_index][mfi].const_array();
			const auto &fspd_x1 = fcx_mf_fspds[w1_index][mfi].const_array();

			// extract both components of magnetic field either side of the cell-edge
			const auto &B0_m = ec_fabs_Bi_ieside[0][0].const_array();
			const auto &B0_p = ec_fabs_Bi_ieside[0][1].const_array();
			const auto &B1_m = ec_fabs_Bi_ieside[1][0].const_array();
			const auto &B1_p = ec_fabs_Bi_ieside[1][1].const_array();

			// extract all four quadrants of the electric field about the cell-edge
			const auto &E2_q0 = ec_fabs_E_q[0].const_array();
			const auto &E2_q1 = ec_fabs_E_q[1].const_array();
			const auto &E2_q2 = ec_fabs_E_q[2].const_array();
			const auto &E2_q3 = ec_fabs_E_q[3].const_array();

			// compute electric field on the cell-edge
			const auto &E2_ave = ec_mf_emf_components[iedge][mfi].array();

			if (emf_avg_type == EMFAvgType::BalsaraSpicer) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double E2_q0_ = E2_q0(i, j, k);
					const double E2_q1_ = E2_q1(i, j, k);
					const double E2_q2_ = E2_q2(i, j, k);
					const double E2_q3_ = E2_q3(i, j, k);
					// Balsara & Spicer averaging scheme:
					E2_ave(i, j, k) = 0.25 * (E2_q0_ + E2_q1_ + E2_q2_ + E2_q3_);
				});
			} else if (emf_avg_type == EMFAvgType::LD04) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double E2_q0_ = E2_q0(i, j, k);
					const double E2_q1_ = E2_q1(i, j, k);
					const double E2_q2_ = E2_q2(i, j, k);
					const double E2_q3_ = E2_q3(i, j, k);

					const double B0_p_ = B0_p(i, j, k);
					const double B0_m_ = B0_m(i, j, k);
					const double B1_p_ = B1_p(i, j, k);
					const double B1_m_ = B1_m(i, j, k);

					// LD04 scheme:
					const double a0_m = std::max(fspd_x0(i, j, k, 0), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 0));
					const double a0_p = std::max(fspd_x0(i, j, k, 1), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 1));
					const double a1_m = std::max(fspd_x1(i, j, k, 0), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 0));
					const double a1_p = std::max(fspd_x1(i, j, k, 1), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 1));

					// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
					// (-,+) | (+,+)
					//   1   |   2
					// ------+------
					//   0   |   3
					// (-,-) | (+,-)

					const double num1 =
					    ((a0_p * a1_p) * E2_q0_ + (a0_m * a1_p) * E2_q3_) + ((a0_p * a1_m) * E2_q1_ + (a0_m * a1_m) * E2_q2_);
					const double num2 =
					    ((a0_p * a1_p) * E2_q0_ + (a0_p * a1_m) * E2_q1_) + ((a0_m * a1_p) * E2_q3_ + (a0_m * a1_m) * E2_q2_);

					// must be averaged for exact floating-point symmetry
					const double numerator = 0.5 * (num1 + num2);
					const double denominator = (a0_m + a0_p) * (a1_m + a1_p);

					// NOTE: there is a sign error in Equation 56 of Felker & Stone
					const double term2 =
					    ((a1_m * a1_p) / (a1_m + a1_p)) * (B0_p_ - B0_m_) - ((a0_m * a0_p) / (a0_m + a0_p)) * (B1_p_ - B1_m_);

					E2_ave(i, j, k) = (numerator / denominator) + term2;
				});
			}
		}
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_FS_FCVel(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder,
					       EMFAvgType emf_avg_type)
{
	const BL_PROFILE("MHDSystem::ComputeEMF_FS_FCVel()");

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
		std::array<amrex::FArrayBox, 3> fc_fabs_Ux = {
		    amrex::FArrayBox(fcx_mf_vel[0][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[1][mfi], amrex::make_alias, 0, 1),
		    amrex::FArrayBox(fcx_mf_vel[2][mfi], amrex::make_alias, 0, 1),
		};
		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};
		// compute the emf components on the cell-edge to inform how much magnetic flux travels through each cell-face
		for (int iedge = 0; iedge < 3; ++iedge) {

			// define the two face-centered velocity/magnetic field components we need at the cell-edge
			// we will want to compute E2 = (U0 * B1 - U1 * B0) along the cell-edge
			std::array<int, 2> field_w_indices = {(iedge + 1) % 3, (iedge + 2) % 3};
			const amrex::Box box_ec = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(field_w_indices[0]) +
									     amrex::IntVect::TheDimensionVector(field_w_indices[1]));
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

			// FArrayBoxes for storing the edge-centered fields produced by reconstructing from the cell-face to the cell-edge
			// indexing: field[2: i-component][2: i-side of edge]
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Ui_ieside;
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;
			// define quantities - allocate with async arena
			for (int icomp = 0; icomp < 2; ++icomp) {
				for (int ieside = 0; ieside < 2; ++ieside) {
					ec_fabs_Ui_ieside[icomp][ieside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
					ec_fabs_Bi_ieside[icomp][ieside] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
			}

			// extrapolate the face-centered fields (normal to the cell-face) to the cell-edge
			for (int icomp = 0; icomp < 2; ++icomp) {
				const auto dir2edge = static_cast<FluxDir>(field_w_indices[(icomp + 1) % 2]);
				const int wcomp = field_w_indices[icomp];
				const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(wcomp);
				const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
				// extrapolate face-centered components to the cell-edge
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Bx[wcomp].array(), ec_fabs_Bi_ieside[icomp][0].array(),
								    ec_fabs_Bi_ieside[icomp][1].array(), box_fc, reconstructionOrder);
				MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Ux[wcomp].array(), ec_fabs_Ui_ieside[icomp][0].array(),
								    ec_fabs_Ui_ieside[icomp][1].array(), box_fc, reconstructionOrder);
			}

			// indexing: field[4: quadrant around edge]
			std::array<amrex::FArrayBox, 4> ec_fabs_E_Q;
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
				std::array<amrex::Array4<const double>, 4> U0s;
				std::array<amrex::Array4<const double>, 4> U1s;
				std::array<amrex::Array4<const double>, 4> B0s;
				std::array<amrex::Array4<const double>, 4> B1s;
				std::array<amrex::Array4<double>, 4> E2s;

				for (int qi = 0; qi < 4; ++qi) {
					const int idx0 = (qi == 0 || qi == 3) ? 0 : 1; // B/T selector for dir-0
					const int idx1 = (qi < 2) ? 0 : 1;	       // L/R selector for dir-1

					// define EMF FArrayBox for each quadrant (we need to allocate outside the kernel)
					ec_fabs_E_Q[qi] = amrex::FArrayBox(box_ec, 1, amrex::The_Async_Arena());

					// extract relevant velocity and magnetic field components (host: get Array4 views)
					U0s[qi] = ec_fabs_Ui_ieside[0][idx0].const_array(); // B/T
					B0s[qi] = ec_fabs_Bi_ieside[0][idx0].const_array(); // B/T
					U1s[qi] = ec_fabs_Ui_ieside[1][idx1].const_array(); // L/R
					B1s[qi] = ec_fabs_Bi_ieside[1][idx1].const_array(); // L/R
					E2s[qi] = ec_fabs_E_Q[qi].array();		    // output EMF view
				}

				// single kernel over the edge-centered box; compute E in all four quadrants
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					for (int qi = 0; qi < 4; ++qi) {
						const double u0 = U0s[qi](i, j, k);
						const double u1 = U1s[qi](i, j, k);
						const double b0 = B0s[qi](i, j, k);
						const double b1 = B1s[qi](i, j, k);
						E2s[qi](i, j, k) = u0 * b1 - u1 * b0; // cross product at the edge
					}
				});
			}

			// extract wavespeeds
			int const w0_index = field_w_indices[0];
			int const w1_index = field_w_indices[1];
			std::array<int, 3> delta_w0 = {0, 0, 0};
			std::array<int, 3> delta_w1 = {0, 0, 0};
			delta_w0[w0_index] = 1;
			delta_w1[w1_index] = 1;
			const auto &fspd_x0 = fcx_mf_fspds[w0_index][mfi].const_array();
			const auto &fspd_x1 = fcx_mf_fspds[w1_index][mfi].const_array();
			// extract both components of magnetic field either side of the cell-edge
			const auto &B0_m = ec_fabs_Bi_ieside[0][0].const_array();
			const auto &B0_p = ec_fabs_Bi_ieside[0][1].const_array();
			const auto &B1_m = ec_fabs_Bi_ieside[1][0].const_array();
			const auto &B1_p = ec_fabs_Bi_ieside[1][1].const_array();
			// extract all four quadrants of the electric field about the cell-edge
			const auto &E2_q0 = ec_fabs_E_Q[0].const_array();
			const auto &E2_q1 = ec_fabs_E_Q[1].const_array();
			const auto &E2_q2 = ec_fabs_E_Q[2].const_array();
			const auto &E2_q3 = ec_fabs_E_Q[3].const_array();
			// compute electric field on the cell-edge
			const auto &E2_ave = ec_mf_emf_components[iedge][mfi].array();
			// only operate on the real cells
			if (emf_avg_type == EMFAvgType::BalsaraSpicer) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double E2_q0_ = E2_q0(i, j, k);
					const double E2_q1_ = E2_q1(i, j, k);
					const double E2_q2_ = E2_q2(i, j, k);
					const double E2_q3_ = E2_q3(i, j, k);
					// Balsara & Spicer averaging scheme:
					E2_ave(i, j, k) = 0.25 * (E2_q0_ + E2_q1_ + E2_q2_ + E2_q3_);
				});
			} else if (emf_avg_type == EMFAvgType::LD04) {
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double E2_q0_ = E2_q0(i, j, k);
					const double E2_q1_ = E2_q1(i, j, k);
					const double E2_q2_ = E2_q2(i, j, k);
					const double E2_q3_ = E2_q3(i, j, k);

					const double B0_p_ = B0_p(i, j, k);
					const double B0_m_ = B0_m(i, j, k);
					const double B1_p_ = B1_p(i, j, k);
					const double B1_m_ = B1_m(i, j, k);

					// LD04 scheme:
					const double a0_m = std::max(fspd_x0(i, j, k, 0), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 0));
					const double a0_p = std::max(fspd_x0(i, j, k, 1), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 1));
					const double a1_m = std::max(fspd_x1(i, j, k, 0), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 0));
					const double a1_p = std::max(fspd_x1(i, j, k, 1), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 1));

					// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
					// (-,+) | (+,+)
					//   1   |   2
					// ------+------
					//   0   |   3
					// (-,-) | (+,-)

					const double num1 =
					    ((a0_p * a1_p) * E2_q0_ + (a0_m * a1_p) * E2_q3_) + ((a0_p * a1_m) * E2_q1_ + (a0_m * a1_m) * E2_q2_);
					const double num2 =
					    ((a0_p * a1_p) * E2_q0_ + (a0_p * a1_m) * E2_q1_) + ((a0_m * a1_p) * E2_q3_ + (a0_m * a1_m) * E2_q2_);

					// must be averaged for exact floating-point symmetry
					const double numerator = 0.5 * (num1 + num2);
					const double denominator = (a0_m + a0_p) * (a1_m + a1_p);

					// NOTE: there is a sign error in Equation 56 of Felker & Stone
					const double term2 =
					    ((a1_m * a1_p) / (a1_m + a1_p)) * (B0_p_ - B0_m_) - ((a0_m * a0_p) / (a0_m + a0_p)) * (B1_p_ - B1_m_);

					E2_ave(i, j, k) = (numerator / denominator) + term2;
				});
			}
		}
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF_Balsara(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars,
					      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
					      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgType emf_avg_type)
{
	// calculating v x B at cell center, v already at cell center, B at face center

	const BL_PROFILE("MHDSystem::ComputeEMF_Balsara()");
	const int nghost_cc = 4; // only 3 ghost zones needed here
	// note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
	// note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells

	// In this function we distinguish between world (w:3), array (i:2), quandrant (q:4), and component (x:3) index-ing by using prefixes. We will
	// use the prefix x- when the w- and i- indexes are the same. We also choose to minimise the storage footprint by only computing and holding
	// onto the quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but allows us to
	// significantly reduces the total memory used, which is a much bigger bottleneck.

	// loop over each box-array on this level
	// constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions

	const auto &ba = cc_mf_cVars.boxArray();
	const auto &dm = cc_mf_cVars.DistributionMap();
	constexpr int nstreams = 1; // only run on 1 GPU stream to avoid race conditions
	amrex::MultiFab cc_mf_EMF(ba, dm, 3, nghost_cc);
	cc_mf_EMF.setVal(0.0, 0, 3, nghost_cc); // initialize to zero everywhere including ghost zones

	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();
		const amrex::Box &box_cc_EMF = mfi.growntilebox(nghost_cc); // Ensure enough ghost cells for EMF computation

		// Access the MultiFab components for this tile
		const auto &cc_a4_EMFx0 = cc_mf_EMF[mfi].array(0);
		const auto &cc_a4_EMFx1 = cc_mf_EMF[mfi].array(1);
		const auto &cc_a4_EMFx2 = cc_mf_EMF[mfi].array(2);

		const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
		std::array<amrex::Array4<amrex::Real>, 3> const cc_a4_EMF_array = {cc_a4_EMFx0, cc_a4_EMFx1, cc_a4_EMFx2};
		std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// Get face-centered B field arrays
		std::array<amrex::Array4<amrex::Real const>, 3> fc_a4_Bx = {fcx_mf_cVars[0][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[1][mfi].const_array(MHDSystem<problem_t>::bfield_index),
									    fcx_mf_cVars[2][mfi].const_array(MHDSystem<problem_t>::bfield_index)};

		// Compute v x B for all three directions in parallel
		amrex::ParallelFor(box_cc_EMF, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
				    0.5 * (fc_a4_Bx[x2ind](i, j, k) + fc_a4_Bx[x2ind](i + delta_x2[0], j + delta_x2[1], k + delta_x2[2]));
				amrex::Real const bx3_avg =
				    0.5 * (fc_a4_Bx[x3ind](i, j, k) + fc_a4_Bx[x3ind](i + delta_x3[0], j + delta_x3[1], k + delta_x3[2]));

				// v x B computation
				cc_a4_EMF_array[idim](i, j, k) = v[x2ind] * bx3_avg - v[x3ind] * bx2_avg; //
			}
		});
	}

	cc_mf_EMF.FillBoundary(); // fill ghost cells
	amrex::Gpu::streamSynchronize();

	// now that EMF is calculated at cell center, we need to interpolate to cell edge
	// we also need to get the magnetic field from the face to cell edge to use the Balsara method
	// otherwise, if other methods are desired, LD04 or BalsaraSpicer just need the reconstructed emf

	// focus first on taking EMF to cell edge:
	for (amrex::MFIter mfi(cc_mf_cVars, amrex::MFItInfo().SetNumStreams(nstreams)); mfi.isValid(); ++mfi) { // keep
		const amrex::Box &box_cc = mfi.validbox();

		std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		for (int iedge = 0; iedge < 3; ++iedge) {
			std::cout << "iedge = " << iedge << std::endl;

			std::array<int, 2> extrap_dirs = {(iedge + 1) % 3, (iedge + 2) % 3};
			std::array<amrex::IntVect, 2> vecs_cc2ec = {amrex::IntVect::TheDimensionVector(extrap_dirs[0]),
								    amrex::IntVect::TheDimensionVector(extrap_dirs[1])};
			const amrex::IntVect vec_cc2ec = vecs_cc2ec[0] + vecs_cc2ec[1];
			const amrex::Box box_ec = amrex::convert(box_cc, vec_cc2ec);
			const amrex::Box box_ec_r = amrex::grow(box_ec, 1);
			// indexing: field[4: quadrant around edge]

			// initializing array to hold EMF at cell edge
			const auto &E2_array = ec_mf_emf_components[iedge][mfi].array();
			std::array<amrex::FArrayBox, 2> ec_fabs_EMF_ieside = {amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena()),
									      amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena())};

			ec_fabs_EMF_ieside[0].setVal<amrex::RunOn::Device>(0.0);
			ec_fabs_EMF_ieside[1].setVal<amrex::RunOn::Device>(0.0);
			std::array<amrex::FArrayBox, 4> ec_fabs_EMF_q;

			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_EMF_q[iquad] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				ec_fabs_EMF_q[iquad].setVal<amrex::RunOn::Device>(0.0);
			}

			// interpolate the cell-centered EMF to the cell-edge
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
				// we expand the domain of cc-data, so that when we reconstruct cc->fc (we include enough ghost cells in the fc->ec
				// dimension), we get as an output (from reconstructing fc->ec) data only in the valid domain
				const amrex::Box box_cc_EMF_edge =
				    amrex::grow(box_cc, (nghost_cc - 1) * vec_fc2ec); // note, the reconstruct function will uniformly grow the bounds by 1
				const amrex::Box box_fc_EMF = amrex::grow(box_fc, (nghost_cc - 1) * vec_fc2ec + 1);

				// extrapolate both required cell-centered EMF to the cell-edge

				// create temporary FArrayBox for storing the face-centered EMF reconstructed from the cell-center
				// indexing: field[2: i-side of face]
				std::array<amrex::FArrayBox, 2> fc_fabs_EMF_ifside = {amrex::FArrayBox(box_fc_EMF, 1, amrex::The_Async_Arena()),
										      amrex::FArrayBox(box_fc_EMF, 1, amrex::The_Async_Arena())};
				// reset values in temporary FArrayBox
				fc_fabs_EMF_ifside[0].setVal<amrex::RunOn::Device>(0.0);
				fc_fabs_EMF_ifside[1].setVal<amrex::RunOn::Device>(0.0);

				// extrapolate cell-centered velocity components to the cell-face
				MHDSystem<problem_t>::ReconstructTo(dir2face, cc_mf_EMF[mfi].array(iedge), fc_fabs_EMF_ifside[0].array(),
								    fc_fabs_EMF_ifside[1].array(), box_cc_EMF_edge, reconstructionOrder);

				// extrapolate face-centered velocity components to the cell-edge
				for (int iface = 0; iface < 2; ++iface) {
					// reset values in temporary FArrayBox
					ec_fabs_EMF_ieside[0].setVal<amrex::RunOn::Device>(0.0);
					ec_fabs_EMF_ieside[1].setVal<amrex::RunOn::Device>(0.0);

					// extrapolate face-centered velocity component to the cell-edge
					MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_EMF_ifside[iface].array(), ec_fabs_EMF_ieside[0].array(),
									    ec_fabs_EMF_ieside[1].array(), box_fc, reconstructionOrder);

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

					ec_fabs_EMF_q[iquad0].plus<amrex::RunOn::Device>(ec_fabs_EMF_ieside[0], 0, 0, 1);
					ec_fabs_EMF_q[iquad1].plus<amrex::RunOn::Device>(ec_fabs_EMF_ieside[1], 0, 0, 1);
				}
			}
			// finish averaging the two different ways for extrapolating emf: cc->fc->ec

			for (int iquad = 0; iquad < 4; ++iquad) {
				ec_fabs_EMF_q[iquad].mult<amrex::RunOn::Device>(0.5, 0, 1);
			}

			if (emf_avg_type == EMFAvgType::BalsaraSpicer) {
				MHDSystem<problem_t>::EMFSolver_BalsaraSpicer(E2_array, ec_fabs_EMF_q, box_ec);
			} else {
				// get fspds to pass to LD04 or Balsara2025_HLL solver if needed
				std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fspds = {
				    fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi), fcx_mf_fspds[2].const_array(mfi)};
				std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;
				for (int icomp = 0; icomp < 2; ++icomp) {
					ec_fabs_Bi_ieside[icomp][0] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
					ec_fabs_Bi_ieside[icomp][1] = amrex::FArrayBox(box_ec_r, 1, amrex::The_Async_Arena());
				}
				// extrapolate the two required face-centered magnetic field components to the cell-edge
				for (int icomp = 0; icomp < 2; ++icomp) {
					const int extrap_dir2edge = extrap_dirs[(icomp + 1) % 2];
					const auto dir2edge = static_cast<FluxDir>(extrap_dir2edge);
					const int wcomp = extrap_dirs[icomp];
					const amrex::IntVect vec_cc2fc = amrex::IntVect::TheDimensionVector(wcomp);
					const amrex::Box box_fc = amrex::convert(box_cc, vec_cc2fc);
					// extrapolate face-centered magnetic components to the cell-edge
					MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Bx[wcomp].array(), ec_fabs_Bi_ieside[icomp][0].array(),
									    ec_fabs_Bi_ieside[icomp][1].array(), box_fc, reconstructionOrder);
				}

				if (emf_avg_type == EMFAvgType::LD04) {
					MHDSystem<problem_t>::EMFSolver_LD04(E2_array, ec_fabs_EMF_q, box_ec, extrap_dirs, fspds, ec_fabs_Bi_ieside);
				} else if (emf_avg_type == EMFAvgType::Balsara2025_HLL) {
					MHDSystem<problem_t>::EMFSolver_Balsara2025_HLL(E2_array, ec_fabs_EMF_q, box_ec, extrap_dirs, fspds, ec_fabs_Bi_ieside);
				} else {
					amrex::Abort("Unknown EMF averaging type");
				}
			}
		}
	}
}

// simplest emf solver: just average the quadrants
template <typename problem_t>
void MHDSystem<problem_t>::EMFSolver_BalsaraSpicer(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q,
						   amrex::Box const &box_ec)
{
	const BL_PROFILE("MHDSystem::ApplyLDEMFsolver()");

	// Get const array views from each FArrayBox
	const auto &E2_q0 = ec_fabs_EMF_q[0].const_array();
	const auto &E2_q1 = ec_fabs_EMF_q[1].const_array();
	const auto &E2_q2 = ec_fabs_EMF_q[2].const_array();
	const auto &E2_q3 = ec_fabs_EMF_q[3].const_array();

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const double E2_q0_ = E2_q0(i, j, k);
		const double E2_q1_ = E2_q1(i, j, k);
		const double E2_q2_ = E2_q2(i, j, k);
		const double E2_q3_ = E2_q3(i, j, k);
		// Balsara & Spicer averaging scheme:
		E2_ave(i, j, k) = 0.25 * (E2_q0_ + E2_q1_ + E2_q2_ + E2_q3_);
	});
}

template <typename problem_t>
void MHDSystem<problem_t>::EMFSolver_LD04(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec,
					  std::array<int, 2> extrap_dirs, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fspds,
					  std::array<std::array<amrex::FArrayBox, 2>, 2> &ec_fabs_Bi_ieside)
{

	const auto &E2_q0 = ec_fabs_EMF_q[0].const_array();
	const auto &E2_q1 = ec_fabs_EMF_q[1].const_array();
	const auto &E2_q2 = ec_fabs_EMF_q[2].const_array();
	const auto &E2_q3 = ec_fabs_EMF_q[3].const_array();

	const auto &B0_m = ec_fabs_Bi_ieside[0][0].const_array();
	const auto &B0_p = ec_fabs_Bi_ieside[0][1].const_array();
	const auto &B1_m = ec_fabs_Bi_ieside[1][0].const_array();
	const auto &B1_p = ec_fabs_Bi_ieside[1][1].const_array();

	int const w0_comp = extrap_dirs[0];
	int const w1_comp = extrap_dirs[1];
	std::array<int, 3> delta_w0 = {0, 0, 0};
	std::array<int, 3> delta_w1 = {0, 0, 0};

	delta_w0[w0_comp] = 1;
	delta_w1[w1_comp] = 1;

	const auto &fspd_x0 = fspds[w0_comp];
	const auto &fspd_x1 = fspds[w1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// 	// LD04 scheme:
		const double a0_m = std::max(fspd_x0(i, j, k, 0), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 0));
		const double a0_p = std::max(fspd_x0(i, j, k, 1), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 1));
		const double a1_m = std::max(fspd_x1(i, j, k, 0), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 0));
		const double a1_p = std::max(fspd_x1(i, j, k, 1), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 1));

		const double E2_q0_ = E2_q0(i, j, k);
		const double E2_q1_ = E2_q1(i, j, k);
		const double E2_q2_ = E2_q2(i, j, k);
		const double E2_q3_ = E2_q3(i, j, k);

		const double B0_p_ = B0_p(i, j, k);
		const double B0_m_ = B0_m(i, j, k);
		const double B1_p_ = B1_p(i, j, k);
		const double B1_m_ = B1_m(i, j, k);
		// 	// note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
		// 	// (-,+) | (+,+)
		// 	//   1   |   2
		// 	// ------+------
		// 	//   0   |   3
		// 	// (-,-) | (+,-)

		const double num1 = ((a0_p * a1_p) * E2_q0_ + (a0_m * a1_p) * E2_q3_) + ((a0_p * a1_m) * E2_q1_ + (a0_m * a1_m) * E2_q2_);
		const double num2 = ((a0_p * a1_p) * E2_q0_ + (a0_p * a1_m) * E2_q1_) + ((a0_m * a1_p) * E2_q3_ + (a0_m * a1_m) * E2_q2_);

		// 	// must be averaged for exact floating-point symmetry
		const double numerator = 0.5 * (num1 + num2);
		const double denominator = (a0_m + a0_p) * (a1_m + a1_p);

		// 	// NOTE: there is a sign error in Equation 56 of Felker & Stone
		const double term2 = ((a1_m * a1_p) / (a1_m + a1_p)) * (B0_p_ - B0_m_) - ((a0_m * a0_p) / (a0_m + a0_p)) * (B1_p_ - B1_m_);

		E2_ave(i, j, k) = (numerator / denominator) + term2;
	});
}

// to pass into EMF solvers: need emf quadrants, and fspds, and if Balsara2025, need B on either side of edge
// out: final emf on edge

template <typename problem_t>
void MHDSystem<problem_t>::EMFSolver_Balsara2025_HLL(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q,
						     amrex::Box const &box_ec, std::array<int, 2> extrap_dirs,
						     std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fspds,
						     std::array<std::array<amrex::FArrayBox, 2>, 2> &ec_fabs_Bi_ieside)
{
	const BL_PROFILE("MHDSystem::ApplyBalsaraEMFSolver()");
	const auto &E2_q0 = ec_fabs_EMF_q[0].const_array();
	const auto &E2_q1 = ec_fabs_EMF_q[1].const_array();
	const auto &E2_q2 = ec_fabs_EMF_q[2].const_array();
	const auto &E2_q3 = ec_fabs_EMF_q[3].const_array();

	const auto &B0_D = ec_fabs_Bi_ieside[0][0].const_array();
	const auto &B0_U = ec_fabs_Bi_ieside[0][1].const_array();
	const auto &B1_R = ec_fabs_Bi_ieside[1][0].const_array();
	const auto &B1_L = ec_fabs_Bi_ieside[1][1].const_array();

	int const w0_comp = extrap_dirs[0];
	int const w1_comp = extrap_dirs[1];
	std::array<int, 3> delta_w0 = {0, 0, 0};
	std::array<int, 3> delta_w1 = {0, 0, 0};

	delta_w0[w0_comp] = 1;
	delta_w1[w1_comp] = 1;

	const auto &fspd_x0 = fspds[w0_comp];
	const auto &fspd_x1 = fspds[w1_comp];

	amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		// Wave speeds
		const double SL = std::max(fspd_x0(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 0), fspd_x0(i, j, k, 0));
		const double SR = std::max(fspd_x0(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 1), fspd_x0(i, j, k, 1));
		const double SU = std::max(fspd_x1(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 0), fspd_x1(i, j, k, 1));
		const double SD = std::max(fspd_x1(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 1), fspd_x1(i, j, k, 0));

		// EMF quadrants
		const auto E2_LD = E2_q0(i, j, k);
		const auto E2_LU = E2_q1(i, j, k);
		const auto E2_RU = E2_q2(i, j, k);
		const auto E2_RD = E2_q3(i, j, k);

		// Magnetic field components
		const auto B0_U_ = B0_U(i,j,k);
		const auto B0_D_ = B0_D(i,j,k);
		const auto B1_R_ = B1_R(i,j,k);
		const auto B1_L_ = B1_L(i,j,k);

		const auto E2_U_star = (SR * E2_LU - SL * E2_RU) / (SR - SL) - (SR * SL) * (B1_R_ - B1_L_) / (SR - SL);
		const auto E2_D_star = (SR * E2_LD - SL * E2_RD) / (SR - SL) - (SR * SL) * (B1_R_ - B1_L_) / (SR - SL);
		const auto E2_R_star = (SU * E2_RD - SD * E2_RU) / (SU - SD) + (SU * SD) * (B0_U_ - B0_D_) / (SU - SD);
		const auto E2_L_star = (SU * E2_LD - SD * E2_LU) / (SU - SD) + (SU * SD) * (B0_U_ - B0_D_) / (SU - SD);

		const auto B0_dstar = (SU * B0_U_ - SD * B0_D_) / (SU - SD) + (E2_LD - E2_LU + E2_RD - E2_RU) / (2.0 * (SU - SD));
		const auto B1_dstar = (SR * B1_R_ - SL * B1_L_) / (SR - SL) + (-E2_LD - E2_LU + E2_RD + E2_RU) / (2.0 * (SR - SL));

		const auto E2_dstar_1 = -(SR + SL) * B1_dstar / 2.0 + (SU * (E2_LD + E2_RD) - SD * (E2_LU + E2_RU)) / (2.0 * (SU - SD)) -
					SU * SD * (B0_D_ - B0_U_) / (SU - SD) + (SR * B1_R_ + SL * B1_L_) / 2.0;

		const auto E2_dstar_2 = (SU + SD) * B0_dstar / 2.0 + (SR * (E2_RU + E2_LU) - SL * (E2_RD + E2_LD)) / (2.0 * (SR - SL)) -
					(SU * B0_U_ + SD * B0_D_) / 2.0 - SR * SL * (B1_R_ - B1_L_) / (SR - SL);

		const auto E2_dstar = 0.5 * (E2_dstar_1 + E2_dstar_2);

		if (SL >= 0.0 && SD >= 0.0) {
			E2_ave(i, j, k) = E2_LD;
		} else if (SR <= 0.0 && SD >= 0.0) {
			E2_ave(i, j, k) = E2_RD;
		} else if (SR <= 0.0 && SU <= 0.0) {
			E2_ave(i, j, k) = E2_RU;
		} else if (SL >= 0.0 && SU <= 0.0) {
			E2_ave(i, j, k) = E2_LU;
		} else if (SL >= 0.0 && SD < 0.0 && SU > 0.0) {
			E2_ave(i, j, k) = E2_L_star;
		} else if (SR <= 0.0 && SD < 0.0 && SU > 0.0) {
			E2_ave(i, j, k) = E2_R_star;
		} else if (SR <= 0.0 && SU <= 0.0 && SL < 0.0) {
			E2_ave(i, j, k) = E2_U_star;
		} else if (SL >= 0.0 && SU <= 0.0 && SR > 0.0) {
			E2_ave(i, j, k) = E2_D_star;
		} else {
			E2_ave(i, j, k) = E2_dstar;
		}
	});
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid,
					 int reconstructionOrder)
{
	const BL_PROFILE("MHDSystem::ReconstructTo()");
	amrex::Box const &box_r = amrex::grow(box_cValid, 1);
	amrex::Box const &box_r_x1 = amrex::surroundingNodes(box_r, static_cast<int>(dir));
	if (reconstructionOrder == 5) {
		// note: only box_r is used. box_r_x1 is unused.
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X1>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X2>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM_EP<FluxDir::X3>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 3) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 2) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1, SlopeLimiter::minmod>(cState, lState, rState, box_r, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2, SlopeLimiter::minmod>(cState, lState, rState, box_r, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3, SlopeLimiter::minmod>(cState, lState, rState, box_r, 1);
				break;
		}
	} else if (reconstructionOrder == 1) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(cState, lState, rState, box_r_x1, 1);
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

#endif // HYDRO_SYSTEM_HPP_
