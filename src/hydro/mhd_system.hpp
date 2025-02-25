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
#include "AMReX_GpuControl.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	enum varIndex_perDim {
		bfield_index = Physics_Indices<problem_t>::mhdFirstIndex,
	};

	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds,
			       int nghost_fc, int reconstructionOrder, amrex::Geometry geom, double time);

	static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &reconstructRange,
				  int reconstructionOrder);

	static std::array<const double, 3> computeWaveSolution(double x1, int itype, double time);

	static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, const double dt,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo,
				      double time);
};

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, const int nghost_fc, int reconstructionOrder,
				      amrex::Geometry geom, double time)
{
	const int nghost_cc = 4; // we only need 4 cc ghost cells when reconstructing cc->fc->ec using PPM

	// loop over each box-array on the level
	// note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
	// note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells
	for (amrex::MFIter mfi(cc_mf_cVars); mfi.isValid(); ++mfi) {
		const amrex::Box &box_cc = mfi.validbox();

		// In this function we distinguish between world (w:3), array (i:2), quandrant (q:4), and component (x:3) index-ing by using prefixes. We will
		// use the prefix x- when the w- and i- indexes are the same. We also choose to minimise the storage footprint by only computing and holding
		// onto the quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but allows us to
		// significantly reduces the total memory used, which is a much bigger bottleneck.

		// extract cell-centered velocity fields
		// indexing: field[3: x-component]
		std::array<amrex::FArrayBox, 3> cc_fabs_Ux;
		{
			const amrex::Box &box_cc_U = amrex::grow(box_cc, nghost_cc);
			cc_fabs_Ux[0].resize(box_cc_U, 1);
			cc_fabs_Ux[1].resize(box_cc_U, 1);
			cc_fabs_Ux[2].resize(box_cc_U, 1);
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

		std::array<amrex::FArrayBox, 3> cc_fabs_Ex;
		{
			const amrex::Box &box_cc_grown = amrex::grow(box_cc, 1); // reconstruction range
			cc_fabs_Ex[0].resize(box_cc_grown, 1);
			cc_fabs_Ex[1].resize(box_cc_grown, 1);
			cc_fabs_Ex[2].resize(box_cc_grown, 1);
			std::array<amrex::FArrayBox, 3> cc_fabs_Bx;
			cc_fabs_Bx[0].resize(box_cc_grown, 1);
			cc_fabs_Bx[1].resize(box_cc_grown, 1);
			cc_fabs_Bx[2].resize(box_cc_grown, 1);
			const auto &cc_a4_Bx0 = cc_fabs_Bx[0].array();
			const auto &cc_a4_Bx1 = cc_fabs_Bx[1].array();
			const auto &cc_a4_Bx2 = cc_fabs_Bx[2].array();
			const auto &fc_a4_Bx0 = fc_fabs_Bx[0].const_array();
			const auto &fc_a4_Bx1 = fc_fabs_Bx[1].const_array();
			const auto &fc_a4_Bx2 = fc_fabs_Bx[2].const_array();
			amrex::ParallelFor(box_cc_grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto Bx0_m = fc_a4_Bx0(i, j, k);
				const auto Bx1_m = fc_a4_Bx1(i, j, k);
				const auto Bx2_m = fc_a4_Bx2(i, j, k);
				const auto Bx0_p = fc_a4_Bx0(i + 1, j, k);
				const auto Bx1_p = fc_a4_Bx1(i, j + 1, k);
				const auto Bx2_p = fc_a4_Bx2(i, j, k + 1);
				cc_a4_Bx0(i, j, k) = 0.5 * (Bx0_m + Bx0_p);
				cc_a4_Bx1(i, j, k) = 0.5 * (Bx1_m + Bx1_p);
				cc_a4_Bx2(i, j, k) = 0.5 * (Bx2_m + Bx2_p);
			});
			const auto &cc_a4_Ex0 = cc_fabs_Ex[0].array();
			const auto &cc_a4_Ex1 = cc_fabs_Ex[1].array();
			const auto &cc_a4_Ex2 = cc_fabs_Ex[2].array();
			const auto &cc_a4_Ux0 = cc_fabs_Ux[0].const_array();
			const auto &cc_a4_Ux1 = cc_fabs_Ux[1].const_array();
			const auto &cc_a4_Ux2 = cc_fabs_Ux[2].const_array();
			amrex::ParallelFor(box_cc_grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const auto Ux0 = cc_a4_Ux0(i, j, k);
				const auto Ux1 = cc_a4_Ux1(i, j, k);
				const auto Ux2 = cc_a4_Ux2(i, j, k);
				const auto Bx0 = cc_a4_Bx0(i, j, k);
				const auto Bx1 = cc_a4_Bx1(i, j, k);
				const auto Bx2 = cc_a4_Bx2(i, j, k);

				const auto Ex0 = Ux1 * Bx2 - Ux2 * Bx1;
				const auto Ex1 = -(Ux0 * Bx2 - Ux2 * Bx0);
				const auto Ex2 = Ux0 * Bx1 - Ux1 * Bx0;

				cc_a4_Ex0(i, j, k) = Ex0;
				cc_a4_Ex1(i, j, k) = Ex1;
				cc_a4_Ex2(i, j, k) = Ex2;

				int tmp = 0;
			});
		}

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
			std::array<amrex::FArrayBox, 2> ec_fabs_U_ieside;
			// define the four possible velocity field quantities that could be reconstructed at the cell-edge
			// also define the temporary velocity field quantities that will be used for computing the extrapolation
			ec_fabs_U_ieside[0].resize(box_ec_r, 1);
			ec_fabs_U_ieside[1].resize(box_ec_r, 1);
			// indexing: field[2: i-compnent][2: i-side of edge]
			// note: magnetic field components cannot be discontinuous along themselves (i.e., either side of the face where they are
			// stored), so there are only two possible values (sides), rather than four (quadrants of) possible reconstructed values
			std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;
			// initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
			// indexing: field[2: i-compnent][4: quadrant around edge]
			std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_Ui_q;
			// define quantities
			for (int icomp = 0; icomp < 2; ++icomp) {
				ec_fabs_Bi_ieside[icomp][0].resize(box_ec_r, 1);
				ec_fabs_Bi_ieside[icomp][1].resize(box_ec_r, 1);
				for (int iquad = 0; iquad < 4; ++iquad) {
					ec_fabs_Ui_q[icomp][iquad].resize(box_ec_r, 1);
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

				// create temporary FArrayBox for storing the face-centered velocity fields reconstructed from the cell-center
				// indexing: field[2: i-compnent][2: i-side of face]
				std::array<std::array<amrex::FArrayBox, 2>, 2> fc_fabs_Ui_ifside;
				// extrapolate both required cell-centered velocity fields to the cell-edge
				for (int icomp = 0; icomp < 2; ++icomp) {
					const int wcomp = extrap_dirs[icomp];
					fc_fabs_Ui_ifside[icomp][0].resize(box_fc_U, 1);
					fc_fabs_Ui_ifside[icomp][1].resize(box_fc_U, 1);
					// extrapolate cell-centered velocity components to the cell-face
					MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_Ux[wcomp].array(), fc_fabs_Ui_ifside[icomp][0].array(),
									    fc_fabs_Ui_ifside[icomp][1].array(), box_cc_U, reconstructionOrder);
					// extrapolate face-centered velocity components to the cell-edge
					for (int iface = 0; iface < 2; ++iface) {
						// reset values in temporary FArrayBox
						ec_fabs_U_ieside[0].setVal<amrex::RunOn::Device>(0.0);
						ec_fabs_U_ieside[1].setVal<amrex::RunOn::Device>(0.0);
						// extrapolate face-centered velocity component to the cell-edge
						MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Ui_ifside[icomp][iface].array(),
										    ec_fabs_U_ieside[0].array(), ec_fabs_U_ieside[1].array(), box_fc,
										    reconstructionOrder);
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

				int tmp = 0; // TODO: for debuging. remove
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
				ec_fabs_E_q[iquad].resize(box_ec, 1);
				const auto &E2_qi = ec_fabs_E_q[iquad].array();
				int w0_comp = extrap_dirs[0];
				int w1_comp = extrap_dirs[1];
				amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					const double u0 = U0_qi(i, j, k);
					const double u1 = U1_qi(i, j, k);
					const double b0 = B0_qi(i, j, k);
					const double b1 = B1_qi(i, j, k);
					double uxb = u0 * b1 - u1 * b0;

					// amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = geom.CellSizeArray();
					// amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = geom.ProbLoArray();
					// amrex::Real x1 = prob_lo[0] + i * dx[0];
					// if ((w0_comp != 0) && (w1_comp != 0)) { x1 = x1 + static_cast<amrex::Real>(0.5) * dx[0]; }
					// std::array<const double, 3> vel = computeWaveSolution(x1, 0, time);
					// std::array<const double, 3> mag = computeWaveSolution(x1, 1, time);
					// const double u0_exact = vel[w0_comp];
					// const double u1_exact = vel[w1_comp];
					// const double b0_exact = mag[w0_comp];
					// const double b1_exact = mag[w1_comp];
					// const double uxb_exact = u0_exact * b1_exact - u1_exact * b0_exact;
					// const double uxb_mix = u0_exact * b1 - u1_exact * b0;

					E2_qi(i, j, k) = uxb;

					int tmp = 0;
				});

				int tmp = 0; // TODO: for debuging. remove
			}

			// extract wavespeeds
			int w0_comp = extrap_dirs[0];
			int w1_comp = extrap_dirs[1];
			std::array<int, 3> delta_w0 = {0, 0, 0};
			std::array<int, 3> delta_w1 = {0, 0, 0};
			delta_w0[w0_comp] = 1;
			delta_w1[w1_comp] = 1;
			const auto &fspd_x0 = fcx_mf_fspds[w0_comp][mfi].const_array();
			const auto &fspd_x1 = fcx_mf_fspds[w1_comp][mfi].const_array();
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
			// extract cell-centered electric field
			const auto &E2_cc = cc_fabs_Ex[iedge].const_array();
			// compute electric field on the cell-edge
			const auto &E2_ave = ec_mf_emf_comps[iedge][mfi].array();
			// only operate on the real cells
			amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				const double E2_q0_ = E2_q0(i, j, k);
				const double E2_q1_ = E2_q1(i, j, k);
				const double E2_q2_ = E2_q2(i, j, k);
				const double E2_q3_ = E2_q3(i, j, k);
				// Balsara & Spicer scheme:
				// E2_ave(i, j, k) = 0.25 * (E2_q0_ + E2_q1_ + E2_q2_ + E2_q3_);

				// GS05 E^0_c scheme:
				// const double E2_cc_0 = E2_cc(i, j, k);
				// const double E2_cc_1 = E2_cc(i - delta_w0[0], j - delta_w0[1], k - delta_w0[2]);
				// const double E2_cc_2 = E2_cc(i - delta_w1[0], j - delta_w1[1], k - delta_w1[2]);
				// const double E2_cc_3 = E2_cc(i - delta_w0[0] - delta_w1[0], j - delta_w0[1] - delta_w1[1], k - delta_w0[2] - delta_w1[2]);
				// E2_ave(i, j, k) = 0.5 * (E2_q0_ + E2_q1_ + E2_q2_ + E2_q3_) - 0.25 * (E2_cc_0 + E2_cc_1 + E2_cc_2 + E2_cc_3);

				// LD04 scheme:
				const double fspd_x0_m = std::max(fspd_x0(i, j, k, 0), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 0));
				const double fspd_x0_p = std::max(fspd_x0(i, j, k, 1), fspd_x0(i + delta_w0[0], j + delta_w0[1], k + delta_w0[2], 1));
				const double fspd_x1_m = std::max(fspd_x1(i, j, k, 0), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 0));
				const double fspd_x1_p = std::max(fspd_x1(i, j, k, 1), fspd_x1(i + delta_w1[0], j + delta_w1[1], k + delta_w1[2], 1));
				const double B0_p_ = B0_p(i, j, k);
				const double B0_m_ = B0_m(i, j, k);
				const double B1_p_ = B1_p(i, j, k);
				const double B1_m_ = B1_m(i, j, k);
				const double denominator = (fspd_x0_m + fspd_x0_p) * (fspd_x1_m + fspd_x1_p);
				E2_ave(i, j, k) = ((fspd_x0_p * fspd_x1_p * E2_q0_ + fspd_x0_p * fspd_x1_m * E2_q1_ + fspd_x0_m * fspd_x1_m * E2_q2_ +
						    fspd_x0_m * fspd_x1_p * E2_q3_) /
						       denominator -
						   fspd_x1_m * fspd_x1_p / (fspd_x1_m + fspd_x1_p) * (B0_p_ - B0_m_) +
						   fspd_x0_m * fspd_x0_p / (fspd_x0_m + fspd_x0_p) * (B1_p_ - B1_m_));

				// if (emf_index == 1) {
				//   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = geom.CellSizeArray();
				//   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = geom.ProbLoArray();
				//   amrex::Real x1 = prob_lo[0] + i * dx[0];
				//   constexpr double sound_speed = 1.0;
				//   constexpr double gamma = 5. / 3.;
				//   constexpr double bg_density = 1.0;
				//   constexpr double bg_pressure = 1.0;
				//   constexpr double bg_mag_amplitude = 10.0;
				//   constexpr double theta_degrees = 0.0; // degrees
				//   constexpr double num_modes = 2;
				//   constexpr double k_amplitude = 2 * M_PI * num_modes;
				//   constexpr double delta_b = 1e-6;
				//   const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
				//   const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);
				//   const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
				//   const double bg_mag_x1 = bg_mag_amplitude * cos_theta;
				//   const double bg_mag_x2 = bg_mag_amplitude * sin_theta;
				//   const double omega = std::sqrt(std::pow(alfven_speed,2) * std::pow(k_amplitude,2) * std::pow(cos_theta,2));
				//   const double cos_wave = std::cos(omega * time - k_amplitude * x1);
				//   E2_ave(i,j,k) = -bg_mag_amplitude * omega * delta_b / (sound_speed * k_amplitude * cos_theta) * cos_wave;
				// } else {
				//   E2_ave(i,j,k) = 0;
				// }
			});

			// std::ofstream ofs("fab-emf-e"+std::to_string((wsolve+2)%3)+"-"+std::to_string(mfi.index()));
			// ec_mf_emf_comps[3-w0_comp-w1_comp][mfi].writeOn(ofs, 0, 1);

			int tmp = 0; // TODO: for debuging. remove
		}
	}
}

template <typename problem_t> std::array<const double, 3> MHDSystem<problem_t>::computeWaveSolution(double x1, int itype, double time)
{
	// constants
	constexpr double sound_speed = 1.0;
	constexpr double gamma = 5. / 3.;
	// background states
	constexpr double bg_density = 1.0;
	constexpr double bg_pressure = 1.0;
	constexpr double bg_mag_amplitude = 10.0;
	// alignment of magnetic field with the direction of wave propogation (in the x1-x2 plane). recall that hat(k) = (1, 0, 0) and hat(delta_u) = (0, 1, 0)
	constexpr double theta_degrees = 0.0; // degrees
	// k = 2 pi / wave length; box length = 1, so |k| in [1, inf)
	constexpr double num_modes = 2;
	constexpr double k_amplitude = 2 * M_PI * num_modes;
	// input perturbation: choose to do this via the relative denisty field in [0, 1]. remember, the linear regime is valid when this perturbation is small
	constexpr double delta_b = 1e-6;

	const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
	const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);

	const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
	const double bg_mag_x1 = bg_mag_amplitude * cos_theta;
	const double bg_mag_x2 = bg_mag_amplitude * sin_theta;

	const double omega = std::sqrt(std::pow(alfven_speed, 2) * std::pow(k_amplitude, 2) * std::pow(cos_theta, 2));

	const double cos_wave = std::cos(omega * time - k_amplitude * x1);

	const double x1mag = bg_mag_x1;
	const double x2mag = bg_mag_x2;
	const double x3mag = bg_mag_amplitude * delta_b * cos_wave;

	const double x1vel = 0;
	const double x2vel = 0;
	const double x3vel = -omega * delta_b / (sound_speed * k_amplitude * cos_theta) * cos_wave;

	if (itype == 0) {
		std::array<const double, 3> array = {x1vel, x2vel, x3vel};
		return array;
	} else {
		std::array<const double, 3> array = {x1mag, x2mag, x3mag};
		return array;
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid,
					 const int reconstructionOrder)
{
	amrex::Box const &box_r = amrex::grow(box_cValid, 1);
	amrex::Box const &box_r_x1 = amrex::surroundingNodes(box_r, static_cast<int>(dir));
	if (reconstructionOrder == 3) {
		// note: only box_r is used. box_r_x1 is unused.
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
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1, SlopeLimiter::MC>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2, SlopeLimiter::MC>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3, SlopeLimiter::MC>(cState, lState, rState, box_r_x1, 1);
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
		amrex::Abort("Invalid reconstruction order specified!");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf,
					     std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, const double dt,
					     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo,
					     double time)
{
	// compute the total right-hand-side for the MOL integration

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	// loop over faces pointing in the w0-direction
	for (int w0 = 0; w0 < 3; ++w0) {
		// you have two edges on the perimeter of this face
		int w1 = (w0 + 1) % 3; // vec_fc(w0) + vec_fc(w1)
		int w2 = (w0 + 2) % 3; // vec_fc(w0) + vec_fc(w2)
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
		// std::cout << w0 << ", " << w1 << ", " << w2 << ", " << delta_w1[0] << ", " << delta_w1[1] << ", " << delta_w1[2] << ", " << delta_w2[0] << ",
		// "
		//	  << delta_w2[1] << ", " << delta_w2[2] << std::endl;

		amrex::ParallelFor(fc_consVarNew_mf[w0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// the ec emfs sit in the opposite fc directions relative to the face
			double emf_w1_m = ec_emf_w1[bx](i, j, k);
			double emf_w2_m = ec_emf_w2[bx](i, j, k);
			double emf_w1_p = ec_emf_w1[bx](i + delta_w2[0], j + delta_w2[1], k + delta_w2[2]);
			double emf_w2_p = ec_emf_w2[bx](i + delta_w1[0], j + delta_w1[1], k + delta_w1[2]);
			double db_dt = (dx1 * (emf_w1_m - emf_w1_p) + dx2 * (emf_w2_p - emf_w2_m)) / (dx1 * dx2);
			if (w0 == 2) {
				amrex::Real x1 = prob_lo[0] + i * dx[0] + static_cast<amrex::Real>(0.5) * dx[0];
				constexpr double sound_speed = 1.0;
				constexpr double gamma = 5. / 3.;
				constexpr double bg_density = 1.0;
				constexpr double bg_pressure = 1.0;
				constexpr double bg_mag_amplitude = 10.0;
				constexpr double theta_degrees = 0.0;
				constexpr double num_modes = 2;
				constexpr double k_amplitude = 2 * M_PI * num_modes;
				constexpr double delta_b = 1e-6;
				const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
				const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);
				const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
				const double bg_mag_x1 = bg_mag_amplitude * cos_theta;
				const double bg_mag_x2 = bg_mag_amplitude * sin_theta;
				const double omega = std::sqrt(std::pow(alfven_speed, 2) * std::pow(k_amplitude, 2) * std::pow(cos_theta, 2));
				const double cos_wave = std::cos(omega * time - k_amplitude * x1);
				const double x1mag = bg_mag_x1;
				const double x2mag = bg_mag_x2;
				const double x3mag = delta_b * cos_wave;
				const double x1vel = 0;
				const double x2vel = 0;
				const double x3vel = -omega * delta_b / (sound_speed * k_amplitude * cos_theta) * cos_wave;
				double db_dt_exact = -bg_mag_amplitude * omega * delta_b / cos_theta * std::sin(omega * time - k_amplitude * x1);
				// const double y_new = fc_consVarOld(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
				// const double y_exact = delta_b * std::cos(omega * time - k_amplitude * x1);
				int tmp = 0; // for debug
			}
			fc_consVarNew[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) =
			    fc_consVarOld[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
		});

		// for (amrex::MFIter mfi(fc_consVarNew_mf[w0]); mfi.isValid(); ++mfi) {
		//   const amrex::Box &box_fc = mfi.validbox();
		//   amrex::FArrayBox fab_bnew(box_fc, 1, amrex::The_Async_Arena());
		//   auto const &ec_emf_w1 = ec_emf_mf[w1].const_array(mfi);
		//   auto const &ec_emf_w2 = ec_emf_mf[w2].const_array(mfi);
		//   auto const &fc_consVarOld = fc_consVarOld_mf[w0].const_array(mfi);
		//   auto fc_consVarNew = fc_consVarNew_mf[w0].array(mfi);
		//   auto array_bnew = fab_bnew.array();
		//   auto const dx1 = dx[w1];
		//   auto const dx2 = dx[w2];
		//   amrex::ParallelFor(box_fc, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		//     double emf_w1_m = ec_emf_w1(i, j, k);
		//     double emf_w2_m = ec_emf_w2(i, j, k);
		//     double emf_w1_p = ec_emf_w1(i+delta_w2[0], j+delta_w2[1], k+delta_w2[2]);
		//     double emf_w2_p = ec_emf_w2(i+delta_w1[0], j+delta_w1[1], k+delta_w1[2]);
		//     double db_dt = 0; //(dx1 * (emf_w1_m - emf_w1_p) + dx2 * (emf_w2_p - emf_w2_m)) / (dx1 * dx2);
		//     if (w0 == 2) {
		//       amrex::Real x1 = prob_lo[0] + i * dx[0] + static_cast<amrex::Real>(0.5) * dx[0];
		//       constexpr double sound_speed = 1.0;
		//       constexpr double gamma = 5. / 3.;
		//       constexpr double bg_density = 1.0;
		//       constexpr double bg_pressure = 1.0;
		//       constexpr double bg_mag_amplitude = 10.0;
		//       constexpr double theta_degrees = 0.0;
		//       constexpr double num_modes = 2;
		//       constexpr double k_amplitude = 2 * M_PI * num_modes;
		//       constexpr double delta_b = 1e-6;
		//       const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
		//       const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);
		//       const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
		//       const double bg_mag_x1 = bg_mag_amplitude * cos_theta;
		//       const double bg_mag_x2 = bg_mag_amplitude * sin_theta;
		//       const double omega = std::sqrt(std::pow(alfven_speed,2) * std::pow(k_amplitude,2) * std::pow(cos_theta,2));
		//       const double cos_wave = std::cos(omega * time - k_amplitude * x1);
		//       const double x1mag = bg_mag_x1;
		//       const double x2mag = bg_mag_x2;
		//       const double x3mag = delta_b * cos_wave;
		//       const double x1vel = 0;
		//       const double x2vel = 0;
		//       const double x3vel = -omega * delta_b / (sound_speed * k_amplitude * cos_theta) * cos_wave;
		//       db_dt = -omega * delta_b / cos_theta * std::sin(omega * time - k_amplitude * x1);
		//       const double y_new = fc_consVarOld(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
		//       const double y_exact = delta_b * std::cos(omega * time - k_amplitude * x1);
		//       if (((i == 0) && (j == 0)) && (k == 0)) {
		//         std::cout << y_new - y_exact << std::endl;
		//       }
		//     }
		//     fc_consVarNew(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) = fc_consVarOld(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) +
		//     dt * db_dt;
		//   });
		//   std::ofstream ofs("fab-bnew-f"+std::to_string(w0));
		//   fab_bnew.writeOn(ofs, 0, 1);
		// }

		int tmp = 0; // TODO: for debuging. remove
	}
}

#endif // HYDRO_SYSTEM_HPP_
