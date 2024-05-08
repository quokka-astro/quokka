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
#include "physics_info.hpp"
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
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

  static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fcx_mf_rhs, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nghost_cc, int nghost_fc);

  static void ReconstructTo(FluxDir dir, amrex::FArrayBox const &q_fab, amrex::FArrayBox &leftState_fab, amrex::FArrayBox &rightState_fab, const amrex::Box &indexRange, int reconstructionOrder, int nghost);
};

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fcx_mf_rhs, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int nghost_cc, const int nghost_fc)
{
  std::cout << "Computing EMF" << std::endl;

  // Loop over each box-array on the level
  // Note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
  for (amrex::MFIter mfi(cc_mf_cVars); mfi.isValid(); ++mfi) {
    const amrex::Box &box_cc = mfi.validbox();

    // In this function we distinguish between world (w:3), array (i:2), quandrant (q:4), and component (x:3) indexing with prefixes. We will use the x-prefix when the w- and i- indexes are the same.
    // We will minimise the storage footprint by only computing and holding onto the quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but also significantly reduces the memory footprint, which is a bigger bottleneck.

    // initialise the rhs of the induction equation
    for (int windex = 0; windex < AMREX_SPACEDIM; ++windex) {
      const auto &fcxw_a4_rhs = fcx_mf_rhs[windex][mfi].array();
      amrex::ParallelFor(box_cc, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        fcxw_a4_rhs(i, j, k, MHDSystem<problem_t>::bfield_index) = 0;
      });
    }

    // extract cell-centered velocity fields
    // indexing: field[3: x-component]
    std::array<amrex::FArrayBox, 3> cc_fabs_Ux;
    cc_fabs_Ux[0].resize(box_cc, 1);
    cc_fabs_Ux[1].resize(box_cc, 1);
    cc_fabs_Ux[2].resize(box_cc, 1);
    const auto &cc_a4_Ux0 = cc_fabs_Ux[0].array();
    const auto &cc_a4_Ux1 = cc_fabs_Ux[1].array();
    const auto &cc_a4_Ux2 = cc_fabs_Ux[2].array();
    const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
    amrex::ParallelFor(box_cc, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
      const auto px1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
      const auto px2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
      const auto px3 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
      cc_a4_Ux0(i,j,k) = px1 / rho;
      cc_a4_Ux1(i,j,k) = px2 / rho;
      cc_a4_Ux2(i,j,k) = px3 / rho;
    });

    // // indexing: field[3: x-component/x-face]
    // // create a view of the b-field data (do not make another copy)
    // std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
    //   FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
    //   FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
    //   FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
    // };

    // alternatively, make a complete copy of the b-field data
    std::array<amrex::FArrayBox, 3> fc_fabs_Bx;
    for (int w_index = 0; w_index < 3; ++w_index) {
      const amrex::Box box_fc = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(w_index));
      fc_fabs_Bx[w_index].resize(box_fc, 1);
      const auto &fc_a4_Bx = fc_fabs_Bx[w_index].array();
      // extract face-centered magnetic fields
      const auto &fc_a4_cVars = fcx_mf_cVars[w_index][mfi].const_array();
      amrex::ParallelFor(box_fc, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        fc_a4_Bx(i, j, k) = fc_a4_cVars(i, j, k, MHDSystem<problem_t>::bfield_index);
      });
    }


    // compute the magnetic flux through each cell-face
    for (int wsolve = 0; wsolve < 3; ++wsolve) {
      // electric field on the cell-edge
      // indexing: field[2: i-edge on cell-face]
      std::array<amrex::FArrayBox, 2> eci_fabs_E;
      // to solve for the magnetic flux through each face we compute the line integral of the EMF around the cell-face.
      // so let's solve for the EMF along each of the two edges along the edge of the cell-face
      for (int iedge_rel2face = 0; iedge_rel2face < 2; ++iedge_rel2face) {
        // for each of the two cell-edges on the cell-face

        // define the two directions we need to extrapolate cell-centered velocity fields to get them to the cell-edge
        // we will want to compute E2 = (U0 * B1 - U1 * B0) along the cell-edge
        std::array<int, 2> w_extrap_dirs = {
          wsolve, // dir-0: w-direction of the cell-face being solved for (relative to the cell-center)
          (wsolve + iedge_rel2face + 1) % 3 // dir-1: w-direction of the cell-edge being solved for (relative to the cell-face)
        };
        const amrex::Box box_ec = amrex::convert(box_cc, amrex::IntVect::TheDimensionVector(w_extrap_dirs[0]) + amrex::IntVect::TheDimensionVector(w_extrap_dirs[1]));
        
        // define output electric field on the cell-edge
        eci_fabs_E[iedge_rel2face].resize(box_ec, 1);

        // initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
        // indexing: field[2: i-compnent][4: quadrant around edge]
        std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_Ui_q;
        // initialise temporary FArrayBox for storing the edge-centered velocity fields reconstructed from the cell-face
        // indexing: field[2: i-side of edge]
        std::array<amrex::FArrayBox, 2> ec_fabs_U_ieside;
        // define the four possible velocity field quantities that could be reconstructed at the cell-edge
        // also define the temporary velocity field quantities that will be used for computing the extrapolation
        ec_fabs_U_ieside[0].resize(box_ec, 1);
        ec_fabs_U_ieside[1].resize(box_ec, 1);
        // indexing: field[2: i-compnent][2: i-side of edge]
        // note: magnetic field components cannot be discontinuous along themselves (i.e., either side of the face where they are stored)
        std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;
        // define quantities
        for (int icomp = 0; icomp < 2; ++icomp) {
          ec_fabs_Bi_ieside[icomp][0].resize(box_ec, 1);
          ec_fabs_Bi_ieside[icomp][1].resize(box_ec, 1);
          for (int iquad = 0; iquad < 4; ++iquad) {
            ec_fabs_Ui_q[icomp][iquad].resize(box_ec, 1);
          }
        }

        // extrapolate the two required cell-centered velocity field components to the cell-edge
        // there are two possible permutations for doing this, that is getting cell-centered quanties to a cell-edge
        // first is cc->fc[dir-0]->ec and second is cc->fc[dir-1]->ec
        for (int iperm = 0; iperm < 2; ++iperm) {
          // for each permutation of extrapolating cc->ec

          // define quantities required for creating face-centered FArrayBox
          const int w_extrap_dir2face = w_extrap_dirs[iperm];
          const amrex::IntVect ivec_cc2fc = amrex::IntVect::TheDimensionVector(w_extrap_dir2face);
          const amrex::Box box_fc = amrex::convert(box_cc, ivec_cc2fc);
          const auto dir2face = static_cast<FluxDir>(w_extrap_dir2face);

          // define extrapolation direction to go from face to edge
          const int w_extrap_dir2edge = w_extrap_dirs[(iperm+1) % 2];
          const auto dir2edge = static_cast<FluxDir>(w_extrap_dir2edge);

          // create temporary FArrayBox for storing the face-centered velocity fields reconstructed from the cell-center
          // indexing: field[2: i-compnent][2: i-side of face]
          std::array<std::array<amrex::FArrayBox, 2>, 2> fc_fabs_Ui_ifside;
          // extrapolate both required cell-centered velocity fields to the cell-edge
          for (int icomp = 0; icomp < 2; ++icomp) {
            const int w_comp = w_extrap_dirs[icomp];
            fc_fabs_Ui_ifside[icomp][0].resize(box_fc, 1);
            fc_fabs_Ui_ifside[icomp][1].resize(box_fc, 1);
            // extrapolate cell-centered velocity components to the cell-face
            MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_Ux[w_comp], fc_fabs_Ui_ifside[icomp][0], fc_fabs_Ui_ifside[icomp][1], box_cc, 1, nghost_cc);
            // extrapolate face-centered velocity components to the cell-edge
            for (int iface = 0; iface < 2; ++iface) {
              // reset values in temporary FArrayBox
              ec_fabs_U_ieside[0].setVal(0.0);
              ec_fabs_U_ieside[1].setVal(0.0);
              // extrapolate face-centered velocity component to the cell-edge
              MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Ui_ifside[icomp][iface], ec_fabs_U_ieside[0], ec_fabs_U_ieside[1], box_fc, 1, nghost_cc);
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
              ec_fabs_Ui_q[icomp][iquad0].atomicAdd(ec_fabs_U_ieside[0], 0, 0, 1);
              ec_fabs_Ui_q[icomp][iquad1].atomicAdd(ec_fabs_U_ieside[1], 0, 0, 1);
            }
          }
        }
        // finish averaging the two different ways for extrapolating cc->ec velocity fields
        for (int icomp = 0; icomp < 2; ++icomp) {
          for (int iquad = 0; iquad < 4; ++iquad) {
            ec_fabs_Ui_q[icomp][iquad].mult(0.5, 0, 1);
          }
        }

        // extrapolate the two required face-centered magnetic field components to the cell-edge
        for (int icomp = 0; icomp < 2; ++icomp) {
          // define extrapolation direction to go from face to edge
          const int w_comp = w_extrap_dirs[icomp];
          const int w_extrap_dir2edge = w_extrap_dirs[(icomp+1) % 2];
          const auto dir2edge = static_cast<FluxDir>(w_extrap_dir2edge);
          const amrex::IntVect ivec_cc2fc = amrex::IntVect::TheDimensionVector(w_extrap_dir2edge);
          const amrex::Box box_fc = amrex::convert(box_cc, ivec_cc2fc);
          // extrapolate face-centered magnetic components to the cell-edge
          MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Bx[w_comp], ec_fabs_Bi_ieside[icomp][0], ec_fabs_Bi_ieside[icomp][1], box_fc, 1, nghost_fc);
        }

        // indexing: field[4: quadrant around edge]
        std::array<amrex::FArrayBox, 4> ec_fabs_E_q;
        // compute the EMF along the cell-edge
        for (int iquad = 0; iquad < 4; ++iquad) {
          // define EMF FArrayBox
          ec_fabs_E_q[iquad].resize(box_ec, 1);
          // extract relevant velocity and magnetic field components
          const auto &U0_qi = ec_fabs_Ui_q[0][iquad].const_array();
          const auto &U1_qi = ec_fabs_Ui_q[1][iquad].const_array();
          const auto &B0_qi = ec_fabs_Bi_ieside[0][(iquad == 0 || iquad == 3) ? 0 : 1].const_array();
          const auto &B1_qi = ec_fabs_Bi_ieside[1][(iquad < 2) ? 0 : 1].const_array();
          // compute electric field in the quadrant about the cell-edge: cross product between velocity and magnetic field in that quadrant
          const auto &E2_qi = ec_fabs_E_q[iquad].array();
          amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            E2_qi(i,j,k) = U0_qi(i,j,k) * B1_qi(i,j,k) - U1_qi(i,j,k) * B0_qi(i,j,k);
          });
        }

        // // extract both components of magnetic field either side of the cell-edge
        // const auto &B0_m = ec_fabs_Bi_ieside[0][0].const_array();
        // const auto &B0_p = ec_fabs_Bi_ieside[0][1].const_array();
        // const auto &B1_m = ec_fabs_Bi_ieside[1][0].const_array();
        // const auto &B1_p = ec_fabs_Bi_ieside[1][1].const_array();
        // // extract all four quadrants of the electric field about the cell-edge
        // const auto &E2_q0 = ec_fabs_E_q[0].const_array();
        // const auto &E2_q1 = ec_fabs_E_q[1].const_array();
        // const auto &E2_q2 = ec_fabs_E_q[2].const_array();
        // const auto &E2_q3 = ec_fabs_E_q[3].const_array();
        // // compute electric field on the cell-edge
        // const auto &E2_ave = eci_fabs_E[iedge_rel2face].array();
        // amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        //   const double spd_x0_m = 1.0;
        //   const double spd_x0_p = 1.0;
        //   const double spd_x1_m = 1.0;
        //   const double spd_x1_p = 1.0;
        //   E2_ave(i,j,k) = \
        //       spd_x0_p * spd_x1_p * E2_q0(i,j,k) +
        //       spd_x0_m * spd_x1_p * E2_q3(i,j,k) +
        //       spd_x0_m * spd_x1_m * E2_q1(i,j,k) +
        //       spd_x0_p * spd_x1_m * E2_q2(i,j,k) -
        //       spd_x1_p * spd_x1_m / (spd_x1_p + spd_x1_m) * (B0_p(i,j,k) - B0_m(i,j,k)) +
        //       spd_x0_p * spd_x0_m / (spd_x0_p + spd_x0_m) * (B1_p(i,j,k) - B1_m(i,j,k));
        // });

        // const int index_E2comp = MHDSystem<problem_t>::bfield_index + (wsolve + 2*iedge_rel2face + 2) % 3;
        // std::array<int, 3> idx = {0, 0, 0};
        // idx[index_E2comp] = 1;
        // // compute the magnetic flux
        // const auto &fcxw_a4_rhs = fcx_mf_rhs[wsolve][mfi].array();
        // amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        //   // the induction equation is written in an additive form: the RHS is evaluated in parts as each edge-centered electric field is computed
        //   fcxw_a4_rhs(i,j,k,MHDSystem<problem_t>::bfield_index) += (E2_ave(i,j,k) - E2_ave(i-idx[0],j-idx[1],k-idx[2])) / dx[wsolve];
        // });
      }
    }
  }

  std::cout << "Done computing EMF" << std::endl;
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, amrex::FArrayBox const &q_fab, amrex::FArrayBox &leftState_fab, amrex::FArrayBox &rightState_fab, const amrex::Box &indexRange, const int reconstructionOrder, const int nghost)
{
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, static_cast<int>(dir));

  if (reconstructionOrder == 3) {
    switch (dir) {
      case FluxDir::X1:
        MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(q_fab, leftState_fab, rightState_fab, reconstructRange, 1);
        break;
      case FluxDir::X2:
        MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(q_fab, leftState_fab, rightState_fab, reconstructRange, 1);
        break;
      case FluxDir::X3:
        MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(q_fab, leftState_fab, rightState_fab, reconstructRange, 1);
        break;
    }
  } else if (reconstructionOrder == 2) {
    switch (dir) {
      case FluxDir::X1:
        MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
      case FluxDir::X2:
        MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
      case FluxDir::X3:
        MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
    }
  } else if (reconstructionOrder == 1) {
    switch (dir) {
      case FluxDir::X1:
        MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
      case FluxDir::X2:
        MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
      case FluxDir::X3:
        MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(q_fab, leftState_fab, rightState_fab, x1ReconstructRange, 1);
        break;
    }
  } else {
    amrex::Abort("Invalid reconstruction order specified!");
  }
}


#endif // HYDRO_SYSTEM_HPP_
