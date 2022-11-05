//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file spherical_collapse.cpp
/// \brief Defines a test problem for pressureless spherical collapse.
///
#include <limits>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_SPACE.H"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "spherical_collapse.hpp"

struct CollapseProblem {};

template <> struct HydroSystem_Traits<CollapseProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<CollapseProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;

  static constexpr int numPassiveScalars = 0; // number of passive scalars
};

constexpr double R_sphere = 0.5;

template <>
void RadhydroSimulation<CollapseProblem>::setInitialConditionsOnGrid(
    quokka::grid grid_elem) {
  // set initial conditions
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
  const amrex::Box &indexRange = grid_elem.indexRange_;
  const amrex::Array4<double>& state_cc = grid_elem.array_;
  
  amrex::Real x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  amrex::Real y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
  amrex::Real z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
    amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
    amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
    amrex::Real const r = std::sqrt(
        std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

    double rho = NAN;
    double P = 1.0e-5;
    if (r < R_sphere) {
      rho = 10.0;
    } else {
      rho = 1.0e-5;
    }

    AMREX_ASSERT(!std::isnan(rho));
    AMREX_ASSERT(!std::isnan(P));

    const amrex::Real gamma = HydroSystem<CollapseProblem>::gamma_;
    state_cc(i, j, k, HydroSystem<CollapseProblem>::density_index) = rho;
    state_cc(i, j, k, HydroSystem<CollapseProblem>::x1Momentum_index) = 0;
    state_cc(i, j, k, HydroSystem<CollapseProblem>::x2Momentum_index) = 0;
    state_cc(i, j, k, HydroSystem<CollapseProblem>::x3Momentum_index) = 0;
    state_cc(i, j, k, HydroSystem<CollapseProblem>::energy_index) = P / (gamma - 1.);
    state_cc(i, j, k, HydroSystem<CollapseProblem>::internalEnergy_index) = P / (gamma - 1.);
  });
}

template <>
void RadhydroSimulation<CollapseProblem>::ErrorEst(int lev,
                                                amrex::TagBoxArray &tags,
                                                amrex::Real /*time*/,
                                                int /*ngrow*/) {
  // tag cells for refinement
  const Real eta_threshold = 0.1;   // gradient refinement threshold
  const Real q_min = 1e-3;          // minimum density for refinement

  for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_cc_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);
    const int nidx = HydroSystem<CollapseProblem>::density_index;

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real const q = state(i, j, k, nidx);
      Real const q_xplus = state(i + 1, j, k, nidx);
      Real const q_xminus = state(i - 1, j, k, nidx);
      Real const q_yplus = state(i, j + 1, k, nidx);
      Real const q_yminus = state(i, j - 1, k, nidx);
      Real const q_zplus = state(i, j, k + 1, nidx);
      Real const q_zminus = state(i, j, k - 1, nidx);

      Real const del_x = 0.5 * (q_xplus - q_xminus);
      Real const del_y = 0.5 * (q_yplus - q_yminus);
      Real const del_z = 0.5 * (q_zplus - q_zminus);
      Real const gradient_indicator = std::sqrt(del_x*del_x + del_y*del_y + del_z*del_z) / q;

      if ((gradient_indicator > eta_threshold) && (q > q_min)) {
        tag(i, j, k) = amrex::TagBox::SET;
      }
    });
  }
}

auto problem_main() -> int {
  auto isNormalComp = [=](int n, int dim) {
    if ((n == HydroSystem<CollapseProblem>::x1Momentum_index) && (dim == 0)) {
      return true;
    }
    if ((n == HydroSystem<CollapseProblem>::x2Momentum_index) && (dim == 1)) {
      return true;
    }
    if ((n == HydroSystem<CollapseProblem>::x3Momentum_index) && (dim == 2)) {
      return true;
    }
    return false;
  };

  const int nvars = RadhydroSimulation<CollapseProblem>::nvarTotal_cc_;
  amrex::Vector<amrex::BCRec> BCs_cc(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      if (isNormalComp(n, i)) {
        BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
        BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
      } else {
        BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
        BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
      }
    }
  }

  // Problem initialization
  RadhydroSimulation<CollapseProblem> sim(BCs_cc);
  
  sim.reconstructionOrder_ = 2; // 2=PLM, 3=PPM
  sim.doPoissonSolve_ = 1;      // enable self-gravity

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  int status = 0;
  return status;
}
