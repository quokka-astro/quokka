//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "test_hydro3d_blast.hpp"

struct SedovProblem {};

// if false, use octant symmetry instead
constexpr bool simulate_full_box = true;

template <> struct EOS_Traits<SedovProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = false;
  static constexpr int nscalars = 0;       // number of passive scalars
};

template <>
void RadhydroSimulation<SedovProblem>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
      geom[lev].ProbHiArray();

  amrex::Real x0 = NAN;
  amrex::Real y0 = NAN;
  amrex::Real z0 = NAN;
  if constexpr (simulate_full_box) {
    x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
    y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
    z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
  } else {
    x0 = 0.;
    y0 = 0.;
    z0 = 0.;
  }

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
      amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
      amrex::Real const r = std::sqrt(
          std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

      double vx = 0.;
      double vy = 0.;
      double vz = 0.;
      double rho = 1.0;
      double P = NAN;

      if (r < 0.1) { // inside sphere
        P = 10.;
      } else {
        P = 0.1;
      }

      AMREX_ASSERT(!std::isnan(vx));
      AMREX_ASSERT(!std::isnan(vy));
      AMREX_ASSERT(!std::isnan(vz));
      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(P));

      const auto v_sq = vx * vx + vy * vy + vz * vz;
      const auto gamma = HydroSystem<SedovProblem>::gamma_;

      for (int n = 0; n < state.nComp(); ++n) {
        state(i, j, k, n) = 0.; // zero fill all components
      }

      state(i, j, k, HydroSystem<SedovProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index) = rho * vx;
      state(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index) = rho * vy;
      state(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index) = rho * vz;
      state(i, j, k, HydroSystem<SedovProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * v_sq;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<SedovProblem>::ErrorEst(int lev,
                                                amrex::TagBoxArray &tags,
                                                amrex::Real /*time*/,
                                                int /*ngrow*/) {
  // tag cells for refinement

  const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
  const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

  for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real const P =
          HydroSystem<SedovProblem>::ComputePressure(state, i, j, k);

      amrex::Real const P_xplus =
          HydroSystem<SedovProblem>::ComputePressure(state, i + 1, j, k);
      amrex::Real const P_xminus =
          HydroSystem<SedovProblem>::ComputePressure(state, i - 1, j, k);
      amrex::Real const P_yplus =
          HydroSystem<SedovProblem>::ComputePressure(state, i, j + 1, k);
      amrex::Real const P_yminus =
          HydroSystem<SedovProblem>::ComputePressure(state, i, j - 1, k);
      amrex::Real const P_zplus =
          HydroSystem<SedovProblem>::ComputePressure(state, i, j, k + 1);
      amrex::Real const P_zminus =
          HydroSystem<SedovProblem>::ComputePressure(state, i, j, k - 1);

      amrex::Real const del_x =
          std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
      amrex::Real const del_y =
          std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));
      amrex::Real const del_z =
          std::max(std::abs(P_zplus - P), std::abs(P - P_zminus));

      amrex::Real const gradient_indicator =
          std::max({del_x, del_y, del_z}) / std::max(P, P_min);

      if (gradient_indicator > eta_threshold) {
        tag(i, j, k) = amrex::TagBox::SET;
      }
    });
  }
}

auto problem_main() -> int {
  auto isNormalComp = [=](int n, int dim) {
    if ((n == HydroSystem<SedovProblem>::x1Momentum_index) && (dim == 0)) {
      return true;
    }
    if ((n == HydroSystem<SedovProblem>::x2Momentum_index) && (dim == 1)) {
      return true;
    }
    if ((n == HydroSystem<SedovProblem>::x3Momentum_index) && (dim == 2)) {
      return true;
    }
    return false;
  };

  const int nvars = RadhydroSimulation<SedovProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      if constexpr (simulate_full_box) { // periodic boundaries
        boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
        boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
      } else { // octant symmetry
        if (isNormalComp(n, i)) {
          boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
          boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
        } else {
          boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
          boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
        }
      }
    }
  }

  // Problem initialization
  RadhydroSimulation<SedovProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
  sim.stopTime_ = 0.5;   // 0.01;
  sim.cflNumber_ = 0.25; // *must* be less than 1/3 in 3D!
  sim.maxTimesteps_ = 100;
  sim.plotfileInterval_ = -1;
  //sim.maxTimesteps_ = 10000;
  //sim.plotfileInterval_ = 100;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
