//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

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
#include "radiation_system.hpp"
#include "test_hydro3d_blast.hpp"
#include <limits>

struct SedovProblem {};

// if false, use octant symmetry instead
constexpr bool simulate_full_box = false;

template <> struct HydroSystem_Traits<SedovProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = false;
  static constexpr int nscalars = 0;       // number of passive scalars
};

template <>
void RadhydroSimulation<SedovProblem>::setInitialConditionsAtLevel(int lev) {
  // initialize a Sedov test problem using parameters due to
  // Richard Klein and J. Bolstad
  // [Reference: J.R. Kamm and F.X. Timmes, On Efficient Generation of
  //   Numerically Robust Sedov Solutions, LA-UR-07-2849.]

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
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

  double rho = 1.0;          // g cm^-3
  double E_blast = 0.851072; // ergs
  double R0 = 0.025;         // cm

  if (!simulate_full_box) {
    E_blast /= 8.0; // only one octant, so 1/8 of the total energy
  }

  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      double rho_e = NAN;
#if 0
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
      amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
      amrex::Real const r = std::sqrt(
          std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

      if (r < R0) {
        rho_e = 1.0;
      } else {
        rho_e = 1.0e-10;
      }
#endif
      static_assert(!simulate_full_box, "single-cell initialization is only "
                                        "implemented for octant symmetry!");
      if ((i == 0) && (j == 0) && (k == 0)) {
        rho_e = E_blast / cell_vol;
      } else {
        rho_e = 1.0e-10 * (E_blast / cell_vol);
      }

      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(rho_e));

      for (int n = 0; n < state.nComp(); ++n) {
        state(i, j, k, n) = 0.; // zero fill all components
      }
      const auto gamma = HydroSystem<SedovProblem>::gamma_;

      state(i, j, k, HydroSystem<SedovProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index) = 0;
      state(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index) = 0;
      state(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index) = 0;
      state(i, j, k, HydroSystem<SedovProblem>::energy_index) = rho_e;
    });
  }

#if 0
  // normalize blast energy (does not work with AMR!)
  const Real E_sum =
      state_new_[lev].sum(HydroSystem<SedovProblem>::energy_index);
  const Real E_tot = E_sum * cell_vol;
  const Real norm = E_blast / E_tot;
  state_new_[lev].mult(norm, HydroSystem<SedovProblem>::energy_index, 1, 0);
#endif

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
          std::max({del_x, del_y, del_z}) / P;

      if ((gradient_indicator > eta_threshold) && (P > P_min)) {
        tag(i, j, k) = amrex::TagBox::SET;
      }
    });
  }
}

template <>
void RadhydroSimulation<SedovProblem>::computeAfterEvolve(
    amrex::Vector<amrex::Real> &initSumCons) {
  // check conservation of total energy
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

  amrex::Real const Egas0 =
      initSumCons[RadSystem<SedovProblem>::gasEnergy_index];
  amrex::Real const Erad0 =
      initSumCons[RadSystem<SedovProblem>::radEnergy_index];
  amrex::Real const Etot0 = Egas0 + (RadSystem<SedovProblem>::c_light_ /
                                     RadSystem<SedovProblem>::c_hat_) *
                                        Erad0;

  amrex::Real const Egas =
      state_new_[0].sum(RadSystem<SedovProblem>::gasEnergy_index) * vol;
  amrex::Real const Erad =
      state_new_[0].sum(RadSystem<SedovProblem>::radEnergy_index) * vol;
  amrex::Real const Etot = Egas + (RadSystem<SedovProblem>::c_light_ /
                                   RadSystem<SedovProblem>::c_hat_) *
                                      Erad;

  // compute kinetic energy
  amrex::MultiFab Ekin_mf(boxArray(0), DistributionMap(0), 1, 0);
  for (amrex::MFIter iter(state_new_[0]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[0].const_array(iter);
    auto const &ekin = Ekin_mf.array(iter);
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      // compute kinetic energy
      Real rho = state(i, j, k, HydroSystem<SedovProblem>::density_index);
      Real px = state(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index);
      Real py = state(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index);
      Real pz = state(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index);
      Real psq = px * px + py * py + pz * pz;
      ekin(i, j, k) = psq / (2.0 * rho) * vol;
    });
  }
  amrex::Real const Ekin = Ekin_mf.sum(0);

  amrex::Real const frac_Ekin = Ekin / Egas;
  amrex::Real const frac_Ekin_exact = 0.218729;

  amrex::Real const abs_err = (Etot - Etot0);
  amrex::Real const rel_err = abs_err / Etot0;

  amrex::Real const rel_err_Ekin = frac_Ekin - frac_Ekin_exact;

  amrex::Print() << "\nInitial gas+radiation energy = " << Etot0 << std::endl;
  amrex::Print() << "Final gas+radiation energy = " << Etot << std::endl;
  amrex::Print() << "\tabsolute conservation error = " << abs_err << std::endl;
  amrex::Print() << "\trelative conservation error = " << rel_err << std::endl;
  amrex::Print() << "\tkinetic energy = " << Ekin << std::endl;
  amrex::Print() << "\trelative K.E. error = " << rel_err_Ekin << std::endl;
  amrex::Print() << std::endl;

  if ((std::abs(rel_err) > 2.0e-15) || std::isnan(rel_err)) {
    amrex::Abort("Energy not conserved to machine precision!");
  } else {
    amrex::Print() << "Energy conservation is OK.\n";
  }

  if ((std::abs(rel_err_Ekin) > 0.01) || std::isnan(rel_err_Ekin)) {
    amrex::Abort(
        "Kinetic energy production is incorrect by more than 1 percent!");
  } else {
    amrex::Print() << "Kinetic energy production is OK.\n";
  }

  amrex::Print() << "\n";
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
  sim.stopTime_ = 1.0;          // seconds
  sim.cflNumber_ = 0.3;         // *must* be less than 1/3 in 3D!
  sim.maxTimesteps_ = 20000;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
