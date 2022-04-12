//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_quirk.cpp
/// \brief Defines a test problem for the odd-even decoupling instability.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_quirk.hpp"
#include <algorithm>
#include <vector>

using Real = amrex::Real;

struct QuirkProblem {};

template <> struct EOS_Traits<QuirkProblem> {
  static constexpr double gamma = 5. / 3.;
  static constexpr bool reconstruct_eint = false;
};

constexpr Real dl = 3.692;
constexpr Real ul = -0.625;
constexpr Real pl = 26.85;
constexpr Real dr = 1.0;
constexpr Real ur = -5.0;
constexpr Real pr = 0.6;
AMREX_GPU_MANAGED int ishock = 0;

template <>
void RadhydroSimulation<QuirkProblem>::setInitialConditionsAtLevel(int lev) {
  // Initial conditions from:
  // T. Hanawa et al. / Journal of Computational Physics 227 (2008) 7952–7976
  // and based on Athena++'s quirk.cpp.

  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();

  Real xshock = 0.4;
  for (ishock = 0; (prob_lo[0] + dx[0] * (ishock + Real(0.5))) < xshock;
       ++ishock) {
  }
  ishock--;
  amrex::Print() << "ishock = " << ishock << "\n";

  Real dd = dl - 0.135;
  Real ud = ul + 0.219;
  Real pd = pl - 1.31;

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      double vx = NAN;
      double vy = 0.;
      double vz = 0.;
      double rho = NAN;
      double P = NAN;

      if (i <= ishock) {
        rho = dl;
        vx = ul;
        P = pl;
      } else {
        rho = dr;
        vx = ur;
        P = pr;
      }

      if ((i == ishock) && (j % 2 == 0)) {
        rho = dd;
        vx = ud;
        P = pd;
      }

      AMREX_ASSERT(!std::isnan(vx));
      AMREX_ASSERT(!std::isnan(vy));
      AMREX_ASSERT(!std::isnan(vz));
      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(P));

      const auto v_sq = vx * vx + vy * vy + vz * vz;
      const auto gamma = HydroSystem<QuirkProblem>::gamma_;

      state(i, j, k, HydroSystem<QuirkProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<QuirkProblem>::x1Momentum_index) = rho * vx;
      state(i, j, k, HydroSystem<QuirkProblem>::x2Momentum_index) = rho * vy;
      state(i, j, k, HydroSystem<QuirkProblem>::x3Momentum_index) = rho * vz;
      state(i, j, k, HydroSystem<QuirkProblem>::energy_index) =
          P / (gamma - 1.) + 0.5 * rho * v_sq;

      // initialize radiation variables to zero
      state(i, j, k, RadSystem<QuirkProblem>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<QuirkProblem>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<QuirkProblem>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<QuirkProblem>::x3RadFlux_index) = 0;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

auto getDeltaEntropyVector() -> std::vector<Real> & {
  static std::vector<Real> delta_s_vec;
  return delta_s_vec;
}

template <> void RadhydroSimulation<QuirkProblem>::computeAfterTimestep() {
  if (amrex::ParallelDescriptor::IOProcessor()) {
    // it should be sufficient examine a single box on level 0
    // (no AMR should be used for this problem, and the odd-even decoupling will
    // manifest in every box, if it happens)
    amrex::MultiFab &mf_state = state_new_[0];
    const int box_no = 0;
    auto const &state = mf_state.array(box_no);
    amrex::GpuArray<int, 3> box_lo = mf_state.box(box_no).loVect3d();
    const int jlo = box_lo[2];
    const int klo = box_lo[3];

    Real dodd =
        state(ishock, jlo + 1, klo, HydroSystem<QuirkProblem>::density_index);
    Real podd =
        HydroSystem<QuirkProblem>::ComputePressure(state, ishock, jlo + 1, klo);
    Real deven =
        state(ishock, jlo, klo, HydroSystem<QuirkProblem>::density_index);
    Real peven =
        HydroSystem<QuirkProblem>::ComputePressure(state, ishock, jlo, klo);

    // the 'entropy function' s == P / rho^gamma
    const Real gamma = HydroSystem<QuirkProblem>::gamma_;
    Real sodd = podd / std::pow(dodd, gamma);
    Real seven = peven / std::pow(deven, gamma);
    Real deltas = std::abs(sodd - seven);
    getDeltaEntropyVector().push_back(deltas);
  }
}

template <>
void RadhydroSimulation<QuirkProblem>::computeAfterEvolve(
    amrex::Vector<amrex::Real> & /*initSumCons*/) {
  if (amrex::ParallelDescriptor::IOProcessor()) {
    auto const &deltas_vec = getDeltaEntropyVector();
    const Real deltas = *std::max_element(deltas_vec.begin(), deltas_vec.end());

    if (deltas > 0.06) {
      amrex::Print()
          << "The scheme suffers from the Carbuncle phenomenon : max delta s = "
          << deltas << "\n\n";
      amrex::Abort("Carbuncle detected!");
    } else {
      amrex::Print()
          << "The scheme looks stable against the Carbuncle phenomenon : "
             "max delta s = "
          << deltas << "\n\n";
    }
  }
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<QuirkProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
    int /*orig_comp*/) {
#if (AMREX_SPACEDIM == 1)
  auto i = iv.toArray()[0];
  int j = 0;
  int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
  auto [i, j] = iv.toArray();
  int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
  auto [i, j, k] = iv.toArray();
#endif

  amrex::Box const &box = geom.Domain();
  amrex::GpuArray<int, 3> lo = box.loVect3d();
  amrex::GpuArray<int, 3> hi = box.hiVect3d();
  const auto gamma = HydroSystem<QuirkProblem>::gamma_;

  if (i < lo[0]) {
    // x1 left side boundary
    consVar(i, j, k, RadSystem<QuirkProblem>::gasEnergy_index) =
        pl / (gamma - 1.) + 0.5 * dl * ul * ul;
    consVar(i, j, k, RadSystem<QuirkProblem>::gasDensity_index) = dl;
    consVar(i, j, k, RadSystem<QuirkProblem>::x1GasMomentum_index) = dl * ul;
    consVar(i, j, k, RadSystem<QuirkProblem>::x2GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<QuirkProblem>::x3GasMomentum_index) = 0.;
  } else if (i >= hi[0]) {
    // x1 right-side boundary
    consVar(i, j, k, RadSystem<QuirkProblem>::gasEnergy_index) =
        pr / (gamma - 1.) + 0.5 * dr * ur * ur;
    consVar(i, j, k, RadSystem<QuirkProblem>::gasDensity_index) = dr;
    consVar(i, j, k, RadSystem<QuirkProblem>::x1GasMomentum_index) = dr * ur;
    consVar(i, j, k, RadSystem<QuirkProblem>::x2GasMomentum_index) = 0.;
    consVar(i, j, k, RadSystem<QuirkProblem>::x3GasMomentum_index) = 0.;
  }
}

auto problem_main() -> int {
  // Boundary conditions
  const int nvars = RadhydroSimulation<QuirkProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    // outflow
    boundaryConditions[0].setLo(0, amrex::BCType::ext_dir);
    boundaryConditions[0].setHi(0, amrex::BCType::ext_dir);
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      // periodic
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }
  }

  // Problem initialization
  RadhydroSimulation<QuirkProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.reconstructionOrder_ = 2; // PLM
  sim.stopTime_ = 0.4;
  sim.cflNumber_ = 0.4;
  sim.maxTimesteps_ = 2000;
  sim.plotfileInterval_ = -1;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}