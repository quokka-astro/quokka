//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include <tuple>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "AMReX_REAL.H"

#include "radiation_system.hpp"
#include "simulation.hpp"
#include "test_radiation_tophat.hpp"

struct TophatProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// "Tophat" pipe flow test (Gentile 2001)
constexpr double kelvin_to_eV = 8.617385e-5;

constexpr double kappa_wall = 200.0; // cm^2 g^-1 (specific opacity)
constexpr double rho_wall = 10.0;    // g cm^-3 (matter density)
constexpr double kappa_pipe = 20.0;  // cm^2 g^-1 (specific opacity)
constexpr double rho_pipe = 0.01;    // g cm^-3 (matter density)
constexpr double T_hohlraum = 500. / kelvin_to_eV;       // K [== 500 eV]
constexpr double T_initial = 50. / kelvin_to_eV;         // K [== 50 eV]
constexpr double c_v = (1.0e15 * 1.0e-6 * kelvin_to_eV); // erg g^-1 K^-1

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct RadSystem_Traits<TophatProblem> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = c_light_cgs_;
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = 5. / 3.;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = false;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputePlanckOpacity(
    const double rho, const double /*Tgas*/) -> double {
  amrex::Real kappa = 0.;
  if (rho == rho_pipe) {
    kappa = kappa_pipe;
  } else if (rho == rho_wall) {
    kappa = kappa_wall;
  } else {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!");
  }
  return kappa;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputeRosselandOpacity(
    const double rho, const double /*Tgas*/) -> double {
  amrex::Real kappa = 0.;
  if (rho == rho_pipe) {
    kappa = kappa_pipe;
  } else if (rho == rho_wall) {
    kappa = kappa_wall;
  } else {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!");
  }
  return kappa;
}

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TophatProblem>::ComputeTgasFromEgas(const double rho,
                                              const double Egas) -> double {
  return Egas / (rho * c_v);
}

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TophatProblem>::ComputeEgasFromTgas(const double rho,
                                              const double Tgas) -> double {
  return rho * c_v * Tgas;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputeEgasTempDerivative(
    const double rho, const double /*Tgas*/) -> double {
  // This is also known as the heat capacity, i.e.
  // 		\del E_g / \del T = \rho c_v,
  // for normal materials.

  return rho * c_v;
}

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TophatProblem>::ComputeEddingtonFactor(const double f_in) -> double {
  // compute Minerbo (1978) closure [piecewise approximation]
  // (For unknown reasons, this closure tends to work better
  // than the Levermore closure on the Su & Olson 1997 test.)
  const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
  const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f * f);
  return chi;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<TophatProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
    int /*orig_comp*/) {
#if (AMREX_SPACEDIM == 2)
  auto [i, j] = iv.toArray();
  int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
  auto [i, j, k] = iv.toArray();
#endif

  amrex::Real const *dx = geom.CellSize();
  auto const *prob_lo = geom.ProbLo();
  amrex::Box const &box = geom.Domain();
  amrex::GpuArray<int, 3> lo = box.loVect3d();

  amrex::Real const y0 = 0.;
  amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

  if (i < lo[0]) {
    // Marshak boundary condition
    double E_inc = NAN;

    const double E_0 =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::radEnergy_index);
    const double Fx_0 =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x1RadFlux_index);
    const double Fy_0 =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x2RadFlux_index);
    const double Fz_0 =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x3RadFlux_index);

    const double Egas =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::gasEnergy_index);
    const double rho =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::gasDensity_index);
    const double px =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x1GasMomentum_index);
    const double py =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x2GasMomentum_index);
    const double pz =
        consVar(lo[0], j, k, RadSystem<TophatProblem>::x3GasMomentum_index);

    double Fx_bdry = NAN;
    double Fy_bdry = NAN;
    double Fz_bdry = NAN;

    if (std::abs(y - y0) < 0.5) {
      E_inc = a_rad * std::pow(T_hohlraum, 4);
      Fx_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fx_0);
      Fy_bdry = 0.;
      Fz_bdry = 0.;
    } else {
      // extrapolated boundary
      E_inc = E_0;
      Fx_bdry = Fx_0;
      Fy_bdry = Fy_0;
      Fz_bdry = Fz_0;
    }
    const amrex::Real Fnorm =
        std::sqrt(Fx_bdry * Fx_bdry + Fy_bdry * Fy_bdry + Fz_bdry * Fz_bdry);
    AMREX_ASSERT((Fnorm / (c * E_inc)) < 1.0); // flux-limiting condition

    // x1 left side boundary (Marshak)
    consVar(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = E_inc;
    consVar(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = Fx_bdry;
    consVar(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = Fy_bdry;
    consVar(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = Fz_bdry;

    // extrapolated/outflow boundary for gas variables
    consVar(i, j, k, RadSystem<TophatProblem>::gasEnergy_index) = Egas;
    consVar(i, j, k, RadSystem<TophatProblem>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<TophatProblem>::x1GasMomentum_index) = px;
    consVar(i, j, k, RadSystem<TophatProblem>::x2GasMomentum_index) = py;
    consVar(i, j, k, RadSystem<TophatProblem>::x3GasMomentum_index) = pz;
  }
}

template <>
void RadhydroSimulation<TophatProblem>::setInitialConditionsOnGrid(
    array_t &state, const amrex::Box &indexRange, const amrex::Geometry &geom) {
  // extract variables required from the geom object
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
  // loop over the grid and set the initial condition
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    const double Erad = a_rad * std::pow(T_initial, 4);
    double rho = rho_wall;

    amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
    amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

    bool inside_region1 =
        ((((x > 0.) && (x <= 2.5)) || ((x > 4.5) && (x < 7.0))) &&
         (std::abs(y) < 0.5));
    bool inside_region2 =
        ((((x > 2.5) && (x < 3.0)) || ((x > 4.) && (x <= 4.5))) &&
         (std::abs(y) < 1.5));
    bool inside_region3 = (((x > 3.0) && (x < 4.0)) &&
                           ((std::abs(y) > 1.0) && (std::abs(y) < 1.5)));

    if (inside_region1 || inside_region2 || inside_region3) {
      rho = rho_pipe;
    }

    const double Egas =
        RadSystem<TophatProblem>::ComputeEgasFromTgas(rho, T_initial);

    state(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = Erad;
    state(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = 0;
    state(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = 0;
    state(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = 0;

    state(i, j, k, RadSystem<TophatProblem>::gasEnergy_index) = Egas;
    state(i, j, k, RadSystem<TophatProblem>::gasDensity_index) = rho;
    state(i, j, k, RadSystem<TophatProblem>::x1GasMomentum_index) = 0.;
    state(i, j, k, RadSystem<TophatProblem>::x2GasMomentum_index) = 0.;
    state(i, j, k, RadSystem<TophatProblem>::x3GasMomentum_index) = 0.;
  });
}

auto problem_main() -> int {
  // Problem parameters
  const int max_timesteps = 10000;
  const double CFL_number = 0.4;
  const double max_time = 5.0e-10; // s
  // const int nx = 700;
  // const int ny = 200;
  // const double Lx = 7.0;	// cm
  // const double Ly = 2.0;	// cm

  auto isNormalComp = [=](int n, int dim) {
    if ((n == RadSystem<TophatProblem>::x1RadFlux_index) && (dim == 0)) {
      return true;
    }
    if ((n == RadSystem<TophatProblem>::x2RadFlux_index) && (dim == 1)) {
      return true;
    }
    if ((n == RadSystem<TophatProblem>::x3RadFlux_index) && (dim == 2)) {
      return true;
    }
    if ((n == RadSystem<TophatProblem>::x1GasMomentum_index) && (dim == 0)) {
      return true;
    }
    if ((n == RadSystem<TophatProblem>::x2GasMomentum_index) && (dim == 1)) {
      return true;
    }
    if ((n == RadSystem<TophatProblem>::x3GasMomentum_index) && (dim == 2)) {
      return true;
    }
    return false;
  };

  // boundary conditions
  constexpr int nvars = RadhydroSimulation<TophatProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[n].setLo(0,
                                amrex::BCType::ext_dir); // left x1 -- Marshak
    boundaryConditions[n].setHi(
        0, amrex::BCType::foextrap); // right x1 -- extrapolate
    for (int i = 1; i < AMREX_SPACEDIM; ++i) {
      if (isNormalComp(n, i)) { // reflect lower
        boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
      } else {
        boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
      }
      // extrapolate upper
      boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
    }
  }

  // Problem initialization
  RadhydroSimulation<TophatProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = false;
  sim.is_radiation_enabled_ = true;
  sim.radiationReconstructionOrder_ = 2; // PLM
  sim.stopTime_ = max_time;
  sim.radiationCflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.plotfileInterval_ = 20;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
