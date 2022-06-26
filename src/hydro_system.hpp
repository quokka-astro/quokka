#ifndef HYDRO_SYSTEM_HPP_ // NOLINT
#define HYDRO_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hydro_system.hpp
/// \brief Defines a class for solving the Euler equations.
///

// c++ headers

// library headers
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Loop.H"
#include "AMReX_REAL.H"

// internal headers
#include "ArrayView.hpp"
#include "hyperbolic_system.hpp"
#include "radiation_system.hpp"
#include "valarray.hpp"
#include <math.h>

// this struct is specialized by the user application code
//
template <typename problem_t> struct HydroSystem_Traits {
  static constexpr double gamma = 5. / 3.;     // default value
  static constexpr double cs_isothermal = NAN; // only used when gamma = 1
  static constexpr int nscalars = 0;           // number of passive scalars
  // if true, reconstruct e_int instead of pressure
  static constexpr bool reconstruct_eint = true;
};

/// Class for the Euler equations of inviscid hydrodynamics
///
template <typename problem_t>
class HydroSystem : public HyperbolicSystem<problem_t> {
public:
  enum consVarIndex {
    density_index = 0,
    x1Momentum_index = 1,
    x2Momentum_index = 2,
    x3Momentum_index = 3,
    energy_index = 4,
    internalEnergy_index = 5, // auxiliary internal energy (rho * e)
    scalar0_index = 6 // first passive scalar (only present if nscalars > 0!)
  };
  enum primVarIndex {
    primDensity_index = 0,
    x1Velocity_index = 1,
    x2Velocity_index = 2,
    x3Velocity_index = 3,
    pressure_index = 4,
    primEint_index = 5, // auxiliary internal energy (rho * e)
    primScalar0_index =
        6 // first passive scalar (only present if nscalars > 0!)
  };

  static constexpr int nscalars_ = HydroSystem_Traits<problem_t>::nscalars;
  static constexpr int nvar_ = 6 + nscalars_;

  static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons,
                                   array_t &primVar,
                                   amrex::Box const &indexRange);

  static void
  ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons,
                        array_t &maxSignal, amrex::Box const &indexRange);
  // requires GPU reductions
  static auto CheckStatesValid(amrex::Box const &indexRange,
                               amrex::Array4<const amrex::Real> const &cons)
      -> bool;
  static void EnforcePressureFloor(amrex::Real densityFloor,
                                   amrex::Real pressureFloor,
                                   amrex::Box const &indexRange,
                                   amrex::Array4<amrex::Real> const &state);

  AMREX_GPU_DEVICE static auto
  ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j,
                  int k) -> amrex::Real;

  AMREX_GPU_DEVICE static auto
  isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j,
               int k) -> bool;

  static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew,
                          std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
                          double dt_in,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
                          amrex::Box const &indexRange, int nvars,
                          amrex::Array4<int> const &redoFlag);

  static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
                           std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
                           double dt_in,
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
                           amrex::Box const &indexRange, int nvars,
                           amrex::Array4<int> const &redoFlag);

  static void AddInternalEnergyPressureTerm(
      amrex::Array4<amrex::Real> const &consVar,
      amrex::Array4<const amrex::Real> const &primVar,
      amrex::Box const &indexRange,
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
      amrex::Real const dt_in);

  template <FluxDir DIR>
  static void
  ComputeFluxes(array_t &x1Flux_in,
                amrex::Array4<const amrex::Real> const &x1LeftState_in,
                amrex::Array4<const amrex::Real> const &x1RightState_in,
                amrex::Array4<const amrex::Real> const &primVar_in,
                amrex::Box const &indexRange);

  template <FluxDir DIR>
  static void
  ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar,
                          array_t &x1FluxDiffusive,
                          amrex::Box const &indexRange);

  template <FluxDir DIR>
  static void
  ComputeFlatteningCoefficients(amrex::Array4<const amrex::Real> const &primVar,
                                array_t &x1Chi, amrex::Box const &indexRange);

  template <FluxDir DIR>
  static void FlattenShocks(amrex::Array4<const amrex::Real> const &q_in,
                            amrex::Array4<const amrex::Real> const &x1Chi_in,
                            amrex::Array4<const amrex::Real> const &x2Chi_in,
                            amrex::Array4<const amrex::Real> const &x3Chi_in,
                            array_t &x1LeftState_in, array_t &x1RightState_in,
                            amrex::Box const &indexRange, int nvars);

  // C++ does not allow constexpr to be uninitialized, even in a templated
  // class!
  static constexpr double gamma_ = HydroSystem_Traits<problem_t>::gamma;
  static constexpr double cs_iso_ = HydroSystem_Traits<problem_t>::cs_isothermal;
  static constexpr bool reconstruct_eint =
      HydroSystem_Traits<problem_t>::reconstruct_eint;

  static constexpr auto is_eos_isothermal() -> bool { return (gamma_ == 1.0); }
};

template <typename problem_t>
void HydroSystem<problem_t>::ConservedToPrimitive(
    amrex::Array4<const amrex::Real> const &cons, array_t &primVar,
    amrex::Box const &indexRange) {
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    const auto rho = cons(i, j, k, density_index);
    const auto px = cons(i, j, k, x1Momentum_index);
    const auto py = cons(i, j, k, x2Momentum_index);
    const auto pz = cons(i, j, k, x3Momentum_index);
    const auto E =
        cons(i, j, k, energy_index); // *total* gas energy per unit volume
    const auto Eint_aux = cons(i, j, k, internalEnergy_index);

    AMREX_ASSERT(!std::isnan(rho));
    AMREX_ASSERT(!std::isnan(px));
    AMREX_ASSERT(!std::isnan(py));
    AMREX_ASSERT(!std::isnan(pz));
    AMREX_ASSERT(!std::isnan(E));

    const auto vx = px / rho;
    const auto vy = py / rho;
    const auto vz = pz / rho;
    const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    const auto thermal_energy_cons = E - kinetic_energy;

    const auto P = thermal_energy_cons * (HydroSystem<problem_t>::gamma_ - 1.0);
    const auto eint_cons =
        thermal_energy_cons / rho; // specific internal energy

    AMREX_ASSERT(rho > 0.);
    if constexpr (!is_eos_isothermal()) {
      AMREX_ASSERT(P > 0.);
    }

    primVar(i, j, k, primDensity_index) = rho;
    primVar(i, j, k, x1Velocity_index) = vx;
    primVar(i, j, k, x2Velocity_index) = vy;
    primVar(i, j, k, x3Velocity_index) = vz;
    if constexpr (reconstruct_eint) {
      // save eint_cons
      primVar(i, j, k, pressure_index) = eint_cons;
    } else {
      // save pressure
      primVar(i, j, k, pressure_index) = P;
    }
    // save auxiliary internal energy (rho * e)
    primVar(i, j, k, primEint_index) = Eint_aux;

    // copy any passive scalars
    for (int nc = 0; nc < nscalars_; ++nc) {
      primVar(i, j, k, primScalar0_index + nc) =
          cons(i, j, k, scalar0_index + nc);
    }
  });
}

template <typename problem_t>
void HydroSystem<problem_t>::ComputeMaxSignalSpeed(
    amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal,
    amrex::Box const &indexRange) {
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    const auto rho = cons(i, j, k, density_index);
    const auto px = cons(i, j, k, x1Momentum_index);
    const auto py = cons(i, j, k, x2Momentum_index);
    const auto pz = cons(i, j, k, x3Momentum_index);
    AMREX_ASSERT(!std::isnan(rho));
    AMREX_ASSERT(!std::isnan(px));
    AMREX_ASSERT(!std::isnan(py));
    AMREX_ASSERT(!std::isnan(pz));

    const auto vx = px / rho;
    const auto vy = py / rho;
    const auto vz = pz / rho;
    const double vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
    double cs = NAN;

    if constexpr (is_eos_isothermal()) {
      cs = cs_iso_;
    } else {
      const auto E =
          cons(i, j, k, energy_index); // *total* gas energy per unit volume
      AMREX_ASSERT(!std::isnan(E));
      const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
      const auto thermal_energy = E - kinetic_energy;
      const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);
      cs = std::sqrt(HydroSystem<problem_t>::gamma_ * P / rho);
    }
    AMREX_ASSERT(cs > 0.);

    const double signal_max = cs + vel_mag;
    maxSignal(i, j, k) = signal_max;
  });
}

template <typename problem_t>
auto HydroSystem<problem_t>::CheckStatesValid(
    amrex::Box const &indexRange, amrex::Array4<const amrex::Real> const &cons)
    -> bool {
  bool areValid = true;
  AMREX_LOOP_3D(indexRange, i, j, k, {
    const auto rho = cons(i, j, k, density_index);
    const auto px = cons(i, j, k, x1Momentum_index);
    const auto py = cons(i, j, k, x2Momentum_index);
    const auto pz = cons(i, j, k, x3Momentum_index);
    const auto E =
        cons(i, j, k, energy_index); // *total* gas energy per unit volume
    const auto vx = px / rho;
    const auto vy = py / rho;
    const auto vz = pz / rho;
    const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    const auto thermal_energy = E - kinetic_energy;
    const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);

    bool negativeDensity = (rho <= 0.);
    bool negativePressure = (P <= 0.);

    if constexpr (is_eos_isothermal()) {
      if (negativeDensity) {
        areValid = false;
        printf("invalid state at (%d, %d, %d): rho %g\n", i, j, k, rho);
      }
    } else {
      if (negativeDensity || negativePressure) {
        areValid = false;
        printf("invalid state at (%d, %d, %d): "
               "rho %g, Etot %g, Eint %g, P %g\n",
               i, j, k, rho, E, thermal_energy, P);
      }
    }
  })

  return areValid;
}

template <typename problem_t>
void HydroSystem<problem_t>::EnforcePressureFloor(
    amrex::Real const densityFloor, amrex::Real const pressureFloor,
    amrex::Box const &indexRange, amrex::Array4<amrex::Real> const &state) {
  // prevent vacuum creation
  amrex::Real const rho_floor = densityFloor; // workaround nvcc bug
  amrex::Real const P_floor = pressureFloor;

  amrex::ParallelFor(
      indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real const rho = state(i, j, k, density_index);
        amrex::Real const vx1 = state(i, j, k, x1Momentum_index) / rho;
        amrex::Real const vx2 = state(i, j, k, x2Momentum_index) / rho;
        amrex::Real const vx3 = state(i, j, k, x3Momentum_index) / rho;
        amrex::Real const vsq = (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);
        amrex::Real const Etot = state(i, j, k, energy_index);

        amrex::Real rho_new = rho;
        if (rho < rho_floor) {
          rho_new = rho_floor;
          state(i, j, k, density_index) = rho_new;
        }

        if (!is_eos_isothermal()) {
          // recompute gas energy (to prevent P < 0)
          amrex::Real const Eint_star = Etot - 0.5 * rho_new * vsq;
          amrex::Real const P_star = Eint_star * (gamma_ - 1.);
          amrex::Real P_new = P_star;
          if (P_star < P_floor) {
            P_new = P_floor;
#pragma nv_diag_suppress divide_by_zero
            amrex::Real const Etot_new =
                P_new / (gamma_ - 1.) + 0.5 * rho_new * vsq;
            state(i, j, k, energy_index) = Etot_new;
          }
        }
      });
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<problem_t>::ComputePressure(
    amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> amrex::Real {
  const auto rho = cons(i, j, k, density_index);
  const auto px = cons(i, j, k, x1Momentum_index);
  const auto py = cons(i, j, k, x2Momentum_index);
  const auto pz = cons(i, j, k, x3Momentum_index);
  const auto E =
      cons(i, j, k, energy_index); // *total* gas energy per unit volume
  const auto vx = px / rho;
  const auto vy = py / rho;
  const auto vz = pz / rho;
  const auto kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  const auto thermal_energy = E - kinetic_energy;
  const auto P = thermal_energy * (HydroSystem<problem_t>::gamma_ - 1.0);
  return P;
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::isStateValid(
    amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool {
  // check if cons(i, j, k) is a valid state
  const amrex::Real rho = cons(i, j, k, density_index);
  bool isDensityPositive = (rho > 0.);

  // when the dual energy method is used, we *cannot* reset on pressure
  // failures. on the other hand, we don't need to -- the auxiliary internal
  // energy is used instead!
#if 0
  bool isPressurePositive = false;
  if constexpr (!is_eos_isothermal()) {
    const amrex::Real P = ComputePressure(cons, i, j, k);
    isPressurePositive = (P > 0.);
  } else {
    isPressurePositive = true;
  }
#endif
  // return (isDensityPositive && isPressurePositive);

  return isDensityPositive;
}

template <typename problem_t>
void HydroSystem<problem_t>::PredictStep(
    arrayconst_t &consVarOld, array_t &consVarNew,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
    amrex::Box const &indexRange, const int nvars_in,
    amrex::Array4<int> const &redoFlag) {
  BL_PROFILE("HydroSystem::PredictStep()");

  // By convention, the fluxes are defined on the left edge of each zone,
  // i.e. flux_(i) is the flux *into* zone i through the interface on the
  // left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
  // the interface on the right of zone i.

  int const nvars = nvars_in; // workaround nvcc bug

  auto const dt = dt_in;
  auto const dx = dx_in[0];
  auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
  auto const dy = dx_in[1];
  auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
  auto const dz = dx_in[2];
  auto const x3Flux = fluxArray[2];
#endif

  amrex::ParallelFor(
      indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < nvars; ++n) {
          consVarNew(i, j, k, n) =
              consVarOld(i, j, k, n) +
              (AMREX_D_TERM(
                  (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
                  +(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
                  +(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
        }

        // check if state is valid -- flag for re-do if not
        if (!isStateValid(consVarNew, i, j, k)) {
          redoFlag(i, j, k) = quokka::redoFlag::redo;
        } else {
          redoFlag(i, j, k) = quokka::redoFlag::none;
        }
      });
}

template <typename problem_t>
void HydroSystem<problem_t>::AddFluxesRK2(
    array_t &U_new, arrayconst_t &U0, arrayconst_t &U1,
    std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
    amrex::Box const &indexRange, const int nvars_in,
    amrex::Array4<int> const &redoFlag) {
  BL_PROFILE("HyperbolicSystem::AddFluxesRK2()");

  // By convention, the fluxes are defined on the left edge of each zone,
  // i.e. flux_(i) is the flux *into* zone i through the interface on the
  // left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
  // the interface on the right of zone i.

  int const nvars = nvars_in; // workaround nvcc bug

  auto const dt = dt_in;
  auto const dx = dx_in[0];
  auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
  auto const dy = dx_in[1];
  auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
  auto const dz = dx_in[2];
  auto const x3Flux = fluxArray[2];
#endif

  amrex::ParallelFor(
      indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < nvars; ++n) {
          // RK-SSP2 integrator
          const double U_0 = U0(i, j, k, n);
          const double U_1 = U1(i, j, k, n);

          const double FxU_1 =
              (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
          const double FyU_1 =
              (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
          const double FzU_1 =
              (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif

          // save results in U_new
          U_new(i, j, k, n) =
              (0.5 * U_0 + 0.5 * U_1) +
              (AMREX_D_TERM(0.5 * FxU_1, +0.5 * FyU_1, +0.5 * FzU_1));
        }

        // check if state is valid -- flag for re-do if not
        if (!isStateValid(U_new, i, j, k)) {
          redoFlag(i, j, k) = quokka::redoFlag::redo;
        } else {
          redoFlag(i, j, k) = quokka::redoFlag::none;
        }
      });
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFlatteningCoefficients(
    amrex::Array4<const amrex::Real> const &primVar_in, array_t &x1Chi_in,
    amrex::Box const &indexRange) {
  quokka::Array4View<const amrex::Real, DIR> primVar(primVar_in);
  quokka::Array4View<amrex::Real, DIR> x1Chi(x1Chi_in);

  // compute the PPM shock flattening coefficient following
  //   Appendix B1 of Mignone+ 2005 [this description has typos].
  // Method originally from Miller & Colella,
  //   Journal of Computational Physics 183, 26–82 (2002) [no typos].

  constexpr double beta_max = 0.85;
  constexpr double beta_min = 0.75;
  constexpr double Zmax = 0.75;
  constexpr double Zmin = 0.25;

  // cell-centered kernel
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in,
                                                      int k_in) {
    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

    amrex::Real Pplus2 = primVar(i + 2, j, k, pressure_index);
    amrex::Real Pplus1 = primVar(i + 1, j, k, pressure_index);
    amrex::Real P = primVar(i, j, k, pressure_index);
    amrex::Real Pminus1 = primVar(i - 1, j, k, pressure_index);
    amrex::Real Pminus2 = primVar(i - 2, j, k, pressure_index);

    if constexpr (reconstruct_eint) {
      // compute (rho e) (gamma - 1)
      Pplus2 *= primVar(i + 2, j, k, primDensity_index) * (gamma_ - 1.0);
      Pplus1 *= primVar(i + 1, j, k, primDensity_index) * (gamma_ - 1.0);
      P *= primVar(i, j, k, primDensity_index) * (gamma_ - 1.0);
      Pminus1 *= primVar(i - 1, j, k, primDensity_index) * (gamma_ - 1.0);
      Pminus2 *= primVar(i - 2, j, k, primDensity_index) * (gamma_ - 1.0);
    }

    if constexpr (is_eos_isothermal()) {
      const amrex::Real cs_sq = cs_iso_ * cs_iso_;
      Pplus2 = primVar(i + 2, j, k, primDensity_index) * cs_sq;
      Pplus1 = primVar(i + 1, j, k, primDensity_index) * cs_sq;
      P = primVar(i, j, k, primDensity_index) * cs_sq;
      Pminus1 = primVar(i - 1, j, k, primDensity_index) * cs_sq;
      Pminus2 = primVar(i - 2, j, k, primDensity_index) * cs_sq;
    }

    // beta is a measure of shock resolution (Eq. 74 of Miller & Colella 2002)
    // Miller & Collela note: "If beta is 1/2, then pressure is linear across
    //   four computational cells. If beta is small enough, then we assume that
    //   any discontinuity is already sufficiently well resolved that additional
    //   dissipation (flattening) is not required."
    const double beta_denom = std::abs(Pplus2 - Pminus2);
    // avoid division by zero (in this case, chi = 1 anyway)
    const double beta =
        (beta_denom != 0) ? (std::abs(Pplus1 - Pminus1) / beta_denom) : 0;

    // Eq. 75 of Miller & Colella 2002
    const double chi_min =
        std::max(0., std::min(1., (beta_max - beta) / (beta_max - beta_min)));

    // Z is a measure of shock strength (Eq. 76 of Miller & Colella 2002)
    const double K_S = gamma_ * P; // equal to \rho c_s^2
    const double Z = std::abs(Pplus1 - Pminus1) / K_S;

    // check for converging flow along the normal direction DIR (Eq. 77)
    int velocity_index = 0;
    if constexpr (DIR == FluxDir::X1) {
      velocity_index = x1Velocity_index;
    } else if constexpr (DIR == FluxDir::X2) {
      velocity_index = x2Velocity_index;
    } else if constexpr (DIR == FluxDir::X3) {
      velocity_index = x3Velocity_index;
    }
    double chi = 1.0;
    if (primVar(i + 1, j, k, velocity_index) <
        primVar(i - 1, j, k, velocity_index)) {
      chi = std::max(chi_min, std::min(1., (Zmax - Z) / (Zmax - Zmin)));
    }

    x1Chi(i, j, k) = chi;
  });
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::FlattenShocks(
    amrex::Array4<const amrex::Real> const &q_in,
    amrex::Array4<const amrex::Real> const &x1Chi_in,
    amrex::Array4<const amrex::Real> const &x2Chi_in,
    amrex::Array4<const amrex::Real> const &x3Chi_in, array_t &x1LeftState_in,
    array_t &x1RightState_in, amrex::Box const &indexRange, const int nvars) {
  quokka::Array4View<const amrex::Real, DIR> q(q_in);
  quokka::Array4View<amrex::Real, DIR> x1LeftState(x1LeftState_in);
  quokka::Array4View<amrex::Real, DIR> x1RightState(x1RightState_in);

  // Apply shock flattening based on Miller & Colella (2002)
  // [This is necessary to get a reasonable solution to the slow-moving
  // shock problem, and reduces post-shock oscillations in other cases.]

  // cell-centered kernel
  amrex::ParallelFor(
      indexRange, nvars,
      [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) {
        // compute coefficient as the minimum from adjacent cells along *each
        // axis*
        //  (Eq. 86 of Miller & Colella 2001; Eq. 78 of Miller & Colella 2002)
        double chi_ijk = std::min({
          x1Chi_in(i_in - 1, j_in, k_in), x1Chi_in(i_in, j_in, k_in),
              x1Chi_in(i_in + 1, j_in, k_in),
#if (AMREX_SPACEDIM >= 2)
              x2Chi_in(i_in, j_in - 1, k_in), x2Chi_in(i_in, j_in, k_in),
              x2Chi_in(i_in, j_in + 1, k_in),
#endif
#if (AMREX_SPACEDIM == 3)
              x3Chi_in(i_in, j_in, k_in - 1), x3Chi_in(i_in, j_in, k_in),
              x3Chi_in(i_in, j_in, k_in + 1),
#endif
        });

        auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

        // get interfaces
        const double a_minus = x1RightState(i, j, k, n);
        const double a_plus = x1LeftState(i + 1, j, k, n);
        const double a_mean = q(i, j, k, n);

        // left side of zone i (Eq. 70a)
        const double new_a_minus = chi_ijk * a_minus + (1. - chi_ijk) * a_mean;

        // right side of zone i (Eq. 70b)
        const double new_a_plus = chi_ijk * a_plus + (1. - chi_ijk) * a_mean;

        x1RightState(i, j, k, n) = new_a_minus;
        x1LeftState(i + 1, j, k, n) = new_a_plus;
      });
}

template <typename problem_t>
void HydroSystem<problem_t>::AddInternalEnergyPressureTerm(
    amrex::Array4<amrex::Real> const &consVar,
    amrex::Array4<const amrex::Real> const &primVar,
    amrex::Box const &indexRange,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::Real const dt_in) {
  // first-order pressure term is added separately to the internal energy
  // [See Li et al. (2007) and Schneider & Robertson (2017).]

  amrex::Real const dt = dt_in; // workaround nvcc bug

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    // compute cell-centered pressure from primitive vars
    double P = primVar(i, j, k, pressure_index);
    if constexpr (reconstruct_eint) {
      P *= primVar(i, j, k, primDensity_index) * (gamma_ - 1.0);
    }

    // compute velocity divergence
    const double v_xplus = primVar(i + 1, j, k, x1Velocity_index);
    const double v_xminus = primVar(i - 1, j, k, x1Velocity_index);
#if AMREX_SPACEDIM >= 2
    const double v_yplus = primVar(i, j + 1, k, x2Velocity_index);
    const double v_yminus = primVar(i, j - 1, k, x2Velocity_index);
#endif
#if AMREX_SPACEDIM == 3
    const double v_zplus = primVar(i, j, k + 1, x3Velocity_index);
    const double v_zminus = primVar(i, j, k - 1, x3Velocity_index);
#endif
    amrex::Real const div_v =
        AMREX_D_TERM((v_xminus - v_xplus) / (2.0 * dx[0]),
                     +(v_yminus - v_yplus) / (2.0 * dx[1]),
                     +(v_zminus - v_zplus) / (2.0 * dx[2]));

    // add pressure term
    amrex::Real const Eint_aux =
        consVar(i, j, k, internalEnergy_index) + dt * (P * div_v);

    // replace Eint with Eint_cons == (Etot - Ekin) if (Eint_cons / E) > eta
    amrex::Real const rho = consVar(i, j, k, density_index);
    amrex::Real const px = consVar(i, j, k, x1Momentum_index);
    amrex::Real const py = consVar(i, j, k, x2Momentum_index);
    amrex::Real const pz = consVar(i, j, k, x3Momentum_index);
    amrex::Real const Ekin = 0.5 * (px * px + py * py + pz * pz) / rho;
    amrex::Real const Etot = consVar(i, j, k, energy_index);
    amrex::Real const Eint_cons = Etot - Ekin;

    // eta value from Flash (https://flash-x.github.io/Flash-X-docs/Hydro.html)
    const amrex::Real eta = 1.0e-4; // dual energy parameter 'eta'

    // Li et al. sync method
    if (Eint_cons > eta * Etot) {
      consVar(i, j, k, internalEnergy_index) = Eint_cons;
    } else { // non-conservative sync
      consVar(i, j, k, internalEnergy_index) = Eint_aux;
      consVar(i, j, k, energy_index) = Eint_aux + Ekin;
    }
  });
}

template <typename problem_t>
template <FluxDir DIR>
void HydroSystem<problem_t>::ComputeFluxes(
    array_t &x1Flux_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
    amrex::Array4<const amrex::Real> const &x1RightState_in,
    amrex::Array4<const amrex::Real> const &primVar_in,
    amrex::Box const &indexRange) {

  quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
  quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
  quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);
  quokka::Array4View<const amrex::Real, DIR> q(primVar_in);

  // By convention, the interfaces are defined on the left edge of each
  // zone, i.e. xinterface_(i) is the solution to the Riemann problem at
  // the left edge of zone i.

  // Indexing note: There are (nx + 1) interfaces for nx zones.

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in,
                                                      int k_in) {
    auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

    // HLLC solver following Toro (1998) and Balsara (2017).
    // [Carbuncle correction:
    //  Minoshima & Miyoshi, "A low-dissipation HLLD approximate Riemann solver
    //  	for a very wide range of Mach numbers," JCP (2021).]

    // gather left- and right- state variables

    const double rho_L = x1LeftState(i, j, k, primDensity_index);
    const double rho_R = x1RightState(i, j, k, primDensity_index);

    const double vx_L = x1LeftState(i, j, k, x1Velocity_index);
    const double vx_R = x1RightState(i, j, k, x1Velocity_index);

    const double vy_L = x1LeftState(i, j, k, x2Velocity_index);
    const double vy_R = x1RightState(i, j, k, x2Velocity_index);

    const double vz_L = x1LeftState(i, j, k, x3Velocity_index);
    const double vz_R = x1RightState(i, j, k, x3Velocity_index);

    const double ke_L = 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
    const double ke_R = 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);

    // auxiliary Eint (rho * e)
    // this is evolved as a passive scalar by the Riemann solver
    const double Eint_L = x1LeftState(i, j, k, primEint_index);
    const double Eint_R = x1RightState(i, j, k, primEint_index);

    double P_L = NAN;
    double P_R = NAN;

    double E_L = NAN;
    double E_R = NAN;

    double cs_L = NAN;
    double cs_R = NAN;

    if constexpr (is_eos_isothermal()) {
      P_L = rho_L * (cs_iso_ * cs_iso_);
      P_R = rho_R * (cs_iso_ * cs_iso_);

      cs_L = cs_iso_;
      cs_R = cs_iso_;
    } else {
      if constexpr (reconstruct_eint) { // pressure_index is actually eint
        // compute pressure from specific internal energy
        const double eint_L = x1LeftState(i, j, k, pressure_index);
        const double eint_R = x1RightState(i, j, k, pressure_index);

        P_L = rho_L * eint_L * (gamma_ - 1.0);
        P_R = rho_R * eint_R * (gamma_ - 1.0);
      } else { // pressure_index is actually pressure
        P_L = x1LeftState(i, j, k, pressure_index);
        P_R = x1RightState(i, j, k, pressure_index);
      }

      cs_L = std::sqrt(gamma_ * P_L / rho_L);
      cs_R = std::sqrt(gamma_ * P_R / rho_R);

      E_L = P_L / (gamma_ - 1.0) + ke_L;
      E_R = P_R / (gamma_ - 1.0) + ke_R;
    }

    AMREX_ASSERT(cs_L > 0.0);
    AMREX_ASSERT(cs_R > 0.0);

    // assign normal component of velocity according to DIR

    double u_L = NAN;
    double u_R = NAN;
    int velN_index = x1Velocity_index;
    int velV_index = x2Velocity_index;
    int velW_index = x3Velocity_index;

    if constexpr (DIR == FluxDir::X1) {
      u_L = vx_L;
      u_R = vx_R;
      velN_index = x1Velocity_index;
      velV_index = x2Velocity_index;
      velW_index = x3Velocity_index;
    } else if constexpr (DIR == FluxDir::X2) {
      u_L = vy_L;
      u_R = vy_R;
      if constexpr (AMREX_SPACEDIM == 2) {
        velN_index = x2Velocity_index;
        velV_index = x1Velocity_index;
        velW_index = x3Velocity_index; // unchanged in 2D
      } else if constexpr (AMREX_SPACEDIM == 3) {
        velN_index = x2Velocity_index;
        velV_index = x3Velocity_index;
        velW_index = x1Velocity_index;
      }
    } else if constexpr (DIR == FluxDir::X3) {
      u_L = vz_L;
      u_R = vz_R;
      velN_index = x3Velocity_index;
      velV_index = x1Velocity_index;
      velW_index = x2Velocity_index;
    }

    // compute PVRS states (Toro 10.5.2)

    const double rho_bar = 0.5 * (rho_L + rho_R);
    const double cs_bar = 0.5 * (cs_L + cs_R);
    const double P_PVRS =
        0.5 * (P_L + P_R) - 0.5 * (u_R - u_L) * (rho_bar * cs_bar);
    const double P_star = std::max(P_PVRS, 0.0);

    const double q_L = (P_star <= P_L)
                           ? 1.0
                           : std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
                                                 ((P_star / P_L) - 1.0));

    const double q_R = (P_star <= P_R)
                           ? 1.0
                           : std::sqrt(1.0 + ((gamma_ + 1.0) / (2.0 * gamma_)) *
                                                 ((P_star / P_R) - 1.0));

    // compute wave speeds

    double S_L = u_L - q_L * cs_L;
    double S_R = u_R + q_R * cs_R;

    // carbuncle correction [Eq. 10 of Minoshima & Miyoshi (2021)]
    const double cs_max = std::max(cs_L, cs_R);
    // difference in normal velocity along normal axis
    const double du = q(i, j, k, velN_index) - q(i - 1, j, k, velN_index);
    // difference in transverse velocity
#if AMREX_SPACEDIM == 1
    const double dw = 0.;
#else
    amrex::Real dvl = std::min(q(i - 1, j + 1, k, velV_index) - q(i - 1, j, k, velV_index),
                 q(i - 1, j, k, velV_index) - q(i - 1, j - 1, k, velV_index));
    amrex::Real dvr = std::min(q(i, j + 1, k, velV_index) - q(i, j, k, velV_index),
                 q(i, j, k, velV_index) - q(i, j - 1, k, velV_index));
    double dw = std::min(dvl, dvr);
#endif
#if AMREX_SPACEDIM == 3
    amrex::Real dwl =
        std::min(q(i - 1, j, k + 1, velW_index) - q(i - 1, j, k, velW_index),
                 q(i - 1, j, k, velW_index) - q(i - 1, j, k - 1, velW_index));
    amrex::Real dwr =
        std::min(q(i, j, k + 1, velW_index) - q(i, j, k, velW_index),
                 q(i, j, k, velW_index) - q(i, j, k - 1, velW_index));
    dw = std::min(std::min(dwl, dwr), dw);
#endif
    const double tp =
        std::min(1., (cs_max - std::min(du, 0.)) / (cs_max - std::min(dw, 0.)));
    const double theta = tp * tp * tp * tp;

    const double S_star = (theta * (P_R - P_L) + (rho_L * u_L * (S_L - u_L) -
                                                  rho_R * u_R * (S_R - u_R))) /
                          (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));

    // Low-dissipation pressure correction 'phi' [Eq. 23 of Minoshima & Miyoshi]
    const double vmag_L = std::sqrt(vx_L * vx_L + vy_L * vy_L + vz_L * vz_L);
    const double vmag_R = std::sqrt(vx_R * vx_R + vy_R * vy_R + vz_R * vz_R);
    const double chi = std::min(1., std::max(vmag_L, vmag_R) / cs_max);
    const double phi = chi * (2. - chi);

    const double P_LR =
        0.5 * (P_L + P_R) + 0.5 * phi *
                                (rho_L * (S_L - u_L) * (S_star - u_L) +
                                 rho_R * (S_R - u_R) * (S_star - u_R));

    /// compute fluxes

    constexpr int fluxdim = nvar_; // including passive scalar components

    // initialize all components to zero
    quokka::valarray<double, fluxdim> D_L(0.);
    quokka::valarray<double, fluxdim> D_R(0.);
    quokka::valarray<double, fluxdim> D_star(0.);

    // N.B.: quokka::valarray is written to allow assigning <= fluxdim
    // components, so this works even if there are more components than
    // enumerated in the initializer list
    if constexpr (DIR == FluxDir::X1) {
      D_L = {0., 1., 0., 0., u_L, 0.};
      D_R = {0., 1., 0., 0., u_R, 0.};
      D_star = {0., 1., 0., 0., S_star, 0.};
    } else if constexpr (DIR == FluxDir::X2) {
      D_L = {0., 0., 1., 0., u_L, 0.};
      D_R = {0., 0., 1., 0., u_R, 0.};
      D_star = {0., 0., 1., 0., S_star, 0.};
    } else if constexpr (DIR == FluxDir::X3) {
      D_L = {0., 0., 0., 1., u_L, 0.};
      D_R = {0., 0., 0., 1., u_R, 0.};
      D_star = {0., 0., 0., 1., S_star, 0.};
    }

    const std::initializer_list<double> state_L = {
        rho_L, rho_L * vx_L, rho_L * vy_L, rho_L * vz_L, E_L, Eint_L};

    const std::initializer_list<double> state_R = {
        rho_R, rho_R * vx_R, rho_R * vy_R, rho_R * vz_R, E_R, Eint_R};

    AMREX_ASSERT(state_L.size() == state_R.size());

    // N.B.: quokka::valarray is written to allow assigning <= fluxdim
    // components, so this works even if there are more components than
    // enumerated in the initializer list
    quokka::valarray<double, fluxdim> U_L = state_L;
    quokka::valarray<double, fluxdim> U_R = state_R;

    // The remaining components are passive scalars, so just copy them from
    // x1LeftState and x1RightState into the (left, right) state vectors U_L and
    // U_R
    for (int nc = static_cast<int>(state_L.size()); nc < fluxdim; ++nc) {
      U_L[nc] = x1LeftState(i, j, k, nc);
      U_R[nc] = x1RightState(i, j, k, nc);
    }

    const quokka::valarray<double, fluxdim> F_L = u_L * U_L + P_L * D_L;
    const quokka::valarray<double, fluxdim> F_R = u_R * U_R + P_R * D_R;

    const quokka::valarray<double, fluxdim> F_starL =
        (S_star * (S_L * U_L - F_L) + S_L * P_LR * D_star) / (S_L - S_star);

    const quokka::valarray<double, fluxdim> F_starR =
        (S_star * (S_R * U_R - F_R) + S_R * P_LR * D_star) / (S_R - S_star);

    // open the Riemann fan
    quokka::valarray<double, fluxdim> F{};

    // HLLC flux
    if (S_L > 0.0) {
      F = F_L;
    } else if ((S_star > 0.0) && (S_L <= 0.0)) {
      F = F_starL;
    } else if ((S_star <= 0.0) && (S_R >= 0.0)) {
      F = F_starR;
    } else { // S_R < 0.0
      F = F_R;
    }

    // set energy fluxes to zero if EOS is isothermal
    if constexpr (is_eos_isothermal()) {
      F[energy_index] = 0;
      F[internalEnergy_index] = 0;
    }

    // copy all flux components to the flux array
    for (int nc = 0; nc < fluxdim; ++nc) {
      AMREX_ASSERT(!std::isnan(F[nc])); // check flux is valid
      x1Flux(i, j, k, nc) = F[nc];
    }
  });
}

#endif // HYDRO_SYSTEM_HPP_
