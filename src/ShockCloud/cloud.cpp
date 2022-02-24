//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///
#include <random>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "cloud.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

using amrex::Real;

struct ShockCloud {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = hydrogen_mass_cgs_;
constexpr double seconds_in_year = 3.154e7;

template <> struct EOS_Traits<ShockCloud> {
  static constexpr double gamma = 5. / 3.; // default value
  // if true, reconstruct e_int instead of pressure
  static constexpr bool reconstruct_eint = true;
};

constexpr Real Tgas0 = 1.0e7;            // K
constexpr Real nH0 = 1.0e-4;             // cm^-3
constexpr Real nH1 = 1.0e-1;             // cm^-3
constexpr Real R_cloud = 5.0 * 3.086e18; // cm [5 pc]
constexpr Real M0 = 2.0;                 // Mach number of shock

constexpr Real T_floor = 10.0;                             // K
constexpr Real P0 = nH0 * Tgas0 * boltzmann_constant_cgs_; // erg cm^-3
constexpr Real rho0 = nH0 * m_H;                           // g cm^-3
constexpr Real rho1 = nH1 * m_H;

template <>
void RadhydroSimulation<ShockCloud>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();

  Real const Lx = (prob_hi[0] - prob_lo[0]);
  Real const Ly = (prob_hi[1] - prob_lo[1]);
  Real const Lz = (prob_hi[2] - prob_lo[2]);

  Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  Real const y0 = prob_lo[1] + 0.8 * (prob_hi[1] - prob_lo[1]);
  Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

  // perturbation parameters
  const int kmin = 0;
  const int kmax = 16;
  Real const A = 0.05 / kmax;

  // generate random phases
  amrex::Array<int, AMREX_SPACEDIM> tlo{AMREX_D_DECL(kmin, kmin, kmin)};
  amrex::Array<int, AMREX_SPACEDIM> thi{AMREX_D_DECL(kmax, kmax, kmax)};
  amrex::TableData<Real, AMREX_SPACEDIM> table_data(tlo, thi);
#ifdef AMREX_USE_GPU
  amrex::TableData<Real, AMREX_SPACEDIM> h_table_data(
      tlo, thi, amrex::The_Pinned_Arena());
  auto const &h_table = h_table_data.table();
#else
  auto const &h_table = table_data.table();
#endif
  // 64-bit Mersenne Twister (do not use 32-bit version for sampling doubles!)
  std::mt19937_64 rng(1); // NOLINT
  std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);

  // Initialize data on the host
  for (int j = tlo[0]; j <= thi[0]; ++j) {
    for (int i = tlo[1]; i <= thi[1]; ++i) {
      for (int k = tlo[2]; k <= thi[2]; ++k) {
        h_table(i, j, k) = sample_phase(rng);
      }
    }
  }
#ifdef AMREX_USE_GPU
  // Copy data to GPU memory
  table_data.copy(h_table_data);
  amrex::Gpu::streamSynchronize();
#endif
  auto const &phase = table_data.const_table(); // const makes it read only

  // cooling tables
  auto tables = cloudyTables.const_tables();

  for (amrex::MFIter iter(state_old_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
      Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
      Real const z = prob_lo[2] + (k + Real(0.5)) * dx[2];
      Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
                               std::pow(z - z0, 2));

      state(i, j, k, RadSystem<ShockCloud>::radEnergy_index) = 0;
      state(i, j, k, RadSystem<ShockCloud>::x1RadFlux_index) = 0;
      state(i, j, k, RadSystem<ShockCloud>::x2RadFlux_index) = 0;
      state(i, j, k, RadSystem<ShockCloud>::x3RadFlux_index) = 0;

      // compute perturbations
      Real delta_rho = 0;
      for (int ki = kmin; ki < kmax; ++ki) {
        for (int kj = kmin; kj < kmax; ++kj) {
          for (int kk = kmin; kk < kmax; ++kk) {
            if ((ki == 0) && (kj == 0) && (kk == 0)) {
              continue;
            }
            Real const kx = 2.0 * M_PI * Real(ki) / Lx;
            Real const ky = 2.0 * M_PI * Real(kj) / Lx;
            Real const kz = 2.0 * M_PI * Real(kk) / Lx;
            delta_rho +=
                A * std::sin(x * kx + y * ky + z * kz + phase(ki, kj, kk));
          }
        }
      }
      AMREX_ALWAYS_ASSERT(delta_rho > -1.0);

      Real rho = rho0 * (1.0 + delta_rho); // background density
      if (R < R_cloud) {
        rho = rho1 * (1.0 + delta_rho); // cloud density
      }
      Real const xmom = 0;
      Real const ymom = 0;
      Real const zmom = 0;
      Real const Eint = (HydroSystem<ShockCloud>::gamma_ - 1.) * P0;
      Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(
          rho, xmom, ymom, zmom, Eint);

      state(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
      state(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
      state(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
      state(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
      state(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShockCloud>::setCustomBoundaryConditions(
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

  if (j >= hi[1]) {
    // x2 upper boundary -- constant
    // compute downstream shock conditions from rho0, P0, and M0
    constexpr Real gamma = HydroSystem<ShockCloud>::gamma_;
    constexpr Real rho2 =
        rho0 * (gamma + 1.) * M0 * M0 / ((gamma - 1.) * M0 * M0 + 2.);
    constexpr Real P2 = P0 * (2. * gamma * M0 * M0 - (gamma - 1.)) / (gamma + 1.);
    Real const v2 = -M0 * std::sqrt(gamma * P2 / rho2);

    Real const rho = rho2;
    Real const xmom = 0;
    Real const ymom = rho2 * v2;
    Real const zmom = 0;
    Real const Eint = (gamma - 1.) * P2;
    Real const Egas =
        RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

    consVar(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
    consVar(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
    consVar(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
    consVar(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
  }
}

struct ODEUserData {
  amrex::Real rho;
  cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  Real rho = udata->rho;
  cloudyGpuConstTables &tables = udata->tables;

  // compute temperature (implicit solve, depends on composition)
  Real Eint = y_data[0];
  Real T =
      ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);

  // compute cooling function
  y_rhs[0] = cloudy_cooling_function(rho, T, tables);
  return 0;
}

void computeCooling(amrex::MultiFab &mf, const Real dt_in,
                    cloudy_tables &cloudyTables) {
  BL_PROFILE("RadhydroSimulation::computeCooling()")

  const Real dt = dt_in;
  const Real reltol_floor = 0.01;
  const Real rtol = 1.0e-4; // not recommended to change this

  auto tables = cloudyTables.const_tables();

  // loop over all cells in MultiFab mf
  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {
      const Real rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      const Real x1Mom =
          state(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
      const Real x2Mom =
          state(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
      const Real x3Mom =
          state(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
      const Real Egas = state(i, j, k, HydroSystem<ShockCloud>::energy_index);

      Real Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom,
                                                             x3Mom, Egas);

      ODEUserData user_data{rho, tables};
      quokka::valarray<Real, 1> y = {Eint};
      quokka::valarray<Real, 1> abstol = {
          reltol_floor * ComputeEgasFromTgas(rho, T_floor,
                                             HydroSystem<ShockCloud>::gamma_,
                                             tables)};

      // do integration with RK2 (Heun's method)
      rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol);

      const Real Egas_new = RadSystem<ShockCloud>::ComputeEgasFromEint(
          rho, x1Mom, x2Mom, x3Mom, y[0]);

      state(i, j, k, HydroSystem<ShockCloud>::energy_index) = Egas_new;
    });
  }
}

template <>
void RadhydroSimulation<ShockCloud>::computeAfterLevelAdvance(
    int lev, amrex::Real /*time*/, amrex::Real dt_lev, int /*iteration*/,
    int /*ncycle*/) {
  // compute operator split physics
  computeCooling(state_new_[lev], dt_lev, cloudyTables);
}

template <>
void HydroSystem<ShockCloud>::EnforcePressureFloor(
    amrex::Real const densityFloor, amrex::Real const /*pressureFloor*/,
    amrex::Box const &indexRange, amrex::Array4<amrex::Real> const &state) {
  // prevent vacuum creation
  amrex::Real const rho_floor = densityFloor; // workaround nvcc bug

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

        amrex::Real const P_floor =
            (rho_new / m_H) * boltzmann_constant_cgs_ * T_floor;

        if (!is_eos_isothermal()) {
          // recompute gas energy (to prevent P < 0)
          amrex::Real const Eint_star = Etot - 0.5 * rho_new * vsq;
          amrex::Real const P_star = Eint_star * (gamma_ - 1.);
          amrex::Real P_new = P_star;
          if (P_star < P_floor) {
            P_new = P_floor;
            amrex::Real const Etot_new =
                P_new / (gamma_ - 1.) + 0.5 * rho_new * vsq;
            state(i, j, k, energy_index) = Etot_new;
          }
        }
      });
}

auto problem_main() -> int {
  // Problem parameters
  const double CFL_number = 0.25;
  const double max_time = 2.0e6 * seconds_in_year; // 2 Myr
  const int max_timesteps = 1e5;

  // Problem initialization
  constexpr int nvars = RadhydroSimulation<ShockCloud>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[n].setLo(0, amrex::BCType::int_dir); // periodic
    boundaryConditions[n].setHi(0, amrex::BCType::int_dir);
    boundaryConditions[n].setLo(1, amrex::BCType::foextrap); // extrapolate
    boundaryConditions[n].setHi(1, amrex::BCType::ext_dir);  // Dirichlet
#if AMREX_SPACEDIM == 3
    boundaryConditions[n].setLo(2, amrex::BCType::int_dir); // periodic
    boundaryConditions[n].setHi(2, amrex::BCType::int_dir);
#endif
  }

  RadhydroSimulation<ShockCloud> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;

  // Standard PPM gives unphysically enormous temperatures when used for
  // this problem (e.g., ~1e14 K or higher), but can be fixed by
  // reconstructing the temperature instead of the pressure
  sim.reconstructionOrder_ = 3; // PLM

  sim.cflNumber_ = CFL_number;
  sim.maxTimesteps_ = max_timesteps;
  sim.stopTime_ = max_time;
  sim.plotfileInterval_ = 100;
  sim.checkpointInterval_ = 2000;

  // Read Cloudy tables
  readCloudyData(sim.cloudyTables);

  // Set initial conditions
  sim.setInitialConditions();

  // run simulation
  sim.evolve();

  // Cleanup and exit
  int status = 0;
  return status;
}
