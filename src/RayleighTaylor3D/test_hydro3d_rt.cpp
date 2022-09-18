//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include <cmath>

#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "test_hydro3d_rt.hpp"

struct RTProblem {};

template <> struct HydroSystem_Traits<RTProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<RTProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;
  
  static constexpr int numPassiveScalars = 1; // number of passive scalars
};

amrex::Real constexpr g_x = 0;
amrex::Real constexpr g_y = 0;
amrex::Real constexpr g_z = -0.1;

template <>
void RadhydroSimulation<RTProblem>::setInitialConditionsOnGrid(
    std::vector<quokka::grid> &grid_vec) {
  // extract variables required from the geom object
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_vec[0].dx;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_vec[0].prob_lo;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_vec[0].prob_hi;
  const amrex::Box &indexRange = grid_vec[0].indexRange;
  const amrex::Array4<double>& state_cc = grid_vec[0].array;

  amrex::Real const A = 0.01;

  // loop over the grid and set the initial condition
  amrex::ParallelForRNG(
      indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &rng) noexcept {
        amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
        amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
        amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

        double rho = NAN;
        double scalar = NAN;
        if (z > 0.0) {
          rho = 2.0;
          scalar = 1.0;
        } else {
          rho = 1.0;
          scalar = 0.0;
        }

        double amp = A * amrex::Random(rng);

        double vx = 0;
        double vy = 0;
        double vz = amp * (1.0 + std::cos(8.0*M_PI*z/3.0))/2.0;
        double P0 = 2.5;
        double P = P0 + rho * g_z * z;

        AMREX_ASSERT(!std::isnan(vx));
        AMREX_ASSERT(!std::isnan(vy));
        AMREX_ASSERT(!std::isnan(vz));
        AMREX_ASSERT(!std::isnan(rho));
        AMREX_ASSERT(!std::isnan(P));

        const auto v_sq = vx * vx + vy * vy + vz * vz;
        const auto gamma = HydroSystem<RTProblem>::gamma_;

        state_cc(i, j, k, HydroSystem<RTProblem>::density_index) = rho;
        state_cc(i, j, k, HydroSystem<RTProblem>::x1Momentum_index) =
            rho * vx;
        state_cc(i, j, k, HydroSystem<RTProblem>::x2Momentum_index) =
            rho * vy;
        state_cc(i, j, k, HydroSystem<RTProblem>::x3Momentum_index) =
            rho * vz;
        state_cc(i, j, k, HydroSystem<RTProblem>::energy_index) =
            P / (gamma - 1.) + 0.5 * rho * v_sq;
        state_cc(i, j, k, HydroSystem<RTProblem>::internalEnergy_index) = 
            P / (gamma - 1.);
        state_cc(i, j, k, HydroSystem<RTProblem>::scalar0_index) = scalar;
      });
}

template <>
void RadhydroSimulation<RTProblem>::addStrangSplitSources(amrex::MultiFab &state_mf,
    const int lev, const amrex::Real time, const amrex::Real dt) {
  // add gravitational source terms
  const auto state = state_mf.arrays();

  amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
      // save initial KE
      amrex::Real const rho = state[bx](i, j, k, HydroSystem<RTProblem>::density_index);
      amrex::Real px = state[bx](i, j, k, HydroSystem<RTProblem>::x1Momentum_index);
      amrex::Real py = state[bx](i, j, k, HydroSystem<RTProblem>::x2Momentum_index);
      amrex::Real pz = state[bx](i, j, k, HydroSystem<RTProblem>::x3Momentum_index);
      amrex::Real const KE_init = (px*px + py*py + pz*pz) / (2.0 * rho);

      // add body forces
      px += dt * rho * g_x;
      py += dt * rho * g_y;
      pz += dt * rho * g_z;

      // compute new KE
      amrex::Real const KE_final = (px*px + py*py + pz*pz) / (2.0 * rho);
      amrex::Real const dKE = KE_final - KE_init;

      // update variables
      state[bx](i, j, k, HydroSystem<RTProblem>::x1Momentum_index) = px;
      state[bx](i, j, k, HydroSystem<RTProblem>::x2Momentum_index) = py;
      state[bx](i, j, k, HydroSystem<RTProblem>::x3Momentum_index) = pz;
      state[bx](i, j, k, HydroSystem<RTProblem>::energy_index) += dKE;
  });
  amrex::Gpu::streamSynchronize();
}

template <>
void RadhydroSimulation<RTProblem>::ErrorEst(
    int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/) {
  // tag cells for refinement

  const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
  const amrex::Real rho_min = 0.1;       // minimum density for refinement

  const auto state = state_new_[lev].const_arrays();
  const auto tag = tags.arrays();

  amrex::ParallelFor(state_new_[lev],
    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
      const int n = HydroSystem<RTProblem>::density_index;
      amrex::Real const rho = state[bx](i, j, k, n);
      amrex::Real const rho_xplus = state[bx](i + 1, j, k, n);
      amrex::Real const rho_xminus = state[bx](i - 1, j, k, n);
      amrex::Real const rho_yplus = state[bx](i, j + 1, k, n);
      amrex::Real const rho_yminus = state[bx](i, j - 1, k, n);
      amrex::Real const rho_zplus = state[bx](i, j, k + 1, n);
      amrex::Real const rho_zminus = state[bx](i, j, k - 1, n);

      amrex::Real const del_x = 0.5 * (rho_xplus - rho_xminus);
      amrex::Real const del_y = 0.5 * (rho_yplus - rho_yminus);
      amrex::Real const del_z = 0.5 * (rho_zplus - rho_zminus);

      amrex::Real const gradient_indicator =
          std::sqrt(del_x*del_x + del_y*del_y + del_z*del_z) / rho;

      if ((gradient_indicator > eta_threshold) && (rho > rho_min)) {
        tag[bx](i, j, k) = amrex::TagBox::SET;
      }
  });
  amrex::Gpu::streamSynchronize();
}

template <>
void RadhydroSimulation<RTProblem>::computeAfterTimestep() {
  // compute 1D mixing profile, save to text file
  static amrex::Long cycle = 0;

  if ((cycle % 10) == 0) {
    int axis = 2; // z-axis
    auto profile = computeAxisAlignedProfile(axis,
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const & state) {
        return state(i, j, k, HydroSystem<RTProblem>::scalar0_index);
    });

    // save profile to text file
    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ofstream file;
      file.open("profile.txt");
      file.precision(17);
      
      for(int i = 0; i < profile.size(); ++i) {
        file << profile[i] << "\n";
      }
      file.close();
    }
  }
  
  cycle++;
}

auto problem_main() -> int {
  // Set boundary conditions
  	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<RTProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<RTProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<RTProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

  const int nvars = RadhydroSimulation<RTProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    // periodic in x- and y-directions
    for (int i = 0; i < (AMREX_SPACEDIM-1); ++i) {
      boundaryConditions[n].setLo(i, amrex::BCType::int_dir);
      boundaryConditions[n].setHi(i, amrex::BCType::int_dir);
    }

    // reflecting in z- direction
    int i = AMREX_SPACEDIM - 1;
		if (isNormalComp(n, i)) {
			boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
			boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
		} else {
			boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
			boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
		}
  }

  // Problem initialization
  RadhydroSimulation<RTProblem> sim(boundaryConditions);
  
  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  return 0;
}
