//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///

#include <random>
#include <variant>
#include <vector>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MFParallelFor.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

#include "cloud.hpp"

using amrex::Real;
using namespace amrex::literals;

struct ShockCloud {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr static bool enableRadiation = false;

constexpr double parsec_in_cm = 3.086e18; // cm == 1 pc
constexpr double solarmass_in_g = 1.99e33; // g == 1 Msun
constexpr double m_H = hydrogen_mass_cgs_; // mass of hydrogen atom

// [Habing FUV field, see Eq. 12.6 of Draine, Physics of the ISM/IGM.]
constexpr static Real G_0 = 5.29e-14; // erg cm^-3

template <> struct HydroSystem_Traits<ShockCloud> {
  static constexpr double gamma = 5. / 3.; // default value
  // if true, reconstruct e_int instead of pressure
  static constexpr bool reconstruct_eint = true;
  static constexpr int nscalars = 1; // number of passive scalars
};

template <> struct RadSystem_Traits<ShockCloud> {
  static constexpr double c_light = c_light_cgs_;
  static constexpr double c_hat = 0.1 * c_light_cgs_;
  static constexpr double radiation_constant = radiation_constant_cgs_;
  static constexpr double mean_molecular_mass = hydrogen_mass_cgs_; // unused
  static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
  static constexpr double gamma = 5. / 3.;
  static constexpr double Erad_floor = 0.;
  static constexpr bool compute_v_over_c_terms = true;
};

// temperature floor
AMREX_GPU_MANAGED static Real T_floor = NAN;

// Cloud parameters (set inside problem_main())
AMREX_GPU_MANAGED static Real rho0 = NAN;
AMREX_GPU_MANAGED static Real rho1 = NAN;
AMREX_GPU_MANAGED static Real P0 = NAN;
AMREX_GPU_MANAGED static Real R_cloud = NAN;

// FUV radiation field parameters
AMREX_GPU_MANAGED static Real Erad0 = NAN;
AMREX_GPU_MANAGED static Real Erad_bdry = NAN;

// cloud-tracking variables needed for Dirichlet boundary condition
AMREX_GPU_MANAGED static Real rho_wind = 0;
AMREX_GPU_MANAGED static Real v_wind = 0;
AMREX_GPU_MANAGED static Real P_wind = 0;
AMREX_GPU_MANAGED static Real delta_vx = 0;

static bool enable_cooling = true;

template <>
void RadhydroSimulation<ShockCloud>::setInitialConditionsAtLevel(int lev) {
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
  amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();

  Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
  Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
  Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

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

      // compute perturbations
      Real delta_rho = 0;

      // compute cloud properties
      Real rho = rho0 * (1.0 + delta_rho); // background density
      Real C = 0.0; // concentration is zero on the background
      if (R < R_cloud) {
        rho = rho1 * (1.0 + delta_rho); // cloud density
        C = 1.0; // concentration is unity inside the cloud
      }

      Real const xmom = 0;
      Real const ymom = 0;
      Real const zmom = 0;
      Real const Eint = P0 / (HydroSystem<ShockCloud>::gamma_ - 1.);
      Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(
          rho, xmom, ymom, zmom, Eint);

      state(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
      state(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
      state(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
      state(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
      state(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
      state(i, j, k, RadSystem<ShockCloud>::gasInternalEnergy_index) = Eint;
      state(i, j, k, RadSystem<ShockCloud>::scalar0_index) = C;

      if (::enableRadiation) {
        state(i, j, k, RadSystem<ShockCloud>::radEnergy_index) = Erad0;
        state(i, j, k, RadSystem<ShockCloud>::x1RadFlux_index) = 0;
        state(i, j, k, RadSystem<ShockCloud>::x2RadFlux_index) = 0;
        state(i, j, k, RadSystem<ShockCloud>::x3RadFlux_index) = 0;
      }
    });
  }

  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShockCloud>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/,
    int /*numcomp*/, amrex::GeometryData const &geom, const Real /*time*/,
    const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/) {
  auto [i, j, k] = iv.toArray();

  amrex::Box const &box = geom.Domain();
  const auto &domain_lo = box.loVect();
  const int ilo = domain_lo[0];

  const Real delta_vx = ::delta_vx;
  const Real v_wind = ::v_wind;
  const Real rho_wind = ::rho_wind;
  const Real P_wind = ::P_wind;

  if (i < ilo) {
    // x1 lower boundary -- constant
    Real const vx = v_wind - delta_vx;
    Real const rho = rho_wind;
    Real const xmom = rho_wind * vx;
    Real const ymom = 0;
    Real const zmom = 0;
    Real const Eint = P_wind / (HydroSystem<ShockCloud>::gamma_ - 1.);
    Real const Egas =
        RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

    consVar(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
    consVar(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
    consVar(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
    consVar(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
    consVar(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
    consVar(i, j, k, RadSystem<ShockCloud>::gasInternalEnergy_index) = Eint;
    consVar(i, j, k, RadSystem<ShockCloud>::scalar0_index) = 0;

    // radiation boundary condition -- streaming
    constexpr double c = c_light_cgs_;
    const double E_inc = Erad_bdry;
    const double F_inc = c * E_inc; // streaming incident flux

    if (::enableRadiation) {
      consVar(i, j, k, RadSystem<ShockCloud>::radEnergy_index) = E_inc;
      consVar(i, j, k, RadSystem<ShockCloud>::x1RadFlux_index) = F_inc;
      consVar(i, j, k, RadSystem<ShockCloud>::x2RadFlux_index) = 0;
      consVar(i, j, k, RadSystem<ShockCloud>::x3RadFlux_index) = 0;
    }
  }
}

struct ODEUserData {
  Real rho;
  cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
         quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
  // unpack user_data
  auto *udata = static_cast<ODEUserData *>(user_data);
  const Real rho = udata->rho;
  const Real gamma = HydroSystem<ShockCloud>::gamma_;
  cloudyGpuConstTables &tables = udata->tables;

  // check whether temperature is out-of-bounds
  const Real Tmin = 10.;
  const Real Tmax = 1.0e9;
  const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
  const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

  // compute temperature and cooling rate
  const Real Eint = y_data[0];

  if (Eint <= Eint_min) {
    // set cooling to value at Tmin
    y_rhs[0] = cloudy_cooling_function(rho, Tmin, tables);
  } else if (Eint >= Eint_max) {
    // set cooling to value at Tmax
    y_rhs[0] = cloudy_cooling_function(rho, Tmax, tables);
  } else {
    // ok, within tabulated cooling limits
    const Real T = ComputeTgasFromEgas(rho, Eint, gamma, tables);
    if (!std::isnan(T)) { // temp iteration succeeded
      y_rhs[0] = cloudy_cooling_function(rho, T, tables);
    } else { // temp iteration failed
      y_rhs[0] = NAN;
      return 1; // failed
    }
  }

  return 0; // success
}

void computeCooling(amrex::MultiFab &mf, const Real dt_in,
                    cloudy_tables &cloudyTables) {
  BL_PROFILE("RadhydroSimulation::computeCooling()")

  const Real dt = dt_in;
  const Real reltol_floor = 0.01;
  const Real rtol = 1.0e-4; // not recommended to change this

  auto tables = cloudyTables.const_tables();
  auto const &state = mf.arrays();

  amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j,
                                                        int k) noexcept {
      const Real rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
      const Real x1Mom =
          state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
      const Real x2Mom =
          state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
      const Real x3Mom =
          state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
      const Real Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);

      const Real Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(
          rho, x1Mom, x2Mom, x3Mom, Egas);

      if (!(Eint > 0.)) {
	printf("rho = %.17e, Eint = %.17e, dt = %.17e\n",
	       rho, Eint, dt);
	amrex::Abort("non-positive internal energy!!");
      }
      
      ODEUserData user_data{rho, tables};
      quokka::valarray<Real, 1> y = {Eint};
      quokka::valarray<Real, 1> abstol = {
          reltol_floor * ComputeEgasFromTgas(rho, T_floor,
                                             HydroSystem<ShockCloud>::gamma_,
                                             tables)};
      
      // do integration with RK2 (Heun's method)
      int nsteps = 0;
      rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol,
                            nsteps);

      // check if integration failed
      if (nsteps >= maxStepsODEIntegrate) {
        Real T = ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_,
                                     tables);
        Real Edot = cloudy_cooling_function(rho, T, tables);
        Real t_cool = Eint / Edot;
        printf(
            "max substeps exceeded! rho = %.17e, Eint = %.17e, T = %g, cooling "
            "time = %g, dt = %.17e\n",
            rho, Eint, T, t_cool, dt);
        amrex::Abort("Max steps exceeded in cooling solve!");
      }
      const Real Eint_new = y[0];
      const Real dEint = Eint_new - Eint;

      state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index) += dEint;
      state[bx](i, j, k, HydroSystem<ShockCloud>::internalEnergy_index) += dEint;
  });
  amrex::Gpu::streamSynchronize();
}

template <>
void RadhydroSimulation<ShockCloud>::addStrangSplitSources(amrex::MultiFab &state, int const lev,
    amrex::Real time, amrex::Real const dt) {
  // compute operator split physics
  if (::enable_cooling) {
    computeCooling(state, dt, cloudyTables);
  }
}

template <>
void RadhydroSimulation<ShockCloud>::computeAfterTimestep(
    const amrex::Real dt_coarse) {
  // perform Galilean transformation (velocity shift to center-of-mass frame)

  // N.B. must weight by passive scalar of cloud, since the background has
  // non-negligible momentum!
  int nc = 1; // number of components in temporary MF
  int ng = 0; // number of ghost cells in temporary MF
  amrex::MultiFab temp_mf(boxArray(0), DistributionMap(0), nc, ng);

  // compute x-momentum
  amrex::MultiFab::Copy(temp_mf, state_new_[0],
                        HydroSystem<ShockCloud>::x1Momentum_index, 0, nc, ng);
  amrex::MultiFab::Multiply(temp_mf, state_new_[0],
                            HydroSystem<ShockCloud>::scalar0_index, 0, nc, ng);
  const Real xmom = temp_mf.sum(0);

  // compute cloud mass within simulation box
  amrex::MultiFab::Copy(temp_mf, state_new_[0],
                        HydroSystem<ShockCloud>::density_index, 0, nc, ng);
  amrex::MultiFab::Multiply(temp_mf, state_new_[0],
                            HydroSystem<ShockCloud>::scalar0_index, 0, nc, ng);
  const Real cloud_mass = temp_mf.sum(0);

  // compute center-of-mass velocity of the cloud
  const Real vx_cm = xmom / cloud_mass;

  // save cumulative position, velocity offsets in simulationMetadata_
  const Real delta_x_prev = std::get<Real>(simulationMetadata_["delta_x"]);
  const Real delta_vx_prev = std::get<Real>(simulationMetadata_["delta_vx"]);
  const Real delta_x = delta_x_prev + dt_coarse * delta_vx_prev;
  const Real delta_vx = delta_vx_prev + vx_cm;
  simulationMetadata_["delta_x"] = delta_x;
  simulationMetadata_["delta_vx"] = delta_vx;
  ::delta_vx = delta_vx;

  amrex::Print() << "\tDelta x = " << (delta_x / parsec_in_cm) << " pc,"
                 << " Delta vx = " << (delta_vx / 1.0e5) << " km/s\n";

  // subtract center-of-mass y-velocity on each level
  // N.B. must update both y-momentum *and* energy!
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const &mf = state_new_[lev];
    auto const &state = state_new_[lev].arrays();
    amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int box, int i, int j,
                                                int k) noexcept {
      Real rho = state[box](i, j, k, HydroSystem<ShockCloud>::density_index);
      Real xmom =
          state[box](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
      Real ymom =
          state[box](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
      Real zmom =
          state[box](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
      Real E = state[box](i, j, k, HydroSystem<ShockCloud>::energy_index);
      Real KE = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
      Real Eint = E - KE;
      Real new_xmom = xmom - rho * vx_cm;
      Real new_KE =
          0.5 * (new_xmom * new_xmom + ymom * ymom + zmom * zmom) / rho;

      state[box](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index) = new_xmom;
      state[box](i, j, k, HydroSystem<ShockCloud>::energy_index) =
          Eint + new_KE;
    });
  }
  amrex::Gpu::streamSynchronize();
}

template <>
void RadhydroSimulation<ShockCloud>::ComputeDerivedVar(
    int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in) const {
  // compute derived variables and save in 'mf'

  if (dname == "log_temperature") {
    const int ncomp = ncomp_in;
    auto tables = cloudyTables.const_tables();
    auto const &output = mf.arrays();
    auto const &state = state_new_[lev].const_arrays();

    amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j,
                                                          int k) noexcept {
      Real rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
      Real x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
      Real x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
      Real x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
      Real Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
      Real Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(
          rho, x1Mom, x2Mom, x3Mom, Egas);
      Real Tgas = ComputeTgasFromEgas(
          rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
      output[bx](i, j, k, ncomp) = std::log10(Tgas);
    });

  } else if (dname == "log_nH") {
    const int ncomp = ncomp_in;
    auto const &output = mf.arrays();
    auto const &state = state_new_[lev].const_arrays();

    amrex::ParallelFor(
      mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
        Real rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
        Real nH = (cloudy_H_mass_fraction * rho) / m_H;
        output[bx](i, j, k, ncomp) = std::log10(nH);
    });

  } else if (dname == "log_cooling_length") {
    const int ncomp = ncomp_in;
    auto tables = cloudyTables.const_tables();
    auto const &output = mf.arrays();
    auto const &state = state_new_[lev].const_arrays();

    amrex::ParallelFor(
      mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
        // compute cooling length in parsec
        Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
        Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
        Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
        Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
        Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
        Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(
          rho, x1Mom, x2Mom, x3Mom, Egas);
        Real const l_cool = ComputeCoolingLength(rho, Eint, HydroSystem<ShockCloud>::gamma_,
          tables);
        output[bx](i, j, k, ncomp) = std::log10(l_cool / parsec_in_cm);
    });
  }
  amrex::Gpu::streamSynchronize();
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
auto ComputeCellTemp(int i, int j, int k, amrex::Array4<const Real> const &state,
  amrex::Real gamma, cloudyGpuConstTables const& tables)
{
  // return cell temperature
  Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
  Real const x1Mom = state(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
  Real const x2Mom = state(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
  Real const x3Mom = state(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
  Real const Egas = state(i, j, k, HydroSystem<ShockCloud>::energy_index);
  Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(
        rho, x1Mom, x2Mom, x3Mom, Egas);
  return ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
}

template <>
auto RadhydroSimulation<ShockCloud>::ComputeStatistics()
	-> std::unordered_map<std::string, amrex::Real> {
	// compute scalar statistics
	std::unordered_map<std::string, amrex::Real> stats;

  // save time
  const Real t_cc = std::get<Real>(simulationMetadata_["t_cc"]);
  const Real time = tNew_[0];
  stats["t_over_tcc"] = time / t_cc;

  // save cloud position, velocity
  const Real dx_cgs = std::get<Real>(simulationMetadata_["delta_x"]);
  const Real dvx_cgs = std::get<Real>(simulationMetadata_["delta_vx"]);

  stats["delta_x"] = dx_cgs / parsec_in_cm; // pc
  stats["delta_vx"] = dvx_cgs / 1.0e5; // km/s

  // save total simulation mass
  const Real sim_mass = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_),
    HydroSystem<ShockCloud>::density_index, geom, ref_ratio);

  stats["sim_mass"] = sim_mass / solarmass_in_g;

  // compute cloud mass according to temperature threshold
  auto tables = cloudyTables.const_tables();

  const Real M_cl_1e4 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (T < 1.0e4) ? rho : 0.0;
      return result;
    });
  const Real M_cl_6000 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (T < 6000.) ? rho : 0.0;
      return result;
    });  
  const Real M_cl_3000 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (T < 3000.) ? rho : 0.0;
      return result;
    });
  const Real M_cl_300 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (T < 300.) ? rho : 0.0;
      return result;
    });

  stats["cloud_mass_1e4"] = M_cl_1e4 / solarmass_in_g;
  stats["cloud_mass_6000"] = M_cl_6000 / solarmass_in_g;
  stats["cloud_mass_3000"] = M_cl_3000 / solarmass_in_g;
  stats["cloud_mass_300"] = M_cl_300 / solarmass_in_g;

  // compute cloud mass according to passive scalar threshold
  const Real M_cl_scalar = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const C = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (C > 1.0e-5) ? rho : 0.0;
      return result;
    });
  const Real M_cl_scalar_01 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const C = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = (C > 0.1) ? rho : 0.0;
      return result;
    });
  const Real M_cl_scalar_01_09 = computeVolumeIntegral(
    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const& state) noexcept {
      Real const C = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index);
      Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const result = ((C > 0.1) && (C < 0.9)) ? rho : 0.0;
      return result;
    });

  stats["cloud_mass_scalar"] = M_cl_scalar / solarmass_in_g;
  stats["cloud_mass_scalar_01"] = M_cl_scalar_01 / solarmass_in_g;
  stats["cloud_mass_scalar_01_09"] = M_cl_scalar_01_09 / solarmass_in_g;

	return stats;
}

template <>
void RadhydroSimulation<ShockCloud>::ErrorEst(int lev, amrex::TagBoxArray &tags,
                                              Real /*time*/, int /*ngrow*/) {
  // tag cells for refinement
  amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  const Real min_dx = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
  auto tables = cloudyTables.const_tables();

  const auto state = state_new_[lev].const_arrays();
  const auto tag = tags.arrays();

  amrex::ParallelFor(state_new_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
      Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
      Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
      Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
      Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
      Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
      Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(
        rho, x1Mom, x2Mom, x3Mom, Egas);
      Real const l_cool = ComputeCoolingLength(rho, Eint, HydroSystem<ShockCloud>::gamma_,
        tables);
      
      if (l_cool < min_dx) {
        tag[bx](i, j, k) = amrex::TagBox::SET;
      }
    });
    amrex::Gpu::streamSynchronize();
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ShockCloud>::ComputePlanckOpacity(const double /*rho*/,
                                            const double /*Tgas*/) -> double {
  return 1000.; // cm^2 g^-1
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ShockCloud>::ComputeRosselandOpacity(const double /*rho*/,
                                               const double /*Tgas*/)
    -> double {
  return 1000.; // cm^2 g^-1
}

template <>
void RadSystem<ShockCloud>::AddSourceTerms(array_t &consVar,
                                           arrayconst_t & /*radEnergySource*/,
                                           arrayconst_t & /*advectionFluxes*/,
                                           amrex::Box const &indexRange,
                                           amrex::Real dt) {
  arrayconst_t &consPrev = consVar; // make read-only
  array_t &consNew = consVar;

  // Add source terms
  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    const double c = c_light_;
    const double chat = c_hat_;

    // load fluid properties
    const double rho = consPrev(i, j, k, gasDensity_index);
    const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
    const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
    const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
    const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
    const double Egas0 =
        ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
    const double Ekin0 = Egastot0 - Egas0;

    // load radiation energy
    const double Erad0 = consPrev(i, j, k, radEnergy_index);
    AMREX_ASSERT(Egas0 > 0.0);
    AMREX_ASSERT(Erad0 > 0.0);

    // 1. Compute radiation energy update
    // (Assuming 4\piB == 0, the implicit update can be written in closed form.
    //  In this limit, the opacity cannot be temperature-dependent.)
    const double T_gas = 0;

    // compute Planck opacity
    const double kappa = ComputePlanckOpacity(rho, T_gas);
    AMREX_ASSERT(kappa >= 0.);

    // compute update in closed form, assuming zero emissivity
    const double Erad_guess = Erad0 / (1.0 + (rho * kappa) * chat * dt);

    // 2. Compute radiation flux update
    amrex::GpuArray<amrex::Real, 3> Frad_t0{};
    amrex::GpuArray<amrex::Real, 3> Frad_t1{};

    Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index);
    Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index);
    Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index);

    amrex::Real const kappaRosseland = ComputeRosselandOpacity(rho, T_gas);

    for (int n = 0; n < 3; ++n) {
      Frad_t1[n] = Frad_t0[n] / (1.0 + (rho * kappaRosseland) * chat * dt);
    }
    consNew(i, j, k, x1RadFlux_index) = Frad_t1[0];
    consNew(i, j, k, x2RadFlux_index) = Frad_t1[1];
    consNew(i, j, k, x3RadFlux_index) = Frad_t1[2];

    // 3. Compute conservative gas momentum update
    amrex::GpuArray<amrex::Real, 3> dF{};
    amrex::GpuArray<amrex::Real, 3> dMomentum{};

    for (int n = 0; n < 3; ++n) {
      dF[n] = Frad_t1[n] - Frad_t0[n];
      dMomentum[n] = -dF[n] / (c * chat);
    }

    consNew(i, j, k, x1GasMomentum_index) += dMomentum[0];
    consNew(i, j, k, x2GasMomentum_index) += dMomentum[1];
    consNew(i, j, k, x3GasMomentum_index) += dMomentum[2];

    if constexpr (gamma_ != 1.0) {
      amrex::Real x1GasMom1 = consNew(i, j, k, x1GasMomentum_index);
      amrex::Real x2GasMom1 = consNew(i, j, k, x2GasMomentum_index);
      amrex::Real x3GasMom1 = consNew(i, j, k, x3GasMomentum_index);

      // 4a. Compute radiation work terms
      amrex::Real const Egastot1 =
          ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas0);

      // compute difference in gas kinetic energy
      amrex::Real const Ekin1 = Egastot1 - Egas0;
      amrex::Real const dEkin_work = Ekin1 - Ekin0;

      // compute loss of radiation energy to gas kinetic energy
      amrex::Real const dErad_work = -(c_hat_ / c_light_) * dEkin_work;

      // 4b. Store new radiation energy, gas energy
      consNew(i, j, k, radEnergy_index) = Erad_guess + dErad_work;
      consNew(i, j, k, gasEnergy_index) = Egastot1;

    } else {
      amrex::ignore_unused(Egas0);
    } // endif gamma != 1.0
  });
}

auto problem_main() -> int {
  // Problem initialization
  constexpr int nvars = RadhydroSimulation<ShockCloud>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet
    boundaryConditions[n].setHi(0, amrex::BCType::foextrap); // extrapolate

    boundaryConditions[n].setLo(1, amrex::BCType::foextrap); // extrapolate
    boundaryConditions[n].setHi(1, amrex::BCType::foextrap);

    boundaryConditions[n].setLo(2, amrex::BCType::foextrap);
    boundaryConditions[n].setHi(2, amrex::BCType::foextrap);
  }

  RadhydroSimulation<ShockCloud> sim(boundaryConditions, ::enableRadiation);

  // Read Cloudy tables
  readCloudyData(sim.cloudyTables);
  amrex::Print() << std::endl;

  // Read problem parameters
  // set global variables (read-only after setting them here)
  amrex::ParmParse pp;
  Real nH_bg = NAN;
  Real nH_cloud = NAN;
  Real P_over_k = NAN;
  Real M0 = NAN;
  int enable_cooling = 1; // default (0 = off, 1, = on)
  
  // enable cooling?
  pp.query("enable_cooling", enable_cooling);
  if (enable_cooling == 1) {
    ::enable_cooling = true;
  } else {
    ::enable_cooling = false;
  }
  
  // background gas H number density
  pp.query("nH_bg", nH_bg); // cm^-3

  // cloud H number density
  pp.query("nH_cloud", nH_cloud); // cm^-3

  // background gas pressure
  pp.query("P_over_k", P_over_k); // K cm^-3

  // cloud radius
  pp.query("R_cloud_pc", ::R_cloud); // pc
  ::R_cloud *= parsec_in_cm;             // convert to cm

  // (pre-shock) Mach number
  // (N.B. *not* the same as Mach_wind!)
  pp.query("Mach_shock", M0); // dimensionless

  // gas temperature floor
  pp.query("T_floor", ::T_floor); // K

  // initial FUV radiation field
  pp.query("Erad_initial_Habing", ::Erad0); // G_0
  ::Erad0 *= G_0;                           // convert to cgs

  // incident FUV radiation field
  pp.query("Erad_incident_Habing", ::Erad_bdry); // G_0
  ::Erad_bdry *= G_0;                            // convert to cgs

  // compute background pressure
  // (pressure equilibrium should hold *before* the shock enters the box)
  ::P0 = P_over_k * boltzmann_constant_cgs_; // erg cm^-3
  amrex::Print() << fmt::format("Pressure = {} K cm^-3\n", P_over_k);

  // compute mass density of background, cloud
  ::rho0 = nH_bg * m_H / cloudy_H_mass_fraction;    // g cm^-3
  ::rho1 = nH_cloud * m_H / cloudy_H_mass_fraction; // g cm^-3

  AMREX_ALWAYS_ASSERT(!std::isnan(::rho0));
  AMREX_ALWAYS_ASSERT(!std::isnan(::rho1));
  AMREX_ALWAYS_ASSERT(!std::isnan(::P0));

  // check temperature of cloud, background
  constexpr Real gamma = HydroSystem<ShockCloud>::gamma_;
  auto tables = sim.cloudyTables.const_tables();
  const Real Eint_bg = ::P0 / (gamma - 1.);
  const Real Eint_cl = ::P0 / (gamma - 1.);
  const Real T_bg = ComputeTgasFromEgas(rho0, Eint_bg, gamma, tables);
  const Real T_cl = ComputeTgasFromEgas(rho1, Eint_cl, gamma, tables);
  amrex::Print() << fmt::format("T_bg = {} K\n", T_bg);
  amrex::Print() << fmt::format("T_cl = {} K\n", T_cl);

  // compute shock jump conditions from rho0, P0, and M0
  const Real v_pre = M0 * std::sqrt(gamma * P0 / rho0);
  const Real rho_post =
      rho0 * (gamma + 1.) * M0 * M0 / ((gamma - 1.) * M0 * M0 + 2.);
  const Real v_post = v_pre * (rho0 / rho_post);
  const Real v_wind = v_pre - v_post;

  const Real P_post = P0 * (2. * gamma * M0 * M0 - (gamma - 1.)) / (gamma + 1.);
  const Real Eint_post = P_post / (gamma - 1.);
  const Real T_post = ComputeTgasFromEgas(rho_post, Eint_post, gamma, tables);
  amrex::Print() << fmt::format("T_wind = {} K\n", T_post);

  ::v_wind = v_wind; // set global variables
  ::rho_wind = rho_post;
  ::P_wind = P_post;
  amrex::Print() << fmt::format("v_wind = {} km/s (v_pre = {}, v_post = {})\n",
                                v_wind / 1.0e5, v_pre / 1.0e5, v_post / 1.0e5);

  // compute cloud-crushing time
  const Real chi = rho1 / rho0;
  const Real t_cc = std::sqrt(chi) * R_cloud / v_wind;
  amrex::Print() << fmt::format("t_cc = {} kyr\n", t_cc / (1.0e3 * 3.15e7));
  amrex::Print() << std::endl;

  // compute maximum simulation time
  const double max_time = 20.0 * t_cc;

  // set simulation parameters
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = ::enableRadiation;

  sim.reconstructionOrder_ = 3;          // PPM for hydro
  sim.radiationReconstructionOrder_ = 2; // PLM for radiation
  sim.densityFloor_ = 1.0e-3 * rho0;     // density floor (to prevent vacuum)
  sim.stopTime_ = max_time;

  // set metadata
  sim.simulationMetadata_["delta_x"] = 0._rt;
  sim.simulationMetadata_["delta_vx"] = 0._rt;
  sim.simulationMetadata_["rho_wind"] = rho_wind;
  sim.simulationMetadata_["v_wind"] = v_wind;
  sim.simulationMetadata_["P_wind"] = P_wind;
  sim.simulationMetadata_["M0"] = M0;
  sim.simulationMetadata_["t_cc"] = t_cc;

  // Set initial conditions
  sim.setInitialConditions();

  // run simulation
  sim.evolve();

  // Cleanup and exit
  int status = 0;
  return status;
}
