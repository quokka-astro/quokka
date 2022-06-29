//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include <limits>
#include <math.h>
#include <iostream>
#include <random>

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
#include "AMReX_GpuDevice.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_TableData.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_Random.H"


#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_sne.hpp"


using amrex::Real;
using namespace amrex;
constexpr double  Const_G  = 6.67e-8;
constexpr double  Const_mH = 1.67e-24;
constexpr double  Msun     = 2.e33;
constexpr double  yr_to_s  = 3.154e7;
constexpr double  Myr      = 1.e6*yr_to_s;
constexpr double  pc       = 3.018e18;
//constexpr double  kpc      = 3.018e21;
//constexpr double  Mu       = 0.6;
//constexpr double  tMAX     = 1.005*Myr;
constexpr double  kmps     = 1.e5; 
// amrex::Real *count;
// constexpr double Tgas0 = 6000.;       // K
constexpr amrex::Real T_floor = 10.0; // K
// constexpr double rho0 = 0.6 * Const_mH;    // g cm^-3
#define MAX 100

struct NewProblem {};



// if false, use octant symmetry instead

template <> struct EOS_Traits<NewProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = false;
};

template <>
void RadhydroSimulation<NewProblem>::setInitialConditionsAtLevel(int lev) {
  
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
  const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
      geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
      geom[lev].ProbHiArray();


  for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[lev].array(iter);
    auto const &setFB = set_feedback_[lev].array(iter);
    amrex::Real prob = amrex::Random();
 
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {


      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
			// double prob = std::srand
 
      
			//double R = std::sqrt(x*x + y*y);
      double sigma1 = 7. * kmps;
      double sigma2 = 70. * kmps;
      double rho01  = 2.85 * Const_mH;
      double rho02  = 1.e-5 * 2.85 * Const_mH;

      /*Calculate DM Potential*/
      double R0, rho_dm, prefac;
      rho_dm = 0.0064 * Msun/pc/pc/pc;
      R0     = 8.e3 * pc;
      prefac = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2);
      double Phidm =  (prefac * std::log(1. + std::pow(z/R0, 2)));

      /*Calculate Stellar Disk Potential*/
      double z_star, Sigma_star, prefac2;
      z_star = 245.0 * pc;
      Sigma_star = 42.0 * Msun/pc/pc;
      prefac2 = 2.* 3.1415 * Const_G * Sigma_star * z_star ;
      double Phist =  prefac2 * (std::pow(1. + z*z/z_star/z_star,0.5) -1.);

      double Phitot = Phist + Phidm;

			double rho = rho01 * std::exp(-Phitot/std::pow(sigma1,2.0)) ;
             rho+= rho02 * std::exp(-Phitot/std::pow(sigma2,2.0));         //in g/cc
			//double P   = rho * boltzmann_constant_cgs_ * 1.e6/Const_mH/Mu; //For an isothermal profile
      double P = rho01 * std::pow(sigma1, 2.0) + rho02 * std::pow(sigma2, 2.0);


      AMREX_ASSERT(!std::isnan(rho));
      AMREX_ASSERT(!std::isnan(vx));
			AMREX_ASSERT(!std::isnan(vy));
			AMREX_ASSERT(!std::isnan(vz));

      for (int n = 0; n < state.nComp(); ++n) {
        state(i, j, k, n) = 0.; // zero fill all components
      }
			const auto gamma = HydroSystem<NewProblem>::gamma_;

      setFB(i,j,k,0) = -10.0;
      if(i==128 && j==128 && k==256){
        setFB(i, j, k, 0) = 10.0;
        // rho = 10.0 * Const_mH;
      }
     

      state(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
      state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
      state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
      state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
      state(i, j, k, HydroSystem<NewProblem>::energy_index) = P / (gamma - 1.);

    });
  }
  // set flag
  areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<NewProblem>::ErrorEst(int lev,
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
          HydroSystem<NewProblem>::ComputePressure(state, i, j, k);

      amrex::Real const P_xplus =
          HydroSystem<NewProblem>::ComputePressure(state, i + 1, j, k);
      amrex::Real const P_xminus =
          HydroSystem<NewProblem>::ComputePressure(state, i - 1, j, k);
      amrex::Real const P_yplus =
          HydroSystem<NewProblem>::ComputePressure(state, i, j + 1, k);
      amrex::Real const P_yminus =
          HydroSystem<NewProblem>::ComputePressure(state, i, j - 1, k);
      amrex::Real const P_zplus =
          HydroSystem<NewProblem>::ComputePressure(state, i, j, k + 1);
      amrex::Real const P_zminus =
          HydroSystem<NewProblem>::ComputePressure(state, i, j, k - 1);

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
 /*
template <>
void RadhydroSimulation<NewProblem>::computeAfterEvolve(
    amrex::Vector<amrex::Real> &initSumCons) {
  // check conservation of total energy
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

  amrex::Real const Egas0 =
      initSumCons[RadSystem<NewProblem>::gasEnergy_index];
  amrex::Real const Erad0 =
      initSumCons[RadSystem<NewProblem>::radEnergy_index];
  amrex::Real const Etot0 = Egas0 + (RadSystem<NewProblem>::c_light_ /
                                     RadSystem<NewProblem>::c_hat_) *
                                        Erad0;

  amrex::Real const Egas =
      state_new_[0].sum(RadSystem<NewProblem>::gasEnergy_index) * vol;
  amrex::Real const Erad =
      state_new_[0].sum(RadSystem<NewProblem>::radEnergy_index) * vol;
  amrex::Real const Etot = Egas + (RadSystem<NewProblem>::c_light_ /
                                   RadSystem<NewProblem>::c_hat_) *
                                      Erad;

  // compute kinetic energy
  amrex::MultiFab Ekin_mf(boxArray(0), DistributionMap(0), 1, 0);
  for (amrex::MFIter iter(state_new_[0]); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = state_new_[0].const_array(iter);
    auto const &ekin = Ekin_mf.array(iter);
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      // compute kinetic energy
      Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
      Real px = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
      Real py = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
      Real pz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
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
*/

/**Writing down the potential*/
/*
double Phitot(double z){
 double R0, rho_dm;
 double prefac;
 rho_dm = 0.0064 * Msun/pc/pc/pc;
 R0     = 8.e3 * pc;
 prefac = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2);
 return (prefac * std::log(1. + std::pow(z/R0, 2)));
}*/

/**Adding Cooling Terms*/


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
      ComputeTgasFromEgas(rho, Eint, HydroSystem<NewProblem>::gamma_, tables);

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
      const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
      const Real x1Mom =
          state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
      const Real x2Mom =
          state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
      const Real x3Mom =
          state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
      const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

      Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom,
                                                              x3Mom, Egas);

      ODEUserData user_data{rho, tables};
      quokka::valarray<Real, 1> y = {Eint};
      quokka::valarray<Real, 1> abstol = {
          reltol_floor * ComputeEgasFromTgas(rho, T_floor,
                                             HydroSystem<NewProblem>::gamma_,
                                             tables)};

      // do integration with RK2 (Heun's method)
      int steps_taken = 0;
      rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, steps_taken);

      const Real Egas_new = RadSystem<NewProblem>::ComputeEgasFromEint(
          rho, x1Mom, x2Mom, x3Mom, y[0]);

      state(i, j, k, HydroSystem<NewProblem>::energy_index) = Egas_new;
    });
  }
}

/**Adding Supernova Source Terms*/

void AddSupernova(amrex::MultiFab &mf, amrex::MultiFab &mf1, const Real dt_in, 
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx){
  BL_PROFILE("HydroSimulation::AddSupernova()")
  
  const Real dt = dt_in;
  
  double Mass_source = 8.* Msun;
  double Energy_source = 1.e51;
  

  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);
    auto const &setFB = mf1.array(iter);
    // amrex::Real prob = amrex::Random();
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {

    double vol, n_sn; //, rho_cell, n_sn, prob_sn, t_ff, mdot, eff=0.01;
	  vol       =  AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	  // rho_cell  = state(i, j, k, HydroSystem<NewProblem>::density_index);
	  // // n_sn      = (rho_cell* vol)/(100.* Msun);
    //n_sn      = 5.0;
    // t_ff      = std::sqrt(3.*(std::atan(1)*4.0)/(32.*Const_G*rho_cell));
    // mdot      = (eff * rho_cell /t_ff) * vol;
    // prob_sn   = mdot*dt/(100.*Msun);
      if(setFB(i,j,k,0)>=0.0 && setFB(i,j,k,0)<1.e2) {
         state(i, j, k, HydroSystem<NewProblem>::density_index)+= setFB(i,j,k,0) * Mass_source/vol;
         state(i, j, k, HydroSystem<NewProblem>::energy_index) += setFB(i,j,k,0) * Energy_source/vol;
         setFB(i,j,k,0) = 1.e2;
      }
    });
  }
  
}

void ResetFBFlag(amrex::MultiFab &mf, amrex::MultiFab &mf1){
  // BL_PROFILE("HydroSimulation::AddSupernova()")
  
  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);
    auto const &setFB = mf1.array(iter);
    // engine.rand_state = amrex::getRandState();

    amrex::ParallelForRNG(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k, RandomEngine const& engine) noexcept {
    
     double rho_cell, rho_thresh;
     double prob = amrex::Random(engine);
	   rho_cell   = state(i, j, k, HydroSystem<NewProblem>::density_index);
     rho_thresh = 50. * Const_mH;
     int n_sn = RandomPoisson(5.0, engine);
      if(rho_cell>rho_thresh && setFB(i,j,k,0)<0.0 && prob>0.9) {
         setFB(i, j, k, 0) = n_sn;
      }
    });
  }  
}


template <>
void RadhydroSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int iteration, int ncycle)
{
  //computeCooling(state_new_[lev], dt_lev, cloudyTables);
  amrex::Real prob= amrex::Random();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi   = geom[lev].ProbHiArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[0].CellSizeArray();

  int index_i = std::ceil(prob*(prob_hi[0]-prob_lo[0])/dx[0]);
  prob= amrex::Random();
  int index_j = std::ceil(prob*(prob_hi[1]-prob_lo[1])/dx[1]);
  prob= amrex::Random();
  int index_k = std::ceil(prob*(prob_hi[2]-prob_lo[2])/dx[2]);

  // RandomEngine engine;
  // engine.rand_state = amrex::getRandState();
  
  AddSupernova(state_new_[lev], set_feedback_[lev], dt_lev, dx); 
  ResetFBFlag(state_new_[lev], set_feedback_[lev]); 

}


template <>
void HydroSystem<NewProblem>::EnforcePressureFloor(
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
            (rho_new / Const_mH) * boltzmann_constant_cgs_ * T_floor;

        if constexpr (!is_eos_isothermal()) {
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
   
  const int nvars = RadhydroSimulation<NewProblem>::nvarTotal_;
  amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
  for (int n = 0; n < nvars; ++n) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
          boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
          boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
        }
      }

  // Problem initialization
  RadhydroSimulation<NewProblem> sim(boundaryConditions);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
  // sim.stopTime_ = tMAX;          // seconds
  sim.cflNumber_ = 0.3;         // *must* be less than 1/3 in 3D!
  // sim.maxTimesteps_ = 50000;
  // sim.plotfileInterval_ = 5;
  // Read Cloudy tables
 readCloudyData(sim.cloudyTables);

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
