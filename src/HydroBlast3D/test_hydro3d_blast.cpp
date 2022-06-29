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
#include <math.h>
#include <iostream>


constexpr double  Const_G  = 6.67e-8;
constexpr double  Const_mH = 1.67e-24;
constexpr double  Msun     = 2.e33;
constexpr double  yr_to_s  = 3.154e7;
constexpr double  Myr      = 1.e6*yr_to_s;
constexpr double  pc       = 3.018e18;
constexpr double  Mu       = 0.6;
constexpr double  tMAX     = 5.*Myr;
constexpr double  kmps     = 1.e5; 


struct NewProblem {};

// if false, use octant symmetry instead

template <> struct HydroSystem_Traits<SedovProblem> {
  static constexpr double gamma = 1.4;
  static constexpr bool reconstruct_eint = false;
  static constexpr int nscalars = 0;       // number of passive scalars
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

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {


      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
			
      
			//double R = std::sqrt(x*x + y*y);
      double sigma1 = 7.  * kmps;
      double sigma2 = 70. * kmps;
      double rho01  = 2.85 * Const_mH;
      double rho02   = 1.e-5 * 2.85 * Const_mH;

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
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

  // check conservation of total energy
  amrex::Real const Egas0 =
      initSumCons[RadSystem<SedovProblem>::gasEnergy_index];
  amrex::Real const Egas =
      state_new_[0].sum(RadSystem<SedovProblem>::gasEnergy_index) * vol;

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

  amrex::Real const abs_err = (Egas - Egas0);
  amrex::Real const rel_err = abs_err / Egas0;

  amrex::Real const rel_err_Ekin = frac_Ekin - frac_Ekin_exact;

  amrex::Print() << "\nInitial energy = " << Egas0 << std::endl;
  amrex::Print() << "Final energy = " << Egas << std::endl;
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

/**Adding Supernova Source Terms*/

void AddSupernova(amrex::MultiFab &mf, const Real dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx){
  BL_PROFILE("RadhydroSimulation::AddSupernova()")
  
  const Real dt = dt_in;
  
  double Mass_source = 8.* Msun;
  double Energy_source = 1.e51;
  

  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);
    amrex::Real prob = amrex::Random();
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {

	 // amrex::Real prob = amrex::Random();
    double vol, rho_cell, n_sn, prob_sn, t_ff, mdot, eff=0.01;
	  vol       =  AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	  rho_cell  = state(i, j, k, HydroSystem<NewProblem>::density_index);
	  n_sn      = (rho_cell* vol)/(100.* Msun);
    t_ff      = std::sqrt(3.*(std::atan(1)*4.0)/(32.*Const_G*rho_cell));
    mdot      = (eff * rho_cell /t_ff) * vol;
    prob_sn   = mdot*dt/(100.*Msun);
      if(prob>0.7) {
         state(i, j, k, HydroSystem<NewProblem>::density_index)+= n_sn * Mass_source/vol;
         state(i, j, k, HydroSystem<NewProblem>::energy_index) += n_sn * Energy_source/vol;
      }
    });
  }
  
}


template <>
void RadhydroSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int iteration, int ncycle)
{
     
amrex::Real prob= amrex::Random();
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[0].CellSizeArray();

if(prob>0.5){
  AddSupernova(state_new_[lev], dt_lev, dx);
}
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
  RadhydroSimulation<SedovProblem> sim(boundaryConditions, false);
  sim.is_hydro_enabled_ = true;
  sim.is_radiation_enabled_ = false;
  sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
  sim.stopTime_ = tMAX;          // seconds
  sim.cflNumber_ = 0.3;         // *must* be less than 1/3 in 3D!
  sim.maxTimesteps_ = 20000;
  sim.plotfileInterval_ = 500;

  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}
