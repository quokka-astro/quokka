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
#include "quadrature.hpp"


using amrex::Real;
using namespace amrex;

#define MAX 100

struct NewProblem {};

template <> struct HydroSystem_Traits<NewProblem> {
  static constexpr double gamma = 5./3.;
  static constexpr bool reconstruct_eint = true; //Set to true - temperature
};
template <> struct Physics_Traits<NewProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;
  static constexpr bool is_mhd_enabled = false;
  static constexpr int numPassiveScalars = 3; // number of passive scalars
};

template <> struct SimulationData<NewProblem> {

  // cloudy_tables cloudyTables;
  std::unique_ptr<amrex::TableData<Real, 3>> table_data;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	int nblast = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN; // rate per unit time per unit volume
	Real E_blast = 1.0e51;	    // ergs
	Real M_ejecta = 0;	    // 10.0 * Msun; // g

	Real refine_threshold = 1.0; // gradient refinement threshold
};



template <>
void RadhydroSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem) {
  
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
    const amrex::Box &indexRange = grid_elem.indexRange_;
    const amrex::Array4<double>& state_cc = grid_elem.array_;
    
    double vol       =  AMREX_D_TERM(dx[0], *dx[1], *dx[2]);


  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

    
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

      double z_star = 245.0 * pc;
      double Sigma_star = 42.0 * Msun/pc/pc;
      double rho_dm = 0.0064 * Msun/pc/pc/pc;
      double R0     = 8.e3 * pc;

      double sigma1 = 7. * kmps;
      double sigma2 = 70. * kmps;
      double rho01  = 2.85 * Const_mH;
      double rho02  = 1.e-5 * 2.85 * Const_mH;

      /*Calculate DM Potential*/
      double prefac;
      prefac = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2);
      double Phidm =  (prefac * std::log(1. + std::pow(z/R0, 2)));

      /*Calculate Stellar Disk Potential*/
      double prefac2;
      prefac2 = 2.* 3.1415 * Const_G * Sigma_star * z_star ;
      double Phist =  prefac2 * (std::pow(1. + z*z/z_star/z_star,0.5) -1.);

      double Phitot = Phist + Phidm;

			double rho, rho_disk, rho_halo;
             rho_disk = rho01 * std::exp(-Phitot/std::pow(sigma1,2.0)) ;
             rho_halo = rho02 * std::exp(-Phitot/std::pow(sigma2,2.0));         //in g/cc
             rho = (rho_disk + rho_halo);
            
      double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

      AMREX_ASSERT(!std::isnan(rho));
      
			const auto gamma = HydroSystem<NewProblem>::gamma_;
     
      if(std::sqrt(z*z)<0.25*kpc) {
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)      = 1.e2/vol;  //Disk tracer
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1)    = 1.e-5/vol;  //Halo tracer
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)    = 1.e-5/vol;  //Injected tracer
       }

       else {
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)      = 1.e-5/vol;  //Disk tracer
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1)    = 1.e2/vol;  //Halo tracer
        state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)    = 1.e-5/vol;  //Injected tracer
       }

      state_cc(i, j, k, HydroSystem<NewProblem>::density_index)    = rho;
      state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
      state_cc(i, j, k, HydroSystem<NewProblem>::energy_index)         = P / (gamma - 1.);

    });
  }

template <>
void RadhydroSimulation<NewProblem>::ErrorEst(int lev,
                                                amrex::TagBoxArray &tags,
                                                amrex::Real /*time*/ ,
                                                int /*ngrow*/) {
  // tag cells for refinement

  const amrex::Real eta_threshold = 4.0; // gradient refinement threshold
 
  for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_cc_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
   
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

        amrex::Real  delMoxy = Msun;
        amrex::Real  Znorm = 1.e3;
        amrex::Real  ZOinit = 8.6e-3;
        amrex::Real rho_oxy_ = ZOinit *  state(i, j, k, HydroSystem<NewProblem>::density_index) ;
         
         amrex::Real scal_xyz   = ZOinit + ((delMoxy/Znorm) * state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)/
                                                    state(i, j, k, HydroSystem<NewProblem>::density_index)) ;

        amrex::Real scal_xplus  = ZOinit + ((delMoxy/Znorm) * state(i+1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)/
                                                              state(i+1, j, k, HydroSystem<NewProblem>::density_index) ) ;

        amrex::Real scal_xminus = ZOinit + ((delMoxy/Znorm) * state(i-1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)/
                                                              state(i-1, j, k, HydroSystem<NewProblem>::density_index)) ;

        amrex::Real scal_yplus  = ZOinit + ((delMoxy/Znorm) *  state(i, j+1, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)/ 
                                                               state(i, j+1, k, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_yminus = ZOinit + ((delMoxy/Znorm) *  state(i, j-1, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) / 
                                                               state(i, j-1, k, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_zplus  = ZOinit + ((delMoxy/Znorm) *  state(i, j, k+1, Physics_Indices<NewProblem>::pscalarFirstIndex+2)/ 
                                                               state(i, j, k+1, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_zminus = ZOinit + ((delMoxy/Znorm) *  state(i, j, k-1, Physics_Indices<NewProblem>::pscalarFirstIndex+2) / 
                                                               state(i, j, k-1, HydroSystem<NewProblem>::density_index));
        
        amrex::Real del_scalx   = std::abs(scal_xplus - scal_xminus)/2;
        amrex::Real del_scaly   = std::abs(scal_yplus - scal_zminus)/2.;
        amrex::Real del_scalz   = std::abs(scal_zplus - scal_zminus)/2.;
        // std::max(std::abs(scal_xplus - scal_xyz), std::abs(scal_xminus - scal_xyz));
        // amrex::Real del_scaly   = std::max(std::abs(scal_yplus - scal_xyz), std::abs(scal_yminus - scal_xyz));
        // amrex::Real del_scalz   = std::max(std::abs(scal_zplus - scal_xyz), std::abs(scal_zminus - scal_xyz));
        
        amrex::Real const grad_scal = (del_scalx  +  del_scaly  + del_scalz )/scal_xyz;          
        
        if ((grad_scal > eta_threshold)) {
        tag(i, j, k) = amrex::TagBox::SET;
        // printf("Reached here=%d, %d, %d, %.2e\n", i, j, k, grad_scal);
      }

     
    });
  }
}


/*****Adding Cooling Terms*****/

// struct ODEUserData {
//   amrex::Real rho;
//   cloudyGpuConstTables tables;
// };

// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
// user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data,
//          quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int {
//   // unpack user_data
//   auto *udata = static_cast<ODEUserData *>(user_data);
//   Real rho = udata->rho;
//   cloudyGpuConstTables &tables = udata->tables;

//   // compute temperature (implicit solve, depends on composition)
//   Real Eint = y_data[0];
//   Real T =
//       ComputeTgasFromEgas(rho, Eint, HydroSystem<NewProblem>::gamma_, tables);
    
//   // compute cooling function
//   y_rhs[0] = cloudy_cooling_function(rho, T, tables);
//   return 0;
// }

// void computeCooling(amrex::MultiFab &mf, const Real dt_in,
//                     cloudy_tables &cloudyTables) {
//   BL_PROFILE("RadhydroSimulation::computeCooling()")

//   const Real dt = dt_in;
//   const Real reltol_floor = 0.01;
//   const Real rtol = 1.0e-4; // not recommended to change this

//   auto tables = cloudyTables.const_tables();
                    
//   // loop over all cells in MultiFab mf
//   for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
//     const amrex::Box &indexRange = iter.validbox();
//     auto const &state = mf.array(iter);

//     amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
//                                                         int k) noexcept {
//       const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
//       const Real x1Mom =
//           state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
//       const Real x2Mom =
//           state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
//       const Real x3Mom =
//           state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
//       const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

//       Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom,
//                                                               x3Mom, Egas);

//       ODEUserData user_data{rho, tables};
//       quokka::valarray<Real, 1> y = {Eint};
//       quokka::valarray<Real, 1> abstol = {
//           reltol_floor * ComputeEgasFromTgas(rho, T_floor,
//                                              HydroSystem<NewProblem>::gamma_,
//                                              tables)};
                                      
//       // do integration with RK2 (Heun's method)
//       int steps_taken = 0;
      
//       rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, steps_taken);


//       const Real Egas_new = RadSystem<NewProblem>::ComputeEgasFromEint(
//           rho, x1Mom, x2Mom, x3Mom, y[0]);

//       const Real Eint_new = y[0];
//       const Real dEint = Eint_new - Eint;
//       Real Temp =
//       ComputeTgasFromEgas(rho, Eint_new, HydroSystem<NewProblem>::gamma_, tables);

//       state(i, j, k, HydroSystem<NewProblem>::energy_index) += dEint;
//       state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += dEint;
      
//     });

//     AMREX_ASSERT(!state.contains_nan(0, state.nComp()));
//   }
// } 

void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// inject energy into cells with stochastic sampling
	BL_PROFILE("RadhydroSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	const Real Lx = prob_hi[0] - prob_lo[0];
	const Real Ly = prob_hi[1] - prob_lo[1];
	const Real Lz = prob_hi[2] - prob_lo[2];

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;
		
   
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = NAN;
				Real y0 = NAN;
				Real z0 = NAN;
        Real Rpds = 0.0;
        
        x0 = std::abs(xc -px(n));
        y0 = std::abs(yc -py(n));
        z0 = std::abs(zc -pz(n));

        if(x0<0.5*dx[0] && y0<0.5*dx[1] && z0< 0.5*dx[2] ) {

        state(i, j, k, HydroSystem<NewProblem>::energy_index)         +=   rho_eint_blast; 
        state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) +=    rho_eint_blast; 
        state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2)+=  1.e3/cell_vol;
        printf("The location of SN=%d,%d,%d\n",i, j, k);
        printf("SN added at level=%d\n", level);
        printf("The total number of SN gone off=%d\n", cum_sn);
        Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index)/Const_mH, -3./7.);
        printf("Rpds = %.2e pc\n", Rpds);
        }
			}
		});
	}
}


template <> void RadhydroSimulation<NewProblem>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution
  
	const Real dt_coarse = dt_[0];
	const Real domain_vol = geom[0].ProbSize();
  const Real domain_area = geom[0].ProbLength(0) * geom[0].ProbLength(1); 
  const Real mean = 0.0;
  const Real stddev = hscale/geom[0].ProbLength(2)/2.;
  
	const Real expectation_value = ks_sigma_sfr * domain_area * dt_coarse;
  
	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
  
	if (count > 0) {
		amrex::Print() << "\t" << count << " SNe to be exploded.\n";
  }
	// resize particle arrays
	amrex::Array<int, 1> const lo{0};
	amrex::Array<int, 1> const hi{count};
	userData_.blast_x = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_y = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_z = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.nblast = count;
	userData_.SN_counter_cumulative += count;

	// for each, sample location at random
	auto const &px = userData_.blast_x->table();
	auto const &py = userData_.blast_y->table();
	auto const &pz = userData_.blast_z->table();
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
	}
}

/*******************************************************************/



template <>
void RadhydroSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int ncycle)
{
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi   = geom[lev].ProbHiArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
  
  AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
  
  // computeCooling(state_new_cc_[lev], dt_lev, userData_.cloudyTables);
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec)
                                  -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> {
 
     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;

      double z_star = 245.0 * pc;
      double Sigma_star = 42.0 * Msun/pc/pc;
      double rho_dm = 0.0064 * Msun/pc/pc/pc;
      double R0     = 8.e3 * pc;
      
      double x = posvec[0];
     
     grad_potential[0] =  0.0;

    #if (AMREX_SPACEDIM >= 2)
       double y = posvec[1];
       grad_potential[1] = 0.0;
    #endif
    #if (AMREX_SPACEDIM >= 3)
       double z      = posvec[2];
       grad_potential[2]  = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2) * (2.* z/std::pow(R0,2))/(1. + std::pow(z,2)/std::pow(R0,2));
       grad_potential[2] += 2.* 3.1415 * Const_G * Sigma_star * (z/z_star) * (std::pow(1. + z*z/(z_star*z_star), -0.5));
    #endif

return grad_potential;
}

/* Add Strang Split Source Term for External Fixed Potential Here */
template <>
void RadhydroSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time,
				 amrex::Real dt_lev)
{
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
  const Real dt = dt_lev;

  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {

      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec, GradPhi;
      double x1mom_new, x2mom_new, x3mom_new;

      const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
      const Real x1mom =
          state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
      const Real x2mom =
          state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
      const Real x3mom =
          state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
      const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

					const auto vx = x1mom / rho;
					const auto vy = x2mom / rho;
					const auto vz = x3mom / rho;
					const double vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
      
      Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom,
                                                              x3mom, Egas);
      

      posvec[0] = prob_lo[0] + (i+0.5)*dx[0];

        #if (AMREX_SPACEDIM >= 2)
          posvec[1] = prob_lo[1] + (j+0.5)*dx[1]; 
        #endif

        #if (AMREX_SPACEDIM >= 3)
          posvec[2] = prob_lo[2] + (k+0.5)*dx[2]; 
        #endif

      GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);   
      // GradPhi[1] = 0.0;
      // GradPhi[2] = 0.0;

      x1mom_new = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) + dt * (-rho * GradPhi[0]);
      x2mom_new = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) + dt * (-rho * GradPhi[1]);
      x3mom_new = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) + dt * (-rho * GradPhi[2]);

      state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
      state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
      state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;
      
      state(i, j, k, HydroSystem<NewProblem>::energy_index) = RadSystem<NewProblem>::ComputeEgasFromEint(
          rho, x1mom_new, x2mom_new, x3mom_new, Eint);
     });  
   }
}

/**************************End Adding Strang Split Source Term *****************/

auto problem_main() -> int {

  const int nvars = RadhydroSimulation<NewProblem>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);


  /*Implementing Outflowing Boundary Conditions in the Z-direction*/

	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				// outflowing boundary conditions
        if(i==2){
				boundaryConditions[n].setLo(i, amrex::BCType::foextrap);
				boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
        }
        else{
          boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
          boundaryConditions[n].setHi(i, amrex::BCType::int_dir); // periodic
        }
        }}


  /**For a Fully Periodic Box*/
	// for (int n = 0; n < nvars; ++n) {
	// 	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				
  //         boundaryConditions[n].setLo(i, amrex::BCType::int_dir); // periodic
  //         boundaryConditions[n].setHi(i, amrex::BCType::int_dir); // periodic
        
  //       }}

   
  // Problem initialization
  RadhydroSimulation<NewProblem> sim(boundaryConditions);
  sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
  sim.cflNumber_ = 0.35;         // *must* be less than 1/3 in 3D!
  

  // readCloudyData(sim.userData_.cloudyTables);
  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}