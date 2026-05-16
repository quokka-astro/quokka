#ifndef EXTERN_TYPE_H
#define EXTERN_TYPE_H

using namespace amrex::literals;

struct eos_param_t {
  bool eos_assume_neutral{true};
  amrex::Real eos_gamma{5.e0 / 3.e0_rt};
};

struct integrator_param_t {
  bool call_eos_in_rhs{true};
  bool integrate_energy{true};
  bool burner_verbose{false};
  bool renormalize_abundances{false};
  bool use_jacobian_caching{true};
  bool use_burn_retry{false};
  bool retry_swap_jacobian{true};
  bool do_species_clip{true};
  bool use_number_densities{false};
  bool subtract_internal_energy{true};
  bool scale_system{false};
  bool nse_include_enu_weak{true};
  bool linalg_do_pivoting{true};
  int jacobian{1};
  int ode_max_steps{150000};
  int nonaka_i{0};
  int nonaka_j{0};
  int nonaka_k{0};
  int nonaka_level{0};
  amrex::Real X_reject_buffer{1.0_rt};
  amrex::Real rtol_spec{1.e-12_rt};
  amrex::Real rtol_enuc{1.e-6_rt};
  amrex::Real atol_spec{1.e-8_rt};
  amrex::Real atol_enuc{1.e-6_rt};
  amrex::Real SMALL_X_SAFE{1.0e-30_rt};
  amrex::Real MAX_TEMP{1.0e11_rt};
  amrex::Real react_boost{-1.e0_rt};
  amrex::Real ode_max_dt{1.e30_rt};
  amrex::Real retry_rtol_spec{-1_rt};
  amrex::Real retry_rtol_enuc{-1_rt};
  amrex::Real retry_atol_spec{-1_rt};
  amrex::Real retry_atol_enuc{-1_rt};
  amrex::Real nse_deriv_dt_factor{0.05_rt};
  std::string nonaka_file{"nonaka_plot.dat"};
};

struct network_param_t {
  bool use_tables{false};
  bool use_c12ag_deboer17{false};
  amrex::Real small_x{1.e-30_rt};
};

struct extern_t {
  eos_param_t eos;
  integrator_param_t integrator;
  network_param_t network;
};

#endif
