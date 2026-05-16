#ifndef EXTERN_TYPE_H
#define EXTERN_TYPE_H

using namespace amrex::literals;

struct eos_param_t {
  amrex::Real eos_gamma_default{1.4_rt};
  amrex::Real species_1_gamma{5. / 3._rt};
  amrex::Real species_1_mass{9.10938188e-28_rt};
  amrex::Real species_2_gamma{5. / 3._rt};
  amrex::Real species_2_mass{1.67262158e-24_rt};
  amrex::Real species_3_gamma{5. / 3._rt};
  amrex::Real species_3_mass{1.67353251819e-24_rt};
  amrex::Real species_4_gamma{5. / 3._rt};
  amrex::Real species_4_mass{1.67444345638e-24_rt};
  amrex::Real species_5_gamma{5. / 3._rt};
  amrex::Real species_5_mass{3.34512158e-24_rt};
  amrex::Real species_6_gamma{5. / 3._rt};
  amrex::Real species_6_mass{3.34603251819e-24_rt};
  amrex::Real species_7_gamma{1.4_rt};
  amrex::Real species_7_mass{3.34615409819e-24_rt};
  amrex::Real species_8_gamma{5. / 3._rt};
  amrex::Real species_8_mass{3.34694345638e-24_rt};
  amrex::Real species_9_gamma{1.4_rt};
  amrex::Real species_9_mass{3.34706503638e-24_rt};
  amrex::Real species_10_gamma{1.4_rt};
  amrex::Real species_10_mass{5.01865409819e-24_rt};
  amrex::Real species_11_gamma{1.4_rt};
  amrex::Real species_11_mass{5.01956503638e-24_rt};
  amrex::Real species_12_gamma{5. / 3._rt};
  amrex::Real species_12_mass{6.69024316e-24_rt};
  amrex::Real species_13_gamma{5. / 3._rt};
  amrex::Real species_13_mass{6.69115409819e-24_rt};
  amrex::Real species_14_gamma{5. / 3._rt};
  amrex::Real species_14_mass{6.69206503638e-24_rt};
  std::string species_1_name{"elec"};
  std::string species_2_name{"hp"};
  std::string species_3_name{"h"};
  std::string species_4_name{"hm"};
  std::string species_5_name{"dp"};
  std::string species_6_name{"d"};
  std::string species_7_name{"h2p"};
  std::string species_8_name{"dm"};
  std::string species_9_name{"h2"};
  std::string species_10_name{"hdp"};
  std::string species_11_name{"hd"};
  std::string species_12_name{"hepp"};
  std::string species_13_name{"hep"};
  std::string species_14_name{"he"};
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
  amrex::Real small_x{1.e-100_rt};
  amrex::Real redshift{30e0_rt};
};

struct extern_t {
  eos_param_t eos;
  integrator_param_t integrator;
  network_param_t network;
};

#endif
