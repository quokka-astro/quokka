#include <AMReX_ParmParse.H>
#include <extern_parameters.H>

#include <AMReX_REAL.H>

namespace eos_rp {
AMREX_GPU_MANAGED amrex::Real eos_gamma_default;
std::string species_1_name;
AMREX_GPU_MANAGED amrex::Real species_1_gamma;
AMREX_GPU_MANAGED amrex::Real species_1_mass;
std::string species_2_name;
AMREX_GPU_MANAGED amrex::Real species_2_gamma;
AMREX_GPU_MANAGED amrex::Real species_2_mass;
std::string species_3_name;
AMREX_GPU_MANAGED amrex::Real species_3_gamma;
AMREX_GPU_MANAGED amrex::Real species_3_mass;
std::string species_4_name;
AMREX_GPU_MANAGED amrex::Real species_4_gamma;
AMREX_GPU_MANAGED amrex::Real species_4_mass;
std::string species_5_name;
AMREX_GPU_MANAGED amrex::Real species_5_gamma;
AMREX_GPU_MANAGED amrex::Real species_5_mass;
std::string species_6_name;
AMREX_GPU_MANAGED amrex::Real species_6_gamma;
AMREX_GPU_MANAGED amrex::Real species_6_mass;
std::string species_7_name;
AMREX_GPU_MANAGED amrex::Real species_7_gamma;
AMREX_GPU_MANAGED amrex::Real species_7_mass;
std::string species_8_name;
AMREX_GPU_MANAGED amrex::Real species_8_gamma;
AMREX_GPU_MANAGED amrex::Real species_8_mass;
std::string species_9_name;
AMREX_GPU_MANAGED amrex::Real species_9_gamma;
AMREX_GPU_MANAGED amrex::Real species_9_mass;
std::string species_10_name;
AMREX_GPU_MANAGED amrex::Real species_10_gamma;
AMREX_GPU_MANAGED amrex::Real species_10_mass;
std::string species_11_name;
AMREX_GPU_MANAGED amrex::Real species_11_gamma;
AMREX_GPU_MANAGED amrex::Real species_11_mass;
std::string species_12_name;
AMREX_GPU_MANAGED amrex::Real species_12_gamma;
AMREX_GPU_MANAGED amrex::Real species_12_mass;
std::string species_13_name;
AMREX_GPU_MANAGED amrex::Real species_13_gamma;
AMREX_GPU_MANAGED amrex::Real species_13_mass;
std::string species_14_name;
AMREX_GPU_MANAGED amrex::Real species_14_gamma;
AMREX_GPU_MANAGED amrex::Real species_14_mass;
} // namespace eos_rp
namespace integrator_rp {
AMREX_GPU_MANAGED amrex::Real X_reject_buffer;
AMREX_GPU_MANAGED bool call_eos_in_rhs;
AMREX_GPU_MANAGED bool integrate_energy;
AMREX_GPU_MANAGED int jacobian;
AMREX_GPU_MANAGED bool burner_verbose;
AMREX_GPU_MANAGED amrex::Real rtol_spec;
AMREX_GPU_MANAGED amrex::Real rtol_enuc;
AMREX_GPU_MANAGED amrex::Real atol_spec;
AMREX_GPU_MANAGED amrex::Real atol_enuc;
AMREX_GPU_MANAGED bool renormalize_abundances;
AMREX_GPU_MANAGED amrex::Real SMALL_X_SAFE;
AMREX_GPU_MANAGED amrex::Real MAX_TEMP;
AMREX_GPU_MANAGED amrex::Real react_boost;
AMREX_GPU_MANAGED int ode_max_steps;
AMREX_GPU_MANAGED amrex::Real ode_max_dt;
AMREX_GPU_MANAGED bool use_jacobian_caching;
AMREX_GPU_MANAGED int nonaka_i;
AMREX_GPU_MANAGED int nonaka_j;
AMREX_GPU_MANAGED int nonaka_k;
AMREX_GPU_MANAGED int nonaka_level;
std::string nonaka_file;
AMREX_GPU_MANAGED bool use_burn_retry;
AMREX_GPU_MANAGED bool retry_swap_jacobian;
AMREX_GPU_MANAGED amrex::Real retry_rtol_spec;
AMREX_GPU_MANAGED amrex::Real retry_rtol_enuc;
AMREX_GPU_MANAGED amrex::Real retry_atol_spec;
AMREX_GPU_MANAGED amrex::Real retry_atol_enuc;
AMREX_GPU_MANAGED bool do_species_clip;
AMREX_GPU_MANAGED bool use_number_densities;
AMREX_GPU_MANAGED bool subtract_internal_energy;
AMREX_GPU_MANAGED bool scale_system;
AMREX_GPU_MANAGED amrex::Real nse_deriv_dt_factor;
AMREX_GPU_MANAGED bool nse_include_enu_weak;
AMREX_GPU_MANAGED bool linalg_do_pivoting;
} // namespace integrator_rp
namespace network_rp {
AMREX_GPU_MANAGED amrex::Real small_x;
AMREX_GPU_MANAGED amrex::Real redshift;
AMREX_GPU_MANAGED bool use_tables;
AMREX_GPU_MANAGED bool use_c12ag_deboer17;
} // namespace network_rp

extern_t init_extern_parameters() {
  using namespace amrex;

  extern_t params;

  // get the value from the inputs file
  {
    amrex::ParmParse pp("eos");
    eos_rp::eos_gamma_default = 1.4_rt;
    pp.query("eos_gamma_default", eos_rp::eos_gamma_default);
    pp.query("eos_gamma_default", params.eos.eos_gamma_default);

    eos_rp::species_1_name = "elec";
    pp.query("species_1_name", eos_rp::species_1_name);
    pp.query("species_1_name", params.eos.species_1_name);

    eos_rp::species_1_gamma = 5. / 3._rt;
    pp.query("species_1_gamma", eos_rp::species_1_gamma);
    pp.query("species_1_gamma", params.eos.species_1_gamma);

    eos_rp::species_1_mass = 9.10938188e-28_rt;
    pp.query("species_1_mass", eos_rp::species_1_mass);
    pp.query("species_1_mass", params.eos.species_1_mass);

    eos_rp::species_2_name = "hp";
    pp.query("species_2_name", eos_rp::species_2_name);
    pp.query("species_2_name", params.eos.species_2_name);

    eos_rp::species_2_gamma = 5. / 3._rt;
    pp.query("species_2_gamma", eos_rp::species_2_gamma);
    pp.query("species_2_gamma", params.eos.species_2_gamma);

    eos_rp::species_2_mass = 1.67262158e-24_rt;
    pp.query("species_2_mass", eos_rp::species_2_mass);
    pp.query("species_2_mass", params.eos.species_2_mass);

    eos_rp::species_3_name = "h";
    pp.query("species_3_name", eos_rp::species_3_name);
    pp.query("species_3_name", params.eos.species_3_name);

    eos_rp::species_3_gamma = 5. / 3._rt;
    pp.query("species_3_gamma", eos_rp::species_3_gamma);
    pp.query("species_3_gamma", params.eos.species_3_gamma);

    eos_rp::species_3_mass = 1.67353251819e-24_rt;
    pp.query("species_3_mass", eos_rp::species_3_mass);
    pp.query("species_3_mass", params.eos.species_3_mass);

    eos_rp::species_4_name = "hm";
    pp.query("species_4_name", eos_rp::species_4_name);
    pp.query("species_4_name", params.eos.species_4_name);

    eos_rp::species_4_gamma = 5. / 3._rt;
    pp.query("species_4_gamma", eos_rp::species_4_gamma);
    pp.query("species_4_gamma", params.eos.species_4_gamma);

    eos_rp::species_4_mass = 1.67444345638e-24_rt;
    pp.query("species_4_mass", eos_rp::species_4_mass);
    pp.query("species_4_mass", params.eos.species_4_mass);

    eos_rp::species_5_name = "dp";
    pp.query("species_5_name", eos_rp::species_5_name);
    pp.query("species_5_name", params.eos.species_5_name);

    eos_rp::species_5_gamma = 5. / 3._rt;
    pp.query("species_5_gamma", eos_rp::species_5_gamma);
    pp.query("species_5_gamma", params.eos.species_5_gamma);

    eos_rp::species_5_mass = 3.34512158e-24_rt;
    pp.query("species_5_mass", eos_rp::species_5_mass);
    pp.query("species_5_mass", params.eos.species_5_mass);

    eos_rp::species_6_name = "d";
    pp.query("species_6_name", eos_rp::species_6_name);
    pp.query("species_6_name", params.eos.species_6_name);

    eos_rp::species_6_gamma = 5. / 3._rt;
    pp.query("species_6_gamma", eos_rp::species_6_gamma);
    pp.query("species_6_gamma", params.eos.species_6_gamma);

    eos_rp::species_6_mass = 3.34603251819e-24_rt;
    pp.query("species_6_mass", eos_rp::species_6_mass);
    pp.query("species_6_mass", params.eos.species_6_mass);

    eos_rp::species_7_name = "h2p";
    pp.query("species_7_name", eos_rp::species_7_name);
    pp.query("species_7_name", params.eos.species_7_name);

    eos_rp::species_7_gamma = 1.4_rt;
    pp.query("species_7_gamma", eos_rp::species_7_gamma);
    pp.query("species_7_gamma", params.eos.species_7_gamma);

    eos_rp::species_7_mass = 3.34615409819e-24_rt;
    pp.query("species_7_mass", eos_rp::species_7_mass);
    pp.query("species_7_mass", params.eos.species_7_mass);

    eos_rp::species_8_name = "dm";
    pp.query("species_8_name", eos_rp::species_8_name);
    pp.query("species_8_name", params.eos.species_8_name);

    eos_rp::species_8_gamma = 5. / 3._rt;
    pp.query("species_8_gamma", eos_rp::species_8_gamma);
    pp.query("species_8_gamma", params.eos.species_8_gamma);

    eos_rp::species_8_mass = 3.34694345638e-24_rt;
    pp.query("species_8_mass", eos_rp::species_8_mass);
    pp.query("species_8_mass", params.eos.species_8_mass);

    eos_rp::species_9_name = "h2";
    pp.query("species_9_name", eos_rp::species_9_name);
    pp.query("species_9_name", params.eos.species_9_name);

    eos_rp::species_9_gamma = 1.4_rt;
    pp.query("species_9_gamma", eos_rp::species_9_gamma);
    pp.query("species_9_gamma", params.eos.species_9_gamma);

    eos_rp::species_9_mass = 3.34706503638e-24_rt;
    pp.query("species_9_mass", eos_rp::species_9_mass);
    pp.query("species_9_mass", params.eos.species_9_mass);

    eos_rp::species_10_name = "hdp";
    pp.query("species_10_name", eos_rp::species_10_name);
    pp.query("species_10_name", params.eos.species_10_name);

    eos_rp::species_10_gamma = 1.4_rt;
    pp.query("species_10_gamma", eos_rp::species_10_gamma);
    pp.query("species_10_gamma", params.eos.species_10_gamma);

    eos_rp::species_10_mass = 5.01865409819e-24_rt;
    pp.query("species_10_mass", eos_rp::species_10_mass);
    pp.query("species_10_mass", params.eos.species_10_mass);

    eos_rp::species_11_name = "hd";
    pp.query("species_11_name", eos_rp::species_11_name);
    pp.query("species_11_name", params.eos.species_11_name);

    eos_rp::species_11_gamma = 1.4_rt;
    pp.query("species_11_gamma", eos_rp::species_11_gamma);
    pp.query("species_11_gamma", params.eos.species_11_gamma);

    eos_rp::species_11_mass = 5.01956503638e-24_rt;
    pp.query("species_11_mass", eos_rp::species_11_mass);
    pp.query("species_11_mass", params.eos.species_11_mass);

    eos_rp::species_12_name = "hepp";
    pp.query("species_12_name", eos_rp::species_12_name);
    pp.query("species_12_name", params.eos.species_12_name);

    eos_rp::species_12_gamma = 5. / 3._rt;
    pp.query("species_12_gamma", eos_rp::species_12_gamma);
    pp.query("species_12_gamma", params.eos.species_12_gamma);

    eos_rp::species_12_mass = 6.69024316e-24_rt;
    pp.query("species_12_mass", eos_rp::species_12_mass);
    pp.query("species_12_mass", params.eos.species_12_mass);

    eos_rp::species_13_name = "hep";
    pp.query("species_13_name", eos_rp::species_13_name);
    pp.query("species_13_name", params.eos.species_13_name);

    eos_rp::species_13_gamma = 5. / 3._rt;
    pp.query("species_13_gamma", eos_rp::species_13_gamma);
    pp.query("species_13_gamma", params.eos.species_13_gamma);

    eos_rp::species_13_mass = 6.69115409819e-24_rt;
    pp.query("species_13_mass", eos_rp::species_13_mass);
    pp.query("species_13_mass", params.eos.species_13_mass);

    eos_rp::species_14_name = "he";
    pp.query("species_14_name", eos_rp::species_14_name);
    pp.query("species_14_name", params.eos.species_14_name);

    eos_rp::species_14_gamma = 5. / 3._rt;
    pp.query("species_14_gamma", eos_rp::species_14_gamma);
    pp.query("species_14_gamma", params.eos.species_14_gamma);

    eos_rp::species_14_mass = 6.69206503638e-24_rt;
    pp.query("species_14_mass", eos_rp::species_14_mass);
    pp.query("species_14_mass", params.eos.species_14_mass);
  }
  {
    amrex::ParmParse pp("integrator");
    integrator_rp::X_reject_buffer = 1.0_rt;
    pp.query("X_reject_buffer", integrator_rp::X_reject_buffer);
    pp.query("X_reject_buffer", params.integrator.X_reject_buffer);

    integrator_rp::call_eos_in_rhs = true;
    pp.query("call_eos_in_rhs", integrator_rp::call_eos_in_rhs);
    pp.query("call_eos_in_rhs", params.integrator.call_eos_in_rhs);

    integrator_rp::integrate_energy = true;
    pp.query("integrate_energy", integrator_rp::integrate_energy);
    pp.query("integrate_energy", params.integrator.integrate_energy);

    integrator_rp::jacobian = 1;
    pp.query("jacobian", integrator_rp::jacobian);
    pp.query("jacobian", params.integrator.jacobian);

    integrator_rp::burner_verbose = false;
    pp.query("burner_verbose", integrator_rp::burner_verbose);
    pp.query("burner_verbose", params.integrator.burner_verbose);

    integrator_rp::rtol_spec = 1.e-12_rt;
    pp.query("rtol_spec", integrator_rp::rtol_spec);
    pp.query("rtol_spec", params.integrator.rtol_spec);

    integrator_rp::rtol_enuc = 1.e-6_rt;
    pp.query("rtol_enuc", integrator_rp::rtol_enuc);
    pp.query("rtol_enuc", params.integrator.rtol_enuc);

    integrator_rp::atol_spec = 1.e-8_rt;
    pp.query("atol_spec", integrator_rp::atol_spec);
    pp.query("atol_spec", params.integrator.atol_spec);

    integrator_rp::atol_enuc = 1.e-6_rt;
    pp.query("atol_enuc", integrator_rp::atol_enuc);
    pp.query("atol_enuc", params.integrator.atol_enuc);

    integrator_rp::renormalize_abundances = false;
    pp.query("renormalize_abundances", integrator_rp::renormalize_abundances);
    pp.query("renormalize_abundances",
             params.integrator.renormalize_abundances);

    integrator_rp::SMALL_X_SAFE = 1.0e-30_rt;
    pp.query("SMALL_X_SAFE", integrator_rp::SMALL_X_SAFE);
    pp.query("SMALL_X_SAFE", params.integrator.SMALL_X_SAFE);

    integrator_rp::MAX_TEMP = 1.0e11_rt;
    pp.query("MAX_TEMP", integrator_rp::MAX_TEMP);
    pp.query("MAX_TEMP", params.integrator.MAX_TEMP);

    integrator_rp::react_boost = -1.e0_rt;
    pp.query("react_boost", integrator_rp::react_boost);
    pp.query("react_boost", params.integrator.react_boost);

    integrator_rp::ode_max_steps = 150000;
    pp.query("ode_max_steps", integrator_rp::ode_max_steps);
    pp.query("ode_max_steps", params.integrator.ode_max_steps);

    integrator_rp::ode_max_dt = 1.e30_rt;
    pp.query("ode_max_dt", integrator_rp::ode_max_dt);
    pp.query("ode_max_dt", params.integrator.ode_max_dt);

    integrator_rp::use_jacobian_caching = true;
    pp.query("use_jacobian_caching", integrator_rp::use_jacobian_caching);
    pp.query("use_jacobian_caching", params.integrator.use_jacobian_caching);

    integrator_rp::nonaka_i = 0;
    pp.query("nonaka_i", integrator_rp::nonaka_i);
    pp.query("nonaka_i", params.integrator.nonaka_i);

    integrator_rp::nonaka_j = 0;
    pp.query("nonaka_j", integrator_rp::nonaka_j);
    pp.query("nonaka_j", params.integrator.nonaka_j);

    integrator_rp::nonaka_k = 0;
    pp.query("nonaka_k", integrator_rp::nonaka_k);
    pp.query("nonaka_k", params.integrator.nonaka_k);

    integrator_rp::nonaka_level = 0;
    pp.query("nonaka_level", integrator_rp::nonaka_level);
    pp.query("nonaka_level", params.integrator.nonaka_level);

    integrator_rp::nonaka_file = "nonaka_plot.dat";
    pp.query("nonaka_file", integrator_rp::nonaka_file);
    pp.query("nonaka_file", params.integrator.nonaka_file);

    integrator_rp::use_burn_retry = false;
    pp.query("use_burn_retry", integrator_rp::use_burn_retry);
    pp.query("use_burn_retry", params.integrator.use_burn_retry);

    integrator_rp::retry_swap_jacobian = true;
    pp.query("retry_swap_jacobian", integrator_rp::retry_swap_jacobian);
    pp.query("retry_swap_jacobian", params.integrator.retry_swap_jacobian);

    integrator_rp::retry_rtol_spec = -1_rt;
    pp.query("retry_rtol_spec", integrator_rp::retry_rtol_spec);
    pp.query("retry_rtol_spec", params.integrator.retry_rtol_spec);

    integrator_rp::retry_rtol_enuc = -1_rt;
    pp.query("retry_rtol_enuc", integrator_rp::retry_rtol_enuc);
    pp.query("retry_rtol_enuc", params.integrator.retry_rtol_enuc);

    integrator_rp::retry_atol_spec = -1_rt;
    pp.query("retry_atol_spec", integrator_rp::retry_atol_spec);
    pp.query("retry_atol_spec", params.integrator.retry_atol_spec);

    integrator_rp::retry_atol_enuc = -1_rt;
    pp.query("retry_atol_enuc", integrator_rp::retry_atol_enuc);
    pp.query("retry_atol_enuc", params.integrator.retry_atol_enuc);

    integrator_rp::do_species_clip = true;
    pp.query("do_species_clip", integrator_rp::do_species_clip);
    pp.query("do_species_clip", params.integrator.do_species_clip);

    integrator_rp::use_number_densities = false;
    pp.query("use_number_densities", integrator_rp::use_number_densities);
    pp.query("use_number_densities", params.integrator.use_number_densities);

    integrator_rp::subtract_internal_energy = true;
    pp.query("subtract_internal_energy",
             integrator_rp::subtract_internal_energy);
    pp.query("subtract_internal_energy",
             params.integrator.subtract_internal_energy);

    integrator_rp::scale_system = false;
    pp.query("scale_system", integrator_rp::scale_system);
    pp.query("scale_system", params.integrator.scale_system);

    integrator_rp::nse_deriv_dt_factor = 0.05_rt;
    pp.query("nse_deriv_dt_factor", integrator_rp::nse_deriv_dt_factor);
    pp.query("nse_deriv_dt_factor", params.integrator.nse_deriv_dt_factor);

    integrator_rp::nse_include_enu_weak = true;
    pp.query("nse_include_enu_weak", integrator_rp::nse_include_enu_weak);
    pp.query("nse_include_enu_weak", params.integrator.nse_include_enu_weak);

    integrator_rp::linalg_do_pivoting = true;
    pp.query("linalg_do_pivoting", integrator_rp::linalg_do_pivoting);
    pp.query("linalg_do_pivoting", params.integrator.linalg_do_pivoting);
  }
  {
    amrex::ParmParse pp("network");
    network_rp::small_x = 1.e-100_rt;
    pp.query("small_x", network_rp::small_x);
    pp.query("small_x", params.network.small_x);

    network_rp::redshift = 30e0_rt;
    pp.query("redshift", network_rp::redshift);
    pp.query("redshift", params.network.redshift);

    network_rp::use_tables = false;
    pp.query("use_tables", network_rp::use_tables);
    pp.query("use_tables", params.network.use_tables);

    network_rp::use_c12ag_deboer17 = false;
    pp.query("use_c12ag_deboer17", network_rp::use_c12ag_deboer17);
    pp.query("use_c12ag_deboer17", params.network.use_c12ag_deboer17);
  }
  return params;
}
