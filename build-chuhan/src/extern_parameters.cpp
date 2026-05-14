#include <extern_parameters.H>
#include <AMReX_ParmParse.H>

#include <AMReX_REAL.H>

  namespace eos_rp {
    AMREX_GPU_MANAGED amrex::Real eos_gamma;
    AMREX_GPU_MANAGED bool eos_assume_neutral;
  }
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
  }
  namespace network_rp {
    AMREX_GPU_MANAGED amrex::Real small_x;
    AMREX_GPU_MANAGED bool use_tables;
    AMREX_GPU_MANAGED bool use_c12ag_deboer17;
  }

  extern_t init_extern_parameters() {
    using namespace amrex;

    extern_t params;

    // get the value from the inputs file
    {
      amrex::ParmParse pp("eos");
      eos_rp::eos_gamma = 5.e0/3.e0_rt;
      pp.query("eos_gamma", eos_rp::eos_gamma);
      pp.query("eos_gamma", params.eos.eos_gamma);

      eos_rp::eos_assume_neutral = true;
      pp.query("eos_assume_neutral", eos_rp::eos_assume_neutral);
      pp.query("eos_assume_neutral", params.eos.eos_assume_neutral);

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
      pp.query("renormalize_abundances", params.integrator.renormalize_abundances);

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
      pp.query("subtract_internal_energy", integrator_rp::subtract_internal_energy);
      pp.query("subtract_internal_energy", params.integrator.subtract_internal_energy);

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
      network_rp::small_x = 1.e-30_rt;
      pp.query("small_x", network_rp::small_x);
      pp.query("small_x", params.network.small_x);

      network_rp::use_tables = false;
      pp.query("use_tables", network_rp::use_tables);
      pp.query("use_tables", params.network.use_tables);

      network_rp::use_c12ag_deboer17 = false;
      pp.query("use_c12ag_deboer17", network_rp::use_c12ag_deboer17);
      pp.query("use_c12ag_deboer17", params.network.use_c12ag_deboer17);

    }
    return params;

  }
