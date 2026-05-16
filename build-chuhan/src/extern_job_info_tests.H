jobInfoFile << (eos_rp::eos_gamma == 5.e0 / 3.e0_rt ? "    " : "[*] ")
            << "eos.eos_gamma = " << eos_rp::eos_gamma << std::endl;
jobInfoFile << (eos_rp::eos_assume_neutral == true ? "    " : "[*] ")
            << "eos.eos_assume_neutral = " << eos_rp::eos_assume_neutral
            << std::endl;
jobInfoFile << (network_rp::small_x == 1.e-30_rt ? "    " : "[*] ")
            << "network.small_x = " << network_rp::small_x << std::endl;
jobInfoFile << (network_rp::use_tables == false ? "    " : "[*] ")
            << "network.use_tables = " << network_rp::use_tables << std::endl;
jobInfoFile << (network_rp::use_c12ag_deboer17 == false ? "    " : "[*] ")
            << "network.use_c12ag_deboer17 = " << network_rp::use_c12ag_deboer17
            << std::endl;
jobInfoFile << (integrator_rp::X_reject_buffer == 1.0_rt ? "    " : "[*] ")
            << "integrator.X_reject_buffer = " << integrator_rp::X_reject_buffer
            << std::endl;
jobInfoFile << (integrator_rp::call_eos_in_rhs == true ? "    " : "[*] ")
            << "integrator.call_eos_in_rhs = " << integrator_rp::call_eos_in_rhs
            << std::endl;
jobInfoFile << (integrator_rp::integrate_energy == true ? "    " : "[*] ")
            << "integrator.integrate_energy = "
            << integrator_rp::integrate_energy << std::endl;
jobInfoFile << (integrator_rp::jacobian == 1 ? "    " : "[*] ")
            << "integrator.jacobian = " << integrator_rp::jacobian << std::endl;
jobInfoFile << (integrator_rp::burner_verbose == false ? "    " : "[*] ")
            << "integrator.burner_verbose = " << integrator_rp::burner_verbose
            << std::endl;
jobInfoFile << (integrator_rp::rtol_spec == 1.e-12_rt ? "    " : "[*] ")
            << "integrator.rtol_spec = " << integrator_rp::rtol_spec
            << std::endl;
jobInfoFile << (integrator_rp::rtol_enuc == 1.e-6_rt ? "    " : "[*] ")
            << "integrator.rtol_enuc = " << integrator_rp::rtol_enuc
            << std::endl;
jobInfoFile << (integrator_rp::atol_spec == 1.e-8_rt ? "    " : "[*] ")
            << "integrator.atol_spec = " << integrator_rp::atol_spec
            << std::endl;
jobInfoFile << (integrator_rp::atol_enuc == 1.e-6_rt ? "    " : "[*] ")
            << "integrator.atol_enuc = " << integrator_rp::atol_enuc
            << std::endl;
jobInfoFile << (integrator_rp::renormalize_abundances == false ? "    "
                                                               : "[*] ")
            << "integrator.renormalize_abundances = "
            << integrator_rp::renormalize_abundances << std::endl;
jobInfoFile << (integrator_rp::SMALL_X_SAFE == 1.0e-30_rt ? "    " : "[*] ")
            << "integrator.SMALL_X_SAFE = " << integrator_rp::SMALL_X_SAFE
            << std::endl;
jobInfoFile << (integrator_rp::MAX_TEMP == 1.0e11_rt ? "    " : "[*] ")
            << "integrator.MAX_TEMP = " << integrator_rp::MAX_TEMP << std::endl;
jobInfoFile << (integrator_rp::react_boost == -1.e0_rt ? "    " : "[*] ")
            << "integrator.react_boost = " << integrator_rp::react_boost
            << std::endl;
jobInfoFile << (integrator_rp::ode_max_steps == 150000 ? "    " : "[*] ")
            << "integrator.ode_max_steps = " << integrator_rp::ode_max_steps
            << std::endl;
jobInfoFile << (integrator_rp::ode_max_dt == 1.e30_rt ? "    " : "[*] ")
            << "integrator.ode_max_dt = " << integrator_rp::ode_max_dt
            << std::endl;
jobInfoFile << (integrator_rp::use_jacobian_caching == true ? "    " : "[*] ")
            << "integrator.use_jacobian_caching = "
            << integrator_rp::use_jacobian_caching << std::endl;
jobInfoFile << (integrator_rp::nonaka_i == 0 ? "    " : "[*] ")
            << "integrator.nonaka_i = " << integrator_rp::nonaka_i << std::endl;
jobInfoFile << (integrator_rp::nonaka_j == 0 ? "    " : "[*] ")
            << "integrator.nonaka_j = " << integrator_rp::nonaka_j << std::endl;
jobInfoFile << (integrator_rp::nonaka_k == 0 ? "    " : "[*] ")
            << "integrator.nonaka_k = " << integrator_rp::nonaka_k << std::endl;
jobInfoFile << (integrator_rp::nonaka_level == 0 ? "    " : "[*] ")
            << "integrator.nonaka_level = " << integrator_rp::nonaka_level
            << std::endl;
jobInfoFile << (integrator_rp::nonaka_file == "nonaka_plot.dat" ? "    "
                                                                : "[*] ")
            << "integrator.nonaka_file = " << integrator_rp::nonaka_file
            << std::endl;
jobInfoFile << (integrator_rp::use_burn_retry == false ? "    " : "[*] ")
            << "integrator.use_burn_retry = " << integrator_rp::use_burn_retry
            << std::endl;
jobInfoFile << (integrator_rp::retry_swap_jacobian == true ? "    " : "[*] ")
            << "integrator.retry_swap_jacobian = "
            << integrator_rp::retry_swap_jacobian << std::endl;
jobInfoFile << (integrator_rp::retry_rtol_spec == -1_rt ? "    " : "[*] ")
            << "integrator.retry_rtol_spec = " << integrator_rp::retry_rtol_spec
            << std::endl;
jobInfoFile << (integrator_rp::retry_rtol_enuc == -1_rt ? "    " : "[*] ")
            << "integrator.retry_rtol_enuc = " << integrator_rp::retry_rtol_enuc
            << std::endl;
jobInfoFile << (integrator_rp::retry_atol_spec == -1_rt ? "    " : "[*] ")
            << "integrator.retry_atol_spec = " << integrator_rp::retry_atol_spec
            << std::endl;
jobInfoFile << (integrator_rp::retry_atol_enuc == -1_rt ? "    " : "[*] ")
            << "integrator.retry_atol_enuc = " << integrator_rp::retry_atol_enuc
            << std::endl;
jobInfoFile << (integrator_rp::do_species_clip == true ? "    " : "[*] ")
            << "integrator.do_species_clip = " << integrator_rp::do_species_clip
            << std::endl;
jobInfoFile << (integrator_rp::use_number_densities == false ? "    " : "[*] ")
            << "integrator.use_number_densities = "
            << integrator_rp::use_number_densities << std::endl;
jobInfoFile << (integrator_rp::subtract_internal_energy == true ? "    "
                                                                : "[*] ")
            << "integrator.subtract_internal_energy = "
            << integrator_rp::subtract_internal_energy << std::endl;
jobInfoFile << (integrator_rp::scale_system == false ? "    " : "[*] ")
            << "integrator.scale_system = " << integrator_rp::scale_system
            << std::endl;
jobInfoFile << (integrator_rp::nse_deriv_dt_factor == 0.05_rt ? "    " : "[*] ")
            << "integrator.nse_deriv_dt_factor = "
            << integrator_rp::nse_deriv_dt_factor << std::endl;
jobInfoFile << (integrator_rp::nse_include_enu_weak == true ? "    " : "[*] ")
            << "integrator.nse_include_enu_weak = "
            << integrator_rp::nse_include_enu_weak << std::endl;
jobInfoFile << (integrator_rp::linalg_do_pivoting == true ? "    " : "[*] ")
            << "integrator.linalg_do_pivoting = "
            << integrator_rp::linalg_do_pivoting << std::endl;
