
#ifndef EXTERNAL_PARAMETERS_H
#define EXTERNAL_PARAMETERS_H
#include <AMReX_BLFort.H>
#include <AMReX_REAL.H>

#include <extern_type.H>

extern_t init_extern_parameters();

namespace eos_rp {
extern AMREX_GPU_MANAGED amrex::Real eos_gamma_default;
extern std::string species_1_name;
extern AMREX_GPU_MANAGED amrex::Real species_1_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_1_mass;
extern std::string species_2_name;
extern AMREX_GPU_MANAGED amrex::Real species_2_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_2_mass;
extern std::string species_3_name;
extern AMREX_GPU_MANAGED amrex::Real species_3_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_3_mass;
extern std::string species_4_name;
extern AMREX_GPU_MANAGED amrex::Real species_4_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_4_mass;
extern std::string species_5_name;
extern AMREX_GPU_MANAGED amrex::Real species_5_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_5_mass;
extern std::string species_6_name;
extern AMREX_GPU_MANAGED amrex::Real species_6_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_6_mass;
extern std::string species_7_name;
extern AMREX_GPU_MANAGED amrex::Real species_7_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_7_mass;
extern std::string species_8_name;
extern AMREX_GPU_MANAGED amrex::Real species_8_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_8_mass;
extern std::string species_9_name;
extern AMREX_GPU_MANAGED amrex::Real species_9_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_9_mass;
extern std::string species_10_name;
extern AMREX_GPU_MANAGED amrex::Real species_10_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_10_mass;
extern std::string species_11_name;
extern AMREX_GPU_MANAGED amrex::Real species_11_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_11_mass;
extern std::string species_12_name;
extern AMREX_GPU_MANAGED amrex::Real species_12_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_12_mass;
extern std::string species_13_name;
extern AMREX_GPU_MANAGED amrex::Real species_13_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_13_mass;
extern std::string species_14_name;
extern AMREX_GPU_MANAGED amrex::Real species_14_gamma;
extern AMREX_GPU_MANAGED amrex::Real species_14_mass;
} // namespace eos_rp
namespace integrator_rp {
extern AMREX_GPU_MANAGED amrex::Real X_reject_buffer;
extern AMREX_GPU_MANAGED bool call_eos_in_rhs;
extern AMREX_GPU_MANAGED bool integrate_energy;
extern AMREX_GPU_MANAGED int jacobian;
extern AMREX_GPU_MANAGED bool burner_verbose;
extern AMREX_GPU_MANAGED amrex::Real rtol_spec;
extern AMREX_GPU_MANAGED amrex::Real rtol_enuc;
extern AMREX_GPU_MANAGED amrex::Real atol_spec;
extern AMREX_GPU_MANAGED amrex::Real atol_enuc;
extern AMREX_GPU_MANAGED bool renormalize_abundances;
extern AMREX_GPU_MANAGED amrex::Real SMALL_X_SAFE;
extern AMREX_GPU_MANAGED amrex::Real MAX_TEMP;
extern AMREX_GPU_MANAGED amrex::Real react_boost;
extern AMREX_GPU_MANAGED int ode_max_steps;
extern AMREX_GPU_MANAGED amrex::Real ode_max_dt;
extern AMREX_GPU_MANAGED bool use_jacobian_caching;
extern AMREX_GPU_MANAGED int nonaka_i;
extern AMREX_GPU_MANAGED int nonaka_j;
extern AMREX_GPU_MANAGED int nonaka_k;
extern AMREX_GPU_MANAGED int nonaka_level;
extern std::string nonaka_file;
extern AMREX_GPU_MANAGED bool use_burn_retry;
extern AMREX_GPU_MANAGED bool retry_swap_jacobian;
extern AMREX_GPU_MANAGED amrex::Real retry_rtol_spec;
extern AMREX_GPU_MANAGED amrex::Real retry_rtol_enuc;
extern AMREX_GPU_MANAGED amrex::Real retry_atol_spec;
extern AMREX_GPU_MANAGED amrex::Real retry_atol_enuc;
extern AMREX_GPU_MANAGED bool do_species_clip;
extern AMREX_GPU_MANAGED bool use_number_densities;
extern AMREX_GPU_MANAGED bool subtract_internal_energy;
extern AMREX_GPU_MANAGED bool scale_system;
extern AMREX_GPU_MANAGED amrex::Real nse_deriv_dt_factor;
extern AMREX_GPU_MANAGED bool nse_include_enu_weak;
extern AMREX_GPU_MANAGED bool linalg_do_pivoting;
} // namespace integrator_rp
namespace network_rp {
extern AMREX_GPU_MANAGED amrex::Real small_x;
extern AMREX_GPU_MANAGED amrex::Real redshift;
extern AMREX_GPU_MANAGED bool use_tables;
extern AMREX_GPU_MANAGED bool use_c12ag_deboer17;
} // namespace network_rp

#endif
