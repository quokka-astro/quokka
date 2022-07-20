// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
  static constexpr bool is_hydro_enabled = false;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_mhd_enabled = false;
  static constexpr bool is_primordial_chem_enabled = false;
  static constexpr bool is_metalicity_enabled = false;
};
