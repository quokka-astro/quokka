#ifndef PARTICLE_TYPES_HPP_
#define PARTICLE_TYPES_HPP_

#include "AMReX_AmrParticles.H"
#include "AMReX_Enum.H"
#include "AMReX_ParIter.H"
#include "physics_info.hpp"

// Function to create bit flags: bitflag(position) = 2^(position - 1)
// Example: bitflag<1>() = 1, bitflag<2>() = 2, bitflag<3>() = 4, ...
template <unsigned int position> constexpr auto bitflag() -> unsigned int { return 1U << (position - 1U); }

// Particle type flags that can be combined using bitwise OR operation (|).
// Example: To enable both CIC and Rad particles, use:
//   particle_switch = ParticleSwitch::CIC | ParticleSwitch::Rad  (= 0b00000011)
// To check if CIC particles are enabled:
//   if (particle_switch & ParticleSwitch::CIC) { ... }
enum class ParticleSwitch : unsigned int {
	None = 0U,			     // No particles, = 0b0000
	CIC = bitflag<1>(),		     // Cloud-In-Cell (gravitating) particles, = 0b0001
	Rad = bitflag<2>(),		     // Radiation particles, = 0b0010
	CICRad = bitflag<3>(),		     // Combined gravitating-radiating particles, = 0b0100
	StochasticStellarPop = bitflag<4>(), // Stellar population particles, = 0b1000
	Sink = bitflag<5>(),		     // Sink particles, = 0b10000
	Test = bitflag<6>()		     // Test particles with all features enabled, = 0b100000
};

// Enable bitwise operations on the enum class
constexpr auto operator|(ParticleSwitch a, ParticleSwitch b) -> ParticleSwitch
{
	return static_cast<ParticleSwitch>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

constexpr auto operator&(ParticleSwitch flags, ParticleSwitch flag) -> bool
{
	return (static_cast<unsigned int>(flags) & static_cast<unsigned int>(flag)) != 0;
}

// This struct should be specialized by the user application code to configure particle behavior.
// The particle_switch member determines which particle types are enabled using bitwise flags.
// Examples:
// - static constexpr ParticleSwitch particle_switch = ParticleSwitch::None             -> No particles enabled
// - static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC              -> Only CIC particles
// - static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::Rad -> Both CIC and Rad particles
// Examples that will cause a compile error:
// - static constexpr int particle_switch = 1;
// enum class TestEnum : unsigned int {
// 	MISTAKE = 0b00000100U,
// };
// - static constexpr TestEnum particle_switch = TestEnum::MISTAKE;
// - static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | TestEnum::MISTAKE;
template <typename problem_t> struct Particle_Traits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None; // Determines which particle types are enabled using bitwise flags.
};

// Static assertion helper to verify that particle_switch is of the correct type
namespace detail
{
template <typename problem_t> constexpr void verify_particle_switch_type()
{
	// This will fail to compile if particle_switch is not of type ParticleSwitch
	static_assert(std::is_same_v<decltype(Particle_Traits<problem_t>::particle_switch), const ParticleSwitch>,
		      "ERROR: Particle_Traits::particle_switch must be of type ParticleSwitch. "
		      "Use any of the members of ParticleSwitch enum class, or combinations with '|'");
}
} // namespace detail

namespace quokka
{

// Enum class to identify different particle types
enum class ParticleType {
	Rad,		      // Radiation particles
	CIC,		      // Gravitating particles
	CICRad,		      // Gravitating radiation particles
	StochasticStellarPop, // Stellar population particles
	Sink,		      // Sink particles
	Test		      // Test particles with all features enabled
};

// Enum for SN schemes: ThermalOnly, ThermalAndMomentum
AMREX_ENUM(SNScheme,				   // NOLINT
	   SN_thermal_only,			   // pure thermal
	   SN_thermal_or_thermal_momentum,	   // pure thermal (RM<1) or thermal+momentum (RM>=1)
	   SN_thermal_kinetic_or_thermal_momentum, // thermal+kinetic (RM<1) or thermal+momentum (RM>=1)
	   SN_pure_kinetic_or_thermal_momentum	   // pure kinetic (RM<1) or thermal+momentum (RM>=1)
);

//-------------------- Radiation particles --------------------

// Indices for radiation particles (Rad_particles) using AMREX_ENUM for automatic string conversion
AMREX_ENUM(RadParticleRealIdx, // NOLINT
	   birth_time,	       // Time when particle becomes active
	   death_time,	       // Time when particle becomes inactive
	   luminosity	       // Base luminosity component (expanded to luminosity_0, luminosity_1, ... in I/O)
);

// Backward compatibility aliases for existing code
constexpr int RadParticleBirthTimeIdx = static_cast<int>(RadParticleRealIdx::birth_time);
constexpr int RadParticleDeathTimeIdx = static_cast<int>(RadParticleRealIdx::death_time);
constexpr int RadParticleLumIdx = static_cast<int>(RadParticleRealIdx::luminosity); // Base index for luminosity components

// Number of real components for Rad_particles, birth time + death time + radiation groups
template <typename problem_t>
constexpr int RadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 2 + Physics_Traits<problem_t>::nGroups; // birth_time death_time lum1 ... lumN
	} else {
		return 2; // birth_time death_time
	}
}();

// Type definitions for Rad_particles container and iterator
template <typename problem_t> using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>;
template <typename problem_t> using RadParticleIterator = amrex::ParIter<RadParticleRealComps<problem_t>>;

#if AMREX_SPACEDIM == 3

//-------------------- Gravitating particles --------------------

// Indices for gravitating particles (CIC_particles) using AMREX_ENUM for automatic string conversion
AMREX_ENUM(CICParticleRealIdx, // NOLINT
	   mass,	       // Mass of the particle
	   vx,		       // Velocity in x direction
	   vy,		       // Velocity in y direction
	   vz		       // Velocity in z direction
);

// Backward compatibility aliases for existing code
constexpr int CICParticleMassIdx = static_cast<int>(CICParticleRealIdx::mass);
constexpr int CICParticleVxIdx = static_cast<int>(CICParticleRealIdx::vx);
constexpr int CICParticleVyIdx = static_cast<int>(CICParticleRealIdx::vy);
constexpr int CICParticleVzIdx = static_cast<int>(CICParticleRealIdx::vz);

// Number of real components for CIC_particles, mass + 3 velocity components
constexpr int CICParticleRealComps = 4;

// Type definitions for CIC_particles container and iterator
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

//-------------------- Gravitating radiation particles --------------------

// Indices for gravitating radiation particles (CICRad_particles) using AMREX_ENUM for automatic string conversion
AMREX_ENUM(CICRadParticleRealIdx, // NOLINT
	   mass,		  // Mass of the particle
	   vx,			  // Velocity in x direction
	   vy,			  // Velocity in y direction
	   vz,			  // Velocity in z direction
	   birth_time,		  // Time when particle becomes active
	   death_time,		  // Time when particle becomes inactive
	   luminosity		  // Base luminosity component (expanded to luminosity_0, luminosity_1, ... in I/O)
);

// Backward compatibility aliases for existing code
constexpr int CICRadParticleMassIdx = static_cast<int>(CICRadParticleRealIdx::mass);
constexpr int CICRadParticleVxIdx = static_cast<int>(CICRadParticleRealIdx::vx);
constexpr int CICRadParticleVyIdx = static_cast<int>(CICRadParticleRealIdx::vy);
constexpr int CICRadParticleVzIdx = static_cast<int>(CICRadParticleRealIdx::vz);
constexpr int CICRadParticleBirthTimeIdx = static_cast<int>(CICRadParticleRealIdx::birth_time);
constexpr int CICRadParticleDeathTimeIdx = static_cast<int>(CICRadParticleRealIdx::death_time);
constexpr int CICRadParticleLumIdx = static_cast<int>(CICRadParticleRealIdx::luminosity); // Base index for luminosity components

// Number of real components for CICRad_particles, mass + 3 velocity components + birth time + death time + radiation groups
template <typename problem_t>
constexpr int CICRadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass, vx, vy, vz, birth_time, death_time, lum[nGroups]
	} else {
		return 6; // mass, vx, vy, vz, birth_time, death_time
	}
}();

// Type definitions for CICRad_particles container and iterator
template <typename problem_t> using CICRadParticleContainer = amrex::AmrParticleContainer<CICRadParticleRealComps<problem_t>>;
template <typename problem_t> using CICRadParticleIterator = amrex::ParIter<CICRadParticleRealComps<problem_t>>;

//-------------------- Stellar evolution stage enum --------------------

// Enum for particle evolution stages. This is designed to be shared among several particle types. However, not all particle types will use all stages.
// - HighMassNonExploding: high-mass stars (> 9 Msun) that will not explode as supernovae in the end of their lifetime
// - SNProgenitor: singular high-mass stars (> 9 Msun) that will explode as supernovae in the end of their lifetime
// - SNRemnant: Supernova remnant stage
// - LowMassComposite: composite of low-mass stars
// - Removed: marked for removal
enum class StellarEvolutionStage { HighMassNonExploding, SNProgenitor, SNRemnant, LowMassComposite, Removed };

//-------------------- Stellar population particles --------------------

// Indices for StochasticStellarPop_particles using AMREX_ENUM for automatic string conversion
AMREX_ENUM(StochasticStellarPopParticleRealIdx, // NOLINT
	   mass,				// Mass of the particle
	   vx,					// Velocity in x direction
	   vy,					// Velocity in y direction
	   vz,					// Velocity in z direction
	   birth_time,				// Time when particle becomes active
	   death_time,				// Time when particle becomes inactive
	   birth_x,				// Birth position x
	   birth_y,				// Birth position y
	   birth_z,				// Birth position z
	   death_x,				// Death position x
	   death_y,				// Death position y
	   death_z,				// Death position z
	   death_density,			// Density at death
	   mass_at_birth,			// Particle mass at birth
	   luminosity				// Base luminosity component (expanded to luminosity_0, luminosity_1, ... in I/O)
);

// Integer component indices using AMREX_ENUM
AMREX_ENUM(StochasticStellarPopParticleIntIdx, // NOLINT
	   evolution_stage		       // Evolution stage of the particle
);

// Backward compatibility aliases for existing code
constexpr int StochasticStellarPopParticleMassIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::mass);
constexpr int StochasticStellarPopParticleVxIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::vx);
constexpr int StochasticStellarPopParticleVyIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::vy);
constexpr int StochasticStellarPopParticleVzIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::vz);
constexpr int StochasticStellarPopParticleBirthTimeIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::birth_time);
constexpr int StochasticStellarPopParticleDeathTimeIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::death_time);
constexpr int StochasticStellarPopParticleBirthPosXIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::birth_x);
constexpr int StochasticStellarPopParticleBirthPosYIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::birth_y);
constexpr int StochasticStellarPopParticleBirthPosZIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::birth_z);
constexpr int StochasticStellarPopParticleDeathPosXIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::death_x);
constexpr int StochasticStellarPopParticleDeathPosYIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::death_y);
constexpr int StochasticStellarPopParticleDeathPosZIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::death_z);
constexpr int StochasticStellarPopParticleDeathDensityIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::death_density);
constexpr int StochasticStellarPopParticleMassAtBirthIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::mass_at_birth);
constexpr int StochasticStellarPopParticleLumIdx = static_cast<int>(StochasticStellarPopParticleRealIdx::luminosity); // Base index for luminosity components
constexpr int StochasticStellarPopParticleStageIdx = static_cast<int>(StochasticStellarPopParticleIntIdx::evolution_stage);

// Number of real components for StochasticStellarPop_particles, mass + 3 velocity components + times + positions + death density + luminosity
template <typename problem_t>
constexpr int StochasticStellarPopParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 14 + Physics_Traits<problem_t>::nGroups; // mass, vx, vy, vz, birth_time, death_time, birth_xyz, death_xyz, death_density, mass_at_birth,
								// lum[nGroups]
	} else {
		return 14; // mass, vx, vy, vz, birth_time, death_time, birth_xyz, death_xyz, death_density, mass_at_birth
	}
}();

// Number of integer components for StochasticStellarPop_particles
constexpr int StochasticStellarPopParticleIntComps = 1; // evolution stage

// Type definitions for StochasticStellarPop_particles container and iterator
template <typename problem_t>
using StochasticStellarPopParticleContainer =
    amrex::AmrParticleContainer<StochasticStellarPopParticleRealComps<problem_t>, StochasticStellarPopParticleIntComps>;
template <typename problem_t>
using StochasticStellarPopParticleIterator = amrex::ParIter<StochasticStellarPopParticleRealComps<problem_t>, StochasticStellarPopParticleIntComps>;

//-------------------- Test particles --------------------

// Indices for test particles (Test_particles) using AMREX_ENUM for automatic string conversion
// The enum values are short names that will appear directly in the Header file
AMREX_ENUM(TestParticleRealIdx, // NOLINT
	   mass,		// Mass of the particle
	   vx,			// Velocity in x direction
	   vy,			// Velocity in y direction
	   vz,			// Velocity in z direction
	   birth_time,		// Time when particle becomes active
	   death_time,		// Time when particle becomes inactive
	   luminosity		// Base luminosity component (expanded to luminosity_0, luminosity_1, ... in I/O)
);

// Integer component indices using AMREX_ENUM
AMREX_ENUM(TestParticleIntIdx, // NOLINT
	   evolution_stage     // Evolution stage of the particle
);

// Backward compatibility aliases for existing code
constexpr int TestParticleMassIdx = static_cast<int>(TestParticleRealIdx::mass);
constexpr int TestParticleVxIdx = static_cast<int>(TestParticleRealIdx::vx);
constexpr int TestParticleVyIdx = static_cast<int>(TestParticleRealIdx::vy);
constexpr int TestParticleVzIdx = static_cast<int>(TestParticleRealIdx::vz);
constexpr int TestParticleBirthTimeIdx = static_cast<int>(TestParticleRealIdx::birth_time);
constexpr int TestParticleDeathTimeIdx = static_cast<int>(TestParticleRealIdx::death_time);
constexpr int TestParticleLumIdx = static_cast<int>(TestParticleRealIdx::luminosity); // Base index for luminosity components
constexpr int TestParticleStageIdx = static_cast<int>(TestParticleIntIdx::evolution_stage);

// Number of real components for Test_particles
template <typename problem_t>
constexpr int TestParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass, vx, vy, vz, birth_time, death_time, lum[nGroups]
	} else {
		return 6; // mass, vx, vy, vz, birth_time, death_time
	}
}();

// Number of integer components for Test_particles
constexpr int TestParticleIntComps = 1; // stellar evolution stage

// Type definitions for Test_particles container and iterator
template <typename problem_t> using TestParticleContainer = amrex::AmrParticleContainer<TestParticleRealComps<problem_t>, TestParticleIntComps>;
template <typename problem_t> using TestParticleIterator = amrex::ParIter<TestParticleRealComps<problem_t>, TestParticleIntComps>;

//-------------------- Sink particles --------------------

// Indices for Sink_particles using AMREX_ENUM for automatic string conversion
AMREX_ENUM(SinkParticleRealIdx, // NOLINT
	   mass,		// Mass of the particle
	   vx,			// Velocity in x direction
	   vy,			// Velocity in y direction
	   vz			// Velocity in z direction
);

// Backward compatibility aliases for existing code
constexpr int SinkParticleMassIdx = static_cast<int>(SinkParticleRealIdx::mass);
constexpr int SinkParticleVxIdx = static_cast<int>(SinkParticleRealIdx::vx);
constexpr int SinkParticleVyIdx = static_cast<int>(SinkParticleRealIdx::vy);
constexpr int SinkParticleVzIdx = static_cast<int>(SinkParticleRealIdx::vz);

// Number of real components for Sink_particles
constexpr int SinkParticleRealComps = 4; // mass, vx, vy, vz

// Type definitions for Sink_particles container and iterator
using SinkParticleContainer = amrex::AmrParticleContainer<SinkParticleRealComps>;
using SinkParticleIterator = amrex::ParIter<SinkParticleRealComps>;

#endif // AMREX_SPACEDIM == 3

//-------------------- Component Names for I/O --------------------

// Helper function to generate component names from an enum type
// If expandLast is true, the last enum component is expanded with _0, _1, ... suffixes
// to fill up to nComps total components
template <typename EnumType, int nComps, bool expandLast> auto expandEnumNames() -> amrex::Vector<std::string>
{
	const std::vector<std::string> enum_names = amrex::getEnumNameStrings<EnumType>();
	const auto enum_size = static_cast<int>(enum_names.size());
	amrex::Vector<std::string> names;

	if constexpr (nComps <= 0) {
		return names;
	}

	if constexpr (!expandLast) {
		// No expansion - use enum names directly
		return {enum_names.begin(), enum_names.end()};
	}

	// Add all components except the last one
	for (int i = 0; i < enum_size - 1; ++i) {
		names.push_back(enum_names[i]);
	}

	// Expand the last component into name_0, name_1, ...
	const std::string &base_name = enum_names.back();
	const int nExtra = nComps - enum_size + 1;
	for (int i = 0; i < nExtra; ++i) {
		names.push_back(base_name + "_" + std::to_string(i));
	}

	return names;
}

// Unified template function to get real component names for any particle type
// Uses AMREX_ENUM's getEnumNameStrings() and expands the last component for particle types with luminosity
template <ParticleType particleType, typename problem_t> auto getParticleRealCompNames() -> amrex::Vector<std::string>
{
	if constexpr (particleType == ParticleType::Rad) {
		return expandEnumNames<RadParticleRealIdx, RadParticleRealComps<problem_t>, true>();
	}
#if AMREX_SPACEDIM == 3
	else if constexpr (particleType == ParticleType::CIC) {
		return expandEnumNames<CICParticleRealIdx, CICParticleRealComps, false>();
	} else if constexpr (particleType == ParticleType::CICRad) {
		return expandEnumNames<CICRadParticleRealIdx, CICRadParticleRealComps<problem_t>, true>();
	} else if constexpr (particleType == ParticleType::StochasticStellarPop) {
		return expandEnumNames<StochasticStellarPopParticleRealIdx, StochasticStellarPopParticleRealComps<problem_t>, true>();
	} else if constexpr (particleType == ParticleType::Sink) {
		return expandEnumNames<SinkParticleRealIdx, SinkParticleRealComps, false>();
	} else if constexpr (particleType == ParticleType::Test) {
		return expandEnumNames<TestParticleRealIdx, TestParticleRealComps<problem_t>, true>();
	}
#endif
	else {
		return {};
	}
}

// Unified template function to get integer component names for any particle type
template <ParticleType particleType, typename problem_t> auto getParticleIntCompNames() -> amrex::Vector<std::string>
{
	amrex::Vector<std::string> names;

	if constexpr (particleType == ParticleType::Rad) { // NOLINT
							   // No integer components
	}
#if AMREX_SPACEDIM == 3
	else if constexpr (particleType == ParticleType::CIC) {	     // NOLINT
								     // No integer components
	} else if constexpr (particleType == ParticleType::CICRad) { // NOLINT
								     // No integer components
	} else if constexpr (particleType == ParticleType::StochasticStellarPop) {
		const std::vector<std::string> enum_names = amrex::getEnumNameStrings<StochasticStellarPopParticleIntIdx>();
		names = {enum_names.begin(), enum_names.end()};
	} else if constexpr (particleType == ParticleType::Sink) { // NOLINT
								   // No integer components
	} else if constexpr (particleType == ParticleType::Test) {
		const std::vector<std::string> enum_names = amrex::getEnumNameStrings<TestParticleIntIdx>();
		names = {enum_names.begin(), enum_names.end()};
	}
#endif
	return names;
}

//-------------------- Units --------------------

// Units data for each particle type as powers of Mass, Length, Time, Temperature
inline auto get_units_data() -> const auto &
{
	static const auto units_data = std::map<ParticleType, std::vector<std::map<std::string, std::array<int, 4>>>>{
	    {ParticleType::Rad, {{{"birth_time", {0, 0, 1, 0}}, {"death_time", {0, 0, 1, 0}}, {"luminosity", {-1, 2, -3, 0}}}}},
	    {ParticleType::CIC, {{{"mass", {1, 0, 0, 0}}, {"vx", {0, 1, -1, 0}}, {"vy", {0, 1, -1, 0}}, {"vz", {0, 1, -1, 0}}}}},
	    {ParticleType::CICRad,
	     {{{"mass", {1, 0, 0, 0}},
	       {"vx", {0, 1, -1, 0}},
	       {"vy", {0, 1, -1, 0}},
	       {"vz", {0, 1, -1, 0}},
	       {"birth_time", {0, 0, 1, 0}},
	       {"death_time", {0, 0, 1, 0}},
	       {"luminosity", {-1, 2, -3, 0}}}}},
	    {ParticleType::StochasticStellarPop,
	     {{{"mass", {1, 0, 0, 0}},
	       {"vx", {0, 1, -1, 0}},
	       {"vy", {0, 1, -1, 0}},
	       {"vz", {0, 1, -1, 0}},
	       {"birth_time", {0, 0, 1, 0}},
	       {"death_time", {0, 0, 1, 0}},
	       {"birth_x", {0, 1, 0, 0}},
	       {"birth_y", {0, 1, 0, 0}},
	       {"birth_z", {0, 1, 0, 0}},
	       {"death_x", {0, 1, 0, 0}},
	       {"death_y", {0, 1, 0, 0}},
	       {"death_z", {0, 1, 0, 0}},
	       {"death_density", {1, -3, 0, 0}},
	       {"mass_at_birth", {1, 0, 0, 0}},
	       {"luminosity", {-1, 2, -3, 0}}}}},
	    {ParticleType::Sink, {{{"mass", {1, 0, 0, 0}}, {"vx", {0, 1, -1, 0}}, {"vy", {0, 1, -1, 0}}, {"vz", {0, 1, -1, 0}}}}},
	    {ParticleType::Test,
	     {{{"mass", {1, 0, 0, 0}},
	       {"vx", {0, 1, -1, 0}},
	       {"vy", {0, 1, -1, 0}},
	       {"vz", {0, 1, -1, 0}},
	       {"birth_time", {0, 0, 1, 0}},
	       {"death_time", {0, 0, 1, 0}},
	       {"luminosity", {-1, 2, -3, 0}}}}}};
	return units_data;
}

// Assumptions for any particle type:
// 1. For massive particles, velocity components start after mass
// 2. Birth time, if existing, is always followed by death time

// Global particle parameters
// The 'inline' keyword is used here to avoid multiple definition errors when this header
// is included in multiple source files. It ensures that all translation units that include
// this header will refer to the same instance of these variables, rather than creating
// their own copies.

// Disable SN feedback when a particle evolves from SNProgenitor to SNRemnant
inline bool disable_SN_feedback = false; // NOLINT

// Placeholder parameters for particles. Used in gravity_3d.cpp tests
inline amrex::Real particle_param1 = -1.0; // NOLINT
inline amrex::Real particle_param2 = -1.0; // NOLINT

inline amrex::Real particle_param3 = -1.0; // NOLINT
inline amrex::Real eps_ff = 0.01;	   // NOLINT

// Scheme for SN feedback
inline SNScheme SN_scheme = SNScheme::SN_thermal_or_thermal_momentum; // NOLINT

// When true, homogenize gas velocity before SN injection to ensure energy conservation.
// When false, do not homogenize gas velocity, causing an extra energy term whichis the cross product of the inhomogeneity and radial momentum.
inline bool SN_smooth_gas_velocity = true; // NOLINT

// Sink particle accretion
inline bool sink_particle_use_uniform_kernel = false; // NOLINT. If true, use uniform accretion kernel in a (7 dx)^3 box

// Verbosity for particle operations
inline int particle_verbose = 0; // NOLINT print particle logistics

// Disable particle drift
inline bool disable_particle_drift = false; // NOLINT

// Maximum velocity limit for stellar particles in cm/s (default: 1000 km/s)
inline amrex::Real stellar_velocity_limit = 1.0e8; // NOLINT

// Maximum mass for LowMassComposite particles. If <= 0, no splitting is performed.
inline amrex::Real low_mass_composite_max_mass = -1.0; // NOLINT

inline int reproducibility_roundoff_redundancy = 20; // NOLINT; remove 20 bits from the significand

// Function to parse particle parameters from input file
// The 'inline' keyword allows this function to be defined in a header file without
// causing multiple definition errors when the header is included in multiple source files.
// It tells the linker that all instances of this function across different translation units
// should be treated as the same function. This is a common pattern for small utility
// functions defined in header files.
inline void particleParmParse()
{
	// Parse particle parameters
	const amrex::ParmParse pp("particles");
	pp.query("disable_SN_feedback", disable_SN_feedback);
	pp.query("sink_particle_use_uniform_kernel", sink_particle_use_uniform_kernel);

	// Handle SNScheme enum
	pp.query("SN_scheme", SN_scheme);

	// SN Galilean invariance option
	pp.query("SN_smooth_gas_velocity", SN_smooth_gas_velocity);

	// Stochastic SF parameters
	pp.query("eps_ff", eps_ff);

	// Handle integer verbose flag
	pp.query("verbose", particle_verbose);

	// Disable particle drift
	pp.query("disable_particle_drift", disable_particle_drift);

	// Stellar velocity limit parameter
	pp.query("stellar_velocity_limit", stellar_velocity_limit);

	// Low-mass composite particle mass cap (split into multiple particles if exceeded)
	pp.query("low_mass_composite_max_mass", low_mass_composite_max_mass);

	// Roundoff factor for particles
	pp.query("reproducibility_roundoff_redundancy", reproducibility_roundoff_redundancy);

	// Placeholder parameters for particles
	pp.query("param1", particle_param1);
	pp.query("param2", particle_param2);
}

} // namespace quokka

#endif // PARTICLE_TYPES_HPP_
