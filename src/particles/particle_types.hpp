#ifndef PARTICLE_TYPES_HPP_
#define PARTICLE_TYPES_HPP_

#include "AMReX_AmrParticles.H"
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
	Test = bitflag<6>(),		     // Test particles with all features enabled, = 0b100000
	Star = bitflag<7>()		     // Star particles, = 0b1000000
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
	Test,		      // Test particles with all features enabled
	Star                       // Star particles
};

// Enum for SN schemes: ThermalOnly, ThermalAndMomentum
AMREX_ENUM(SNScheme,				   // NOLINT
	   SN_thermal_only,			   // pure thermal
	   SN_thermal_or_thermal_momentum,	   // pure thermal (RM<1) or thermal+momentum (RM>=1)
	   SN_thermal_kinetic_or_thermal_momentum, // thermal+kinetic (RM<1) or thermal+momentum (RM>=1)
	   SN_pure_kinetic_or_thermal_momentum	   // pure kinetic (RM<1) or thermal+momentum (RM>=1)
);

//-------------------- Radiation particles --------------------

// Indices for radiation particles (Rad_particles), birth time + death time + radiation groups
enum RadParticleDataIdx {
	RadParticleBirthTimeIdx = 0, // Time when particle becomes active
	RadParticleDeathTimeIdx,     // Time when particle becomes inactive
	RadParticleLumIdx	     // Base index for luminosity components
};

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

// Indices for gravitating particles (CIC_particles), mass + 3 velocity components
enum CICParticleDataIdx {
	CICParticleMassIdx = 0, // Mass of the particle
	CICParticleVxIdx,	// Velocity in x direction
	CICParticleVyIdx,	// Velocity in y direction
	CICParticleVzIdx	// Velocity in z direction
};

// Number of real components for CIC_particles, mass + 3 velocity components
constexpr int CICParticleRealComps = 4;

// Type definitions for CIC_particles container and iterator
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

//-------------------- Gravitating radiation particles --------------------

// Indices for gravitating radiation particles (CICRad_particles), mass + 3 velocity components + birth time + death time + radiation groups
enum CICRadParticleDataIdx {
	CICRadParticleMassIdx = 0,  // Mass of the particle
	CICRadParticleVxIdx,	    // Velocity in x direction
	CICRadParticleVyIdx,	    // Velocity in y direction
	CICRadParticleVzIdx,	    // Velocity in z direction
	CICRadParticleBirthTimeIdx, // Time when particle becomes active
	CICRadParticleDeathTimeIdx, // Time when particle becomes inactive
	CICRadParticleLumIdx	    // Base index for luminosity components
};

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
// - LowMassStar: singular low mass star
// - SNProgenitor: singular high-mass stars (> 9 Msun). Depending on the SF and SN scheme, these stars will either explode as supernovae in
//   the end of their lifetime unconditionally, or conditionally according to their evolutional track.
// - SNRemnant: Supernova remnant stage
// - LowMassComposite: composite of low-mass stars
// - Removed: marked for removal
enum class StellarEvolutionStage { LowMassStar, SNProgenitor, SNRemnant, LowMassComposite, Removed };

//-------------------- Stellar population particles --------------------

// Indices for StochasticStellarPop_particles
enum StochasticStellarPopParticleDataIdx {
	StochasticStellarPopParticleMassIdx = 0,  // Mass of the particle
	StochasticStellarPopParticleVxIdx,	  // Velocity in x direction
	StochasticStellarPopParticleVyIdx,	  // Velocity in y direction
	StochasticStellarPopParticleVzIdx,	  // Velocity in z direction
	StochasticStellarPopParticleBirthTimeIdx, // Time when particle becomes active
	StochasticStellarPopParticleDeathTimeIdx, // Time when particle becomes inactive
	StochasticStellarPopParticleLumIdx	  // Base index for luminosity components
};

constexpr int StochasticStellarPopParticleStageIdx = 0; // Evolution stage of the particle, index in the integer components

// Number of real components for StochasticStellarPop_particles, mass + 3 velocity components + luminosity
template <typename problem_t>
constexpr int StochasticStellarPopParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass, vx, vy, vz, birth_time, death_time, lum[nGroups]
	} else {
		return 6; // mass, vx, vy, vz, birth_time, death_time
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

// Indices for test particles (Test_particles)
enum TestParticleDataIdx {
	TestParticleMassIdx = 0,  // Mass of the particle
	TestParticleVxIdx,	  // Velocity in x direction
	TestParticleVyIdx,	  // Velocity in y direction
	TestParticleVzIdx,	  // Velocity in z direction
	TestParticleBirthTimeIdx, // Time when particle becomes active
	TestParticleDeathTimeIdx, // Time when particle becomes inactive
	TestParticleLumIdx	  // Base index for luminosity components
};

constexpr int TestParticleStageIdx = 0; // Evolution stage of the particle, index in the integer components

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

// Indices for Sink_particles
enum SinkParticleDataIdx {
	SinkParticleMassIdx = 0, // Mass of the particle
	SinkParticleVxIdx,	 // Velocity in x direction
	SinkParticleVyIdx,	 // Velocity in y direction
	SinkParticleVzIdx,	 // Velocity in z direction
};

// Number of real components for Sink_particles
constexpr int SinkParticleRealComps = 4; // mass, vx, vy, vz

// Type definitions for Sink_particles container and iterator
using SinkParticleContainer = amrex::AmrParticleContainer<SinkParticleRealComps>;
using SinkParticleIterator = amrex::ParIter<SinkParticleRealComps>;

//-------------------- Star particles --------------------

// Indices for Star_particles
enum StarParticleDataIdx {
  	StarParticleBirthTimeIdx = 0, // Time when particle becomes active
	StarParticleLumIdx,	  // Base index for luminosity components
	StarParticleMassLastIdx,  // Mass of particle in the last time step
	StarParticleMassIdx, // Mass of the particle
	StarParticleVxIdx,	 // Velocity in x direction
	StarParticleVyIdx,	 // Velocity in y direction
	StarParticleVzIdx,	 // Time step size
	StarParticleDtIdx,	 // Velocity in z direction
	StarParticleAMxIdx,  // Angular Momentum in x direction
	StarParticleAMyIdx,  // Angular Momentum in y direction
	StarParticleAMzIdx,  // Angular Momentum in z direction
	StarParticleMdeutIdx,  // Mass of gas that still contains deuterium
	StarParticlepolytropicIdx,  // n
	StarParticleMdotxIdx,  // Current mass accretion rate
	StarParticleBurnStateIdx,  // burnState
	StarParticleL_histIdx,  // Past masses
};

// Number of real components for StarParticles
constexpr int StarParticleRealComps = 12; // mass, vx, vy, vz, dt, amx, amy, amz, mdeut, n, mdot,burnState,l_hist

// Type definitions for Star_particles container and iterator
using StarParticleContainer = amrex::AmrParticleContainer<StarParticleRealComps>;
using StarParticleIterator = amrex::ParIter<StarParticleRealComps>;

// Indices for burnState
enum burningState {
	Uninitialized,
	None,
	VariableCoreDeuterium,
	SteadyCoreDeuterium,
	ShellDeuterium,
	ZAMS
};
  
#endif // AMREX_SPACEDIM == 3

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
	       {"luminosity", {-1, 2, -3, 0}}}}},
	    {ParticleType::Sink, {{{"mass", {1, 0, 0, 0}}, {"vx", {0, 1, -1, 0}}, {"vy", {0, 1, -1, 0}}, {"vz", {0, 1, -1, 0}}}}},
	    {ParticleType::Star,
	     {{{"mass", {1, 0, 0, 0}},
	       {"vx", {0, 1, -1, 0}},
	       {"vy", {0, 1, -1, 0}},
	       {"vz", {0, 1, -1, 0}},
	       {"amx", {1, 2, -1, 0}},
	       {"amy", {1, 2, -1, 0}},
	       {"amz", {1, 2, -1, 0}},
	       {"mdeut", {1, 0, 0, 0}},
	       {"n", {0, 0, 0, 0}},
	       {"mdot", {1, 0, -1, 0}},
	       {"burnState", {0, 0, 0, 0}},
	       {"l_hist", {1, 0, 0, 0}}}}},
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

// Sink particle accretion
inline bool sink_particle_use_uniform_kernel = false; // NOLINT. If true, use uniform accretion kernel in a (7 dx)^3 box

// Star particle accretion
inline bool star_particle_use_uniform_kernel = false; // NOLINT. If true, use uniform accretion kernel in a (7 dx)^3 box

// Verbosity for particle operations
inline int particle_verbose = 0; // NOLINT print particle logistics

// Disable particle drift
inline bool disable_particle_drift = false; // NOLINT

// Maximum velocity limit for stellar particles in cm/s (default: 1000 km/s)
inline amrex::Real stellar_velocity_limit = 1.0e8; // NOLINT

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
	pp.query("star_particle_use_uniform_kernel", star_particle_use_uniform_kernel);

	// Handle SNScheme enum
	pp.query("SN_scheme", SN_scheme);

	// Stochastic SF parameters
	pp.query("eps_ff", eps_ff);

	// Handle integer verbose flag
	pp.query("verbose", particle_verbose);

	// Disable particle drift
	pp.query("disable_particle_drift", disable_particle_drift);

	// Stellar velocity limit parameter
	pp.query("stellar_velocity_limit", stellar_velocity_limit);

	// Roundoff factor for particles
	pp.query("reproducibility_roundoff_redundancy", reproducibility_roundoff_redundancy);

	// Placeholder parameters for particles
	pp.query("param1", particle_param1);
	pp.query("param2", particle_param2);
}

} // namespace quokka

#endif // PARTICLE_TYPES_HPP_
