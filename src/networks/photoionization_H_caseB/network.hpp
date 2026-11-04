#ifndef JAFF_NETWORK_HPP_
#define JAFF_NETWORK_HPP_
/// \file network.hpp
/// \brief The constants and runtime parameters of a jaff-generated network. A
/// Quokka problem names the network type with Microphysics_Traits::Network.

#include <cmath> // INFINITY, for an open top band edge
#include <string_view>

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

/// The jaff network's own namespace, named after the network, so that several
/// jaff networks can be linked into one executable.
// $$JAFF SUB label
namespace quokka::networks::photoionization_H_caseB
{
// $$JAFF END

// The runtime parameters of Network, read from the `network` prefix. They are
// defined in network.cpp, not inline here: HIP cannot register inline managed
// variables.

/// floor of the species and radiation after a burn, and of the mass scalars
extern AMREX_GPU_MANAGED amrex::Real small_x; // NOLINT

// $$JAFF SUB label
/// The jaff network photoionization_H_caseB.
// $$JAFF END
struct Network {
	// $$JAFF SUB nspec
	static constexpr int NumSpec = 3;
	// $$JAFF END
	/// indices of the species in xn
	enum Species : int { // NOLINT(cppcoreguidelines-use-enum-class)
		// $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign
		h = 0,
		hj = 1,
		ek = 2,
		// $$JAFF END
	};

	/// mass of each species [g]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesMass() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_mass IN specie_masses
		return {{1.673773e-24, 1.6728620616289998e-24, 9.10938371e-28}};
		// $$JAFF END
	}
	/// adiabatic index of each species, from the network's [network.eos]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesGamma() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_gamma IN specie_gammas
		return {{1.6666666666666667, 1.6666666666666667, 1.6666666666666667}};
		// $$JAFF END
	}
	/// the equation of state the network assumes, as jaff names it (the type of
	/// [network.eos]); Quokka maps it to one of its EOSs
	// $$JAFF SUB eos_type
	static constexpr std::string_view EOSType = "ideal";
	// $$JAFF END

	// $$JAFF SUB nbands
	static constexpr int NumChemBands = 1;
	// $$JAFF END
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1>
	{
		// $$JAFF REPEAT radiation_boundary IN radiation_boundaries
		return {{13.6, 26.0}};
		// $$JAFF END
	}
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a
	/// chemical band; Quokka takes one alpha for all bands
	// $$JAFF GET profile_index IN 0
	static constexpr double ChemBandsPowerLawIndex = 0;
	// $$JAFF END
	// $$JAFF REPEAT idx, profile_index IN profile_indices
	static_assert(0 == ChemBandsPowerLawIndex, "band 0: Quokka takes one profile index for all chemical bands");
	// $$JAFF END

	static constexpr int NumThermalBands = 0;

	/// floor of the species and radiation after a burn, and of the mass scalars
	/// (small_x)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real { return small_x; }

	/// Read the runtime parameters.
	static void ReadParameters();
};

// $$JAFF SUB label
} // namespace quokka::networks::photoionization_H_caseB
// $$JAFF END

#endif // JAFF_NETWORK_HPP_
