#ifndef JAFF_NETWORK_HPP_
#define JAFF_NETWORK_HPP_
/// \file network.hpp
/// \brief The constants and runtime parameters of a jaff-generated network. A
/// Quokka problem names the network type with Microphysics_Traits::Network.

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::networks
{

/// The runtime parameters of JaffNetwork, read from the `network` prefix. They
/// are defined in network.cpp, not inline here: HIP cannot register inline
/// managed variables.
namespace jaff_network
{
/// floor of the species and radiation after a burn, and of the mass scalars
extern AMREX_GPU_MANAGED amrex::Real small_x; // NOLINT
} // namespace jaff_network

// $JAFF SUB label
/// The jaff network photoionization_H_caseB.
// $JAFF END
struct JaffNetwork {
	// $JAFF SUB nspec
	static constexpr int NumSpec = 3;
	// $JAFF END
	/// indices of the species in xn
	enum Species : int {
		// $JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign
		h = 0,
		hj = 1,
		ek = 2,
		// $JAFF END
	};

	// $JAFF SUB nbands
	static constexpr int NumChemBands = 1;
	// $JAFF END
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1>
	{
		// $JAFF REPEAT radiation_boundary IN radiation_boundaries
		return {{13.6, 26.0}};
		// $JAFF END
	}
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a
	/// chemical band
	// $JAFF SUB profile_index
	static constexpr double ChemBandsPowerLawIndex = 0;
	// $JAFF END

	static constexpr int NumThermalBands = 0;

	/// floor of the species and radiation after a burn, and of the mass scalars
	/// (jaff_network::small_x)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real { return jaff_network::small_x; }

	/// Read the runtime parameters (namespace jaff_network).
	static void ReadParameters();
};

} // namespace quokka::networks

#endif // JAFF_NETWORK_HPP_
