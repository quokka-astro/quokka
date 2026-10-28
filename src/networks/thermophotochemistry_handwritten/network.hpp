#ifndef NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
#define NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
/// \file network.hpp
/// \brief The constants of the thermophotochemistry_handwritten network.

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

namespace quokka::networks
{

/// Hydrogen photoionization in one chemical band, with dust in two thermal bands.
struct Thermophotochemistry {
	static constexpr int NumSpec = 3;
	/// indices of the species in xn
	enum Species : int { H = 0, H_p, e };

	static constexpr int NumChemBands = 1;
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return {{13.6, 26.0}}; }
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a chemical band
	static constexpr double ChemBandsPowerLawIndex = 0;

	/// the IR and the optical band, in that order
	static constexpr int NumThermalBands = 2;
};

} // namespace quokka::networks

#endif // NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
