#ifndef CATALYST_H_
#define CATALYST_H_

#include <string>

#include "AMReX_BaseFwd.H"
#include "AMReX_Geometry.H"
#include "AMReX_Vector.H"

/**
 * \brief This class aims at dumping performing in-situ analysis and visualization
 * with Catalyst. Catalyst initialize and finalize are called in the constructor
 * and destructor respectivelyThe function WriteToFile takes in the particles,
 * meshes, and fields, and formats them to be used by given pipeline scripts. All
 * exports are defined and executed externally through the given catalyst pipeline
 * and implementation.
 */
class FlushFormatCatalyst
{
      public:
	// Initailize
	FlushFormatCatalyst();

	/** Send particle, mesh, and field information through the catalyst pipeline */
	void WriteToFile(amrex::Vector<std::string> const &varnames, amrex::Vector<amrex::MultiFab> const &mf, amrex::Vector<amrex::Geometry> const &geom,
			 amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Vector<int> const &iteration, const double time, int nlev) const;

	// Finalize
	~FlushFormatCatalyst();
};

#endif
