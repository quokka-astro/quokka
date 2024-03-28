#ifndef DIAGPDF_H
#define DIAGPDF_H

#include "DiagBase.H"

class DiagPDF : public DiagBase::Register<DiagPDF>
{
      public:
	static auto identifier() -> std::string { return "DiagPDF"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			 const amrex::Vector<std::string> &a_stateVar) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	static auto MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;
	static auto MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;
	void writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf, const amrex::Real &a_sum);

	void close() override {}

      private:
	// PDF properties
	std::string m_fieldName;	// Field name
	int m_nBins{-1};		// Number of bins
	int m_normalized{1};		// Normalized ?
	int m_volWeighted{1};		// Volume weighted ?
	bool m_useFieldMinMax{true};	// Use min/max from field
	bool m_useLogSpacedBins{false}; // Use log-spaced bins
	amrex::Real m_lowBnd{0.0};	// User-defined low bound
	amrex::Real m_highBnd{0.0};	// User-defined high bound

	// Geometrical data
	amrex::Vector<amrex::Geometry> m_geoms; // Squirrel away the geoms
	amrex::Vector<amrex::IntVect> m_refRatio;

	// helper functions
	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto getBinIndex(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
								   const amrex::Real &transformedBinWidth, bool doLog) -> int;
};

#endif
