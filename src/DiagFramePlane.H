#ifndef DIAGFRAMEPLANE_H
#define DIAGFRAMEPLANE_H

#include "AMReX_VisMF.H"
#include "DiagBase.H"

class DiagFramePlane : public DiagBase::Register<DiagFramePlane>
{
      public:
	static auto identifier() -> std::string { return "DiagFramePlane"; }

	enum InterpType { Linear, Quadratic };

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
			 const amrex::Vector<std::string> &a_varNames) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice,
				       const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time,
				       const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref);

	static void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name);

	static void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm);

	static void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr,
				  amrex::VisMF::Header::Version /*whichVersion*/, amrex::NFilesIter &nfi, int nOutFile, MPI_Comm comm);

	static void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar);

	static void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray,
					  const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time,
					  const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio,
					  const std::string &versionName = "HyperCLaw-V1.1", const std::string &levelPrefix = "Level_",
					  const std::string &mfPrefix = "Cell");

	void close() override {}

      private:
	// Variables output
	amrex::Vector<std::string> m_fieldNames;
	amrex::Gpu::DeviceVector<int> m_fieldIndices_d;

	// Plane definition
	int m_normal;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_center;

	// Interpolation data
	InterpType m_interpType;
	amrex::Vector<amrex::GpuArray<amrex::Real, 3>> m_intwgt;
	amrex::Vector<int> m_k0;

	// 2D-plane boxArray vector
	amrex::Geometry m_geomLev0;
	amrex::Vector<amrex::BoxArray> m_sliceBA;
	amrex::Vector<amrex::Vector<int>> m_dmConvert;
	amrex::Vector<amrex::DistributionMapping> m_sliceDM;
};

#endif
