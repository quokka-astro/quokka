#ifndef DIAGBASE_H
#define DIAGBASE_H

#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "DiagFilter.H"
#include "Factory.H"
#include "yaml-cpp/yaml.h"
#include <functional>
#include <unordered_map>

class DiagBase : public quokka::Factory<DiagBase>
{
      public:
	~DiagBase() override = default;

	static auto base_identifier() -> std::string { return "DiagBase"; }

	virtual void init(const std::string &a_prefix, std::string_view a_diagName);

	virtual void close() = 0;

	[[nodiscard]] auto needUpdate() const -> bool { return need_update; }

	virtual auto doDiag(const amrex::Real &a_time, int a_nstep) -> bool;

	virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames);

	// Callback type for writing physics particles (allows handling templated particle registers)
	using ParticleWriterFunc = std::function<void(const std::string &plotfilename)>;

	// Type for projection data (used by DiagProjectionPlot)
	using ProjectionData = std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>;

	virtual void processDiag(int a_nstep, const amrex::Real &a_time, const amrex::Vector<const amrex::MultiFab *> &a_state,
				 const amrex::Vector<std::string> &a_varNames, int finest_level, const amrex::Vector<amrex::Geometry> &a_geoms,
				 const amrex::Vector<int> &a_istep, const amrex::Vector<amrex::IntVect> &a_refRatio, void *tracerPC_ptr,
				 const std::array<amrex::Vector<const amrex::MultiFab *>, AMREX_SPACEDIM> *a_state_fc,
				 const std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> *a_varNames_fc, const ParticleWriterFunc &particleWriter,
				 const YAML::Node &simulationMetadata, const ProjectionData *projectionData = nullptr,
				 amrex::Direction projectionDir = amrex::Direction::x) = 0;

	virtual void addVars(amrex::Vector<std::string> &a_varList);

	static auto getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int;

	static auto getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int>;

      protected:
	std::string m_diagfile;
	int m_verbose{0};
	amrex::Real m_per{-1.0};
	int m_interval{-1};
	bool need_update{true};
	bool first_time{true};
	amrex::Vector<DiagFilter> m_filters{};
	amrex::Gpu::DeviceVector<DiagFilterData> m_filterData;
};

#endif
