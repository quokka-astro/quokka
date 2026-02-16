#ifndef DERIVED_FIELD_BASE_H_
#define DERIVED_FIELD_BASE_H_

#include <string>
#include <string_view>
#include <functional>

#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Vector.H"
#include "Factory.H"

namespace quokka
{

class DerivedFieldBase : public Factory<DerivedFieldBase>
{
      public:
	~DerivedFieldBase() override = default;

	static auto base_identifier() -> std::string { return "DerivedField"; }

	virtual void init(const std::string &a_prefix, std::string_view a_fieldName);
	virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_availableVars);

	virtual void addVars(amrex::Vector<std::string> &a_varList);
	struct ComputeContext {
		std::function<void(const std::string &, amrex::MultiFab &, int, int)> depositParticleMassDensity;
	};
	virtual auto computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool;

	[[nodiscard]] auto hasField(std::string_view field) const -> bool;

      protected:
	std::string m_fieldGroupName;
	amrex::Vector<std::string> m_fieldNames;
};

} // namespace quokka

#endif // DERIVED_FIELD_BASE_H_
