#include "io/DerivedFieldBase.H"

#include <algorithm>

namespace quokka
{

void DerivedFieldBase::init(const std::string & /*a_prefix*/, std::string_view a_fieldName) { m_fieldGroupName = std::string(a_fieldName); }

void DerivedFieldBase::prepare(int /*a_nlevels*/, const amrex::Vector<amrex::Geometry> & /*a_geoms*/, const amrex::Vector<amrex::BoxArray> & /*a_grids*/,
			       const amrex::Vector<amrex::DistributionMapping> & /*a_dmap*/,
			       const amrex::Vector<std::string> & /*a_availableVars*/)
{
}

void DerivedFieldBase::addVars(amrex::Vector<std::string> &a_varList)
{
	for (auto const &name : m_fieldNames) {
		a_varList.push_back(name);
	}
}

auto DerivedFieldBase::computeField(int /*lev*/, const std::string & /*fieldName*/, amrex::MultiFab & /*mf*/, int /*ncomp*/,
				    ComputeContext const & /*ctx*/) const -> bool
{
	return false;
}

auto DerivedFieldBase::hasField(std::string_view field) const -> bool
{
	return std::ranges::any_of(m_fieldNames, [field](std::string const &name) { return name == field; });
}

} // namespace quokka
