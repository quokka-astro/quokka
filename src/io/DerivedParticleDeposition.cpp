#include "io/DerivedParticleDeposition.H"

#include <algorithm>
#include <array>
#include <iterator>
#include <unordered_set>

#include "AMReX_ParmParse.H"

namespace quokka
{

auto DerivedParticleDeposition::isSupportedParticleType(std::string_view particleType) -> bool
{
	static constexpr std::array<std::string_view, 5> supportedTypes = {"CIC", "CICRad", "StochasticStellarPop", "Sink", "Test"};
	return std::find(supportedTypes.begin(), supportedTypes.end(), particleType) != supportedTypes.end();
}

void DerivedParticleDeposition::init(const std::string &a_prefix, std::string_view a_fieldName)
{
	DerivedFieldBase::init(a_prefix, a_fieldName);

	amrex::ParmParse const pp(a_prefix);
	pp.query("prefix", m_prefix);

	amrex::Vector<std::string> particleTypes = {"CIC"};
	if (pp.countval("particle_types") > 0) {
		pp.queryarr("particle_types", particleTypes);
	}
	amrex::Vector<std::string> depositFields = {"mass"};
	if (pp.countval("deposit_fields") > 0) {
		pp.queryarr("deposit_fields", depositFields);
	}

	for (auto const &token : particleTypes) {
		if (!isSupportedParticleType(token)) {
			amrex::Abort("Unsupported particle type in DerivedParticleDeposition: " + token);
		}
		m_particleTypes.push_back(token);
	}
	for (auto const &token : depositFields) {
		if (token != "mass") {
			amrex::Abort("DerivedParticleDeposition currently supports only deposit_fields = mass");
		}
		m_depositFields.push_back(token);
	}

	if (m_particleTypes.empty()) {
		amrex::Abort("DerivedParticleDeposition requires at least one particle type.");
	}

	if (m_depositFields.empty()) {
		amrex::Abort("DerivedParticleDeposition requires at least one deposit field.");
	}

	std::unordered_set<std::string> outputSet;
	for (auto const &ptype : m_particleTypes) {
		for (auto const &field : m_depositFields) {
			auto const outputName = getFieldName(ptype, field);
			if (!outputSet.insert(outputName).second) {
				amrex::Abort("Duplicate output field generated in DerivedParticleDeposition: " + outputName);
			}
			m_outputs.push_back({ptype, field, outputName});
			m_fieldNames.push_back(outputName);
		}
	}
}

auto DerivedParticleDeposition::computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool
{
	if (!hasField(fieldName)) {
		return false;
	}

	for (auto const &output : m_outputs) {
		if (output.outputName != fieldName) {
			continue;
		}

		if (output.depositField != "mass") {
			amrex::Abort("DerivedParticleDeposition currently supports only deposit_fields = mass");
		}

		mf.setVal(0.0, ncomp, 1, mf.nGrow());
		ctx.depositParticleMassDensity(output.particleType, mf, lev, ncomp);
		return true;
	}

	amrex::Abort("DerivedParticleDeposition failed to resolve field: " + fieldName);
	return false;
}

auto DerivedParticleDeposition::getFieldName(const std::string &particleType, const std::string &field) const -> std::string
{
	if (field == "mass") {
		return m_prefix + "." + particleType + ".mass_density";
	}

	amrex::Abort("Unknown deposit field in DerivedParticleDeposition: " + field);
	return "";
}

} // namespace quokka
