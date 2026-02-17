#include "io/DerivedParticleDeposition.H"

#include <algorithm>
#include <array>
#include <iterator>
#include <unordered_set>

#include "AMReX_Parser.H"
#include "AMReX_ParmParse.H"
#include "fundamental_constants.H"

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
	pp.query("mass_min", m_massMin);
	pp.query("mass_max", m_massMax);
	m_hasAgeFilter = pp.query("t_age", m_tAgeMax);
	std::string normalizationExpr;
	if (pp.query("normalization_expr", normalizationExpr)) {
		amrex::Parser parser(normalizationExpr);
		parser.setConstant("Msun", C::M_solar);
		parser.setConstant("yr", 3.15576e7);
		parser.setConstant("kpc", 1.0e3 * C::parsec);
		auto const parserExe = parser.compileHost<0>();
		m_normalization = static_cast<amrex::Real>(parserExe());
	}
	std::string explicitOutputName;
	bool const hasExplicitOutputName = pp.query("output_name", explicitOutputName);

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
		if (token != "mass" && token != "birth_mass") {
			amrex::Abort("DerivedParticleDeposition currently supports only deposit_fields = mass or birth_mass");
		}
		m_depositFields.push_back(token);
	}

	if (m_particleTypes.empty()) {
		amrex::Abort("DerivedParticleDeposition requires at least one particle type.");
	}

	if (m_depositFields.empty()) {
		amrex::Abort("DerivedParticleDeposition requires at least one deposit field.");
	}
	if (m_massMin > m_massMax) {
		amrex::Abort("DerivedParticleDeposition requires mass_min <= mass_max.");
	}
	if (m_hasAgeFilter && m_tAgeMax < 0.0) {
		amrex::Abort("DerivedParticleDeposition requires t_age >= 0 when provided.");
	}

	const int totalOutputs = static_cast<int>(m_particleTypes.size() * m_depositFields.size());
	if (hasExplicitOutputName && totalOutputs != 1) {
		amrex::Abort("DerivedParticleDeposition: output_name is only valid when exactly one output is produced.");
	}

	std::unordered_set<std::string> outputSet;
	for (auto const &ptype : m_particleTypes) {
		for (auto const &field : m_depositFields) {
			std::string outputName;
			if (hasExplicitOutputName) {
				outputName = explicitOutputName;
			} else if (totalOutputs == 1) {
				// Use the provider group name so users can configure via:
				// derived_vars = <name>
				// quokka.<name>.type = DerivedParticleDeposition
				outputName = m_fieldGroupName;
			} else {
				outputName = getFieldName(ptype, field);
			}
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

		if (output.depositField != "mass" && output.depositField != "birth_mass") {
			amrex::Abort("DerivedParticleDeposition currently supports only deposit_fields = mass or birth_mass");
		}

		mf.setVal(0.0, ncomp, 1, mf.nGrow());
		ctx.depositParticleMassDensity(output.particleType, output.depositField, mf, lev, ncomp, m_massMin, m_massMax, m_hasAgeFilter, m_tAgeMax);
		mf.mult(m_normalization, ncomp, 1, mf.nGrow());
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
	if (field == "birth_mass") {
		return m_prefix + "." + particleType + ".birth_mass_density";
	}

	amrex::Abort("Unknown deposit field in DerivedParticleDeposition: " + field);
	return "";
}

} // namespace quokka
