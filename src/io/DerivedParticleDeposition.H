#ifndef DERIVED_PARTICLE_DEPOSITION_H_
#define DERIVED_PARTICLE_DEPOSITION_H_

#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "io/DerivedFieldBase.H"

namespace quokka
{

class DerivedParticleDeposition : public DerivedFieldBase::Register<DerivedParticleDeposition>
{
      public:
	static auto identifier() -> std::string { return "DerivedParticleDeposition"; }

	void init(const std::string &a_prefix, std::string_view a_fieldName) override;
	auto computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool override;

      private:
	struct OutputSpec {
		std::string particleType;
		std::string depositField;
		std::string outputName;
	};

	std::string m_prefix{"particle"};
	std::vector<std::string> m_particleTypes;
	std::vector<std::string> m_depositFields;
	std::vector<OutputSpec> m_outputs;
	amrex::Real m_massMin{std::numeric_limits<amrex::Real>::lowest()};
	amrex::Real m_massMax{std::numeric_limits<amrex::Real>::max()};
	bool m_hasAgeFilter{false};
	amrex::Real m_tAgeMax{std::numeric_limits<amrex::Real>::max()};
	amrex::Real m_normalization{1.0};

	[[nodiscard]] static auto isSupportedParticleType(std::string_view particleType) -> bool;

	[[nodiscard]] auto getFieldName(const std::string &particleType, const std::string &field) const -> std::string;
};

} // namespace quokka

#endif // DERIVED_PARTICLE_DEPOSITION_H_
