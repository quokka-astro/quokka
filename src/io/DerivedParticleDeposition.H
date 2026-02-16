#ifndef DERIVED_PARTICLE_DEPOSITION_H_
#define DERIVED_PARTICLE_DEPOSITION_H_

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

	[[nodiscard]] static auto isSupportedParticleType(std::string_view particleType) -> bool;

	[[nodiscard]] auto getFieldName(const std::string &particleType, const std::string &field) const -> std::string;
};

} // namespace quokka

#endif // DERIVED_PARTICLE_DEPOSITION_H_
