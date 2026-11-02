#ifndef QUOKKA_PARTICLE_SCHEMA_HPP_
#define QUOKKA_PARTICLE_SCHEMA_HPP_

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace quokka
{
enum class ParticleType { Rad, CIC, CICRad, StochasticStellarPop, Sink, Test, Star, Tracer };

// Logical indices only. RealStorage applies the position prefix exactly once.
struct ParticleLayout {
	int numReal = 0;
	int numInt = 0;
	int mass = -1;
	int velocity = -1;
	int birthTime = -1;
	int deathTime = -1;
	int luminosity = -1;
	int numLuminosities = 0;
	int evolutionStage = -1;
	int massAtBirth = -1;
	int accretionRate = -1;
	int angularMomentum = -1;

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto RealStorage(int logical) const -> int
	{
		AMREX_ALWAYS_ASSERT(logical >= 0 && logical < numReal);
		return AMREX_SPACEDIM + logical;
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto IntStorage(int logical) const -> int
	{
		AMREX_ALWAYS_ASSERT(logical >= 0 && logical < numInt);
		return logical;
	}
};
static_assert(std::is_trivially_copyable_v<ParticleLayout>);

// Each units key refers to the existing per-population Fields.yaml metadata.
// Empty keys identify fields for which the legacy output declares no units.
class ParticleSchema
{
      public:
	using Units = std::map<std::string, std::array<int, 4>>;
	ParticleSchema(ParticleType population, std::string name, amrex::Vector<std::string> realNames, amrex::Vector<std::string> intNames,
		       amrex::Vector<amrex::ParticleReal> realDefaults, amrex::Vector<int> intDefaults, amrex::Vector<std::string> realUnitsKeys,
		       ParticleLayout layout, Units units = {})
	    : population_(population), name_(std::move(name)), realNames_(std::move(realNames)), intNames_(std::move(intNames)),
	      realDefaults_(std::move(realDefaults)), intDefaults_(std::move(intDefaults)), realUnitsKeys_(std::move(realUnitsKeys)), layout_(layout),
	      units_(std::move(units))
	{
		Validate();
	}

	[[nodiscard]] auto Population() const -> ParticleType { return population_; }
	[[nodiscard]] auto Name() const -> const std::string & { return name_; }
	[[nodiscard]] auto RealNames() const -> const amrex::Vector<std::string> & { return realNames_; }
	[[nodiscard]] auto IntNames() const -> const amrex::Vector<std::string> & { return intNames_; }
	[[nodiscard]] auto RealDefaults() const -> const amrex::Vector<amrex::ParticleReal> & { return realDefaults_; }
	[[nodiscard]] auto IntDefaults() const -> const amrex::Vector<int> & { return intDefaults_; }
	[[nodiscard]] auto RealUnitsKeys() const -> const amrex::Vector<std::string> & { return realUnitsKeys_; }
	[[nodiscard]] auto UnitDefinitions() const -> const Units & { return units_; }
	[[nodiscard]] auto Layout() const -> ParticleLayout { return layout_; }
	[[nodiscard]] auto RealIndex(const std::string &name) const -> int { return Find(realNames_, name); }
	[[nodiscard]] auto IntIndex(const std::string &name) const -> int { return Find(intNames_, name); }

      private:
	static auto Find(const amrex::Vector<std::string> &names, const std::string &name) -> int
	{
		const auto it = std::find(names.begin(), names.end(), name);
		if (it == names.end()) {
			throw std::out_of_range("Unknown particle field: " + name);
		}
		return static_cast<int>(it - names.begin());
	}
	static void ValidateNames(const amrex::Vector<std::string> &names)
	{
		std::set<std::string> unique;
		for (const auto &name : names) {
			if (name.empty() || !unique.insert(name).second) {
				throw std::invalid_argument("Particle field names must be nonempty and unique within each scalar type");
			}
		}
	}
	void Validate() const
	{
		ValidateNames(realNames_);
		ValidateNames(intNames_);
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			if (std::find(realNames_.begin(), realNames_.end(), "__position_" + std::to_string(d)) != realNames_.end()) {
				throw std::invalid_argument("Particle field name conflicts with reserved position storage");
			}
		}
		if (name_.empty() || realNames_.size() != realDefaults_.size() || intNames_.size() != intDefaults_.size() ||
		    realNames_.size() != realUnitsKeys_.size() || realNames_.size() > std::numeric_limits<int>::max() - AMREX_SPACEDIM ||
		    intNames_.size() > std::numeric_limits<int>::max() || layout_.numReal != static_cast<int>(realNames_.size()) ||
		    layout_.numInt != static_cast<int>(intNames_.size())) {
			throw std::invalid_argument("Particle schema metadata and component counts must agree");
		}
		for (const auto &key : realUnitsKeys_) {
			if (!key.empty() && !units_.contains(key)) {
				throw std::invalid_argument("Particle field units key is absent from schema unit definitions");
			}
		}
		const auto check = [](int index, int count, int width = 1) {
			if (index < -1 || (index >= 0 && (width < 1 || index > count - width))) {
				throw std::invalid_argument("Particle capability index is outside its schema");
			}
		};
		for (const int index : {layout_.mass, layout_.birthTime, layout_.deathTime, layout_.massAtBirth, layout_.accretionRate}) {
			check(index, layout_.numReal);
		}
#if AMREX_SPACEDIM == 3
		check(layout_.velocity, layout_.numReal, 3);
#else
		check(layout_.velocity, layout_.numReal, population_ == ParticleType::Tracer ? AMREX_SPACEDIM : 3);
#endif
		check(layout_.angularMomentum, layout_.numReal, 3);
		check(layout_.evolutionStage, layout_.numInt);
		if (layout_.numLuminosities < 0 || ((layout_.luminosity == -1) != (layout_.numLuminosities == 0))) {
			throw std::invalid_argument("Particle luminosity capability requires a consistent index and count");
		}
		check(layout_.luminosity, layout_.numReal, layout_.numLuminosities);
	}

	ParticleType population_;
	std::string name_;
	amrex::Vector<std::string> realNames_;
	amrex::Vector<std::string> intNames_;
	amrex::Vector<amrex::ParticleReal> realDefaults_;
	amrex::Vector<int> intDefaults_;
	amrex::Vector<std::string> realUnitsKeys_;
	ParticleLayout layout_;
	Units units_;
};

// Device views contain no ownership or host metadata. An index always identifies
// one record in a tile; adjacent fields need not be contiguous in memory.
template <typename TileData> class ParticleView
{
      public:
	AMREX_GPU_HOST_DEVICE ParticleView(TileData data, amrex::Long index, ParticleLayout layout) : data_(data), index_(index), layout_(layout)
	{
		AMREX_ALWAYS_ASSERT(index >= 0);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto Layout() const -> ParticleLayout { return layout_; }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto pos(int direction) const -> decltype(auto)
	{
		AMREX_ALWAYS_ASSERT(direction >= 0 && direction < AMREX_SPACEDIM);
		return data_.pos(direction, index_);
	}
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto rdata(int logical) const -> decltype(auto) { return data_.rdata(layout_.RealStorage(logical), index_); }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto idata(int logical) const -> decltype(auto) { return data_.idata(layout_.IntStorage(logical), index_); }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto id() const -> decltype(auto) { return data_.id(index_); }
	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto cpu() const -> decltype(auto) { return data_.cpu(index_); }

      private:
	TileData data_;
	amrex::Long index_;
	ParticleLayout layout_;
};
// Adapt AMReX mesh transfer callbacks to the schema-aware particle view.
template <typename F> struct ParticleMeshFunctor {
	F function;
	ParticleLayout layout;
	template <typename TileData, typename Array, typename GeometryArray>
	AMREX_GPU_DEVICE void operator()(const TileData &data, int i, const Array &mesh, const GeometryArray &plo, const GeometryArray &dxi) const
	{
		function(ParticleView(data, i, layout), mesh, plo, dxi);
	}
};
template <typename F> ParticleMeshFunctor(F, ParticleLayout) -> ParticleMeshFunctor<F>;
} // namespace quokka
#endif
