#ifndef TYPED_MULTIFAB_HPP_
#define TYPED_MULTIFAB_HPP_

#include <AMReX_Array4.H>
#include <AMReX_MultiFab.H>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "util/TypeList.hpp"
#include "util/VariableTypes.hpp"

namespace quokka
{

template <class TL> class TypedMultifab
{
      private:
	struct ComponentInfo {
		amrex::MultiFab *mf = nullptr; // Non-owning pointer
		int component_index = 0;       // Component index within the MultiFab
		bool owned = false;	       // Whether we own this MultiFab
	};

	// Map from type_index to component info
	std::unordered_map<std::type_index, ComponentInfo> component_map_;

	// Owned MultiFabs (for when we create new ones)
	std::vector<std::unique_ptr<amrex::MultiFab>> owned_multifabs_;

	// Box array and distribution mapping (stored for creating new MultiFabs if needed)
	amrex::BoxArray ba_;
	amrex::DistributionMapping dm_;
	int nghost_ = 0;

	// Helper to get component info for a type
	template <class VarType> ComponentInfo &getComponentInfo()
	{
		auto it = component_map_.find(std::type_index(typeid(VarType)));
		if (it == component_map_.end()) {
			amrex::Abort("TypedMultifab: Variable type not found");
		}
		return it->second;
	}

	template <class VarType> const ComponentInfo &getComponentInfo() const
	{
		auto it = component_map_.find(std::type_index(typeid(VarType)));
		if (it == component_map_.end()) {
			amrex::Abort("TypedMultifab: Variable type not found");
		}
		return it->second;
	}

      public:
	// Constructor that creates a new MultiFab with all components
	TypedMultifab(const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, int nghost) : ba_(ba), dm_(dm), nghost_(nghost)
	{
		// Create a single MultiFab to hold all components
		auto mf = std::make_unique<amrex::MultiFab>(ba, dm, TL::n_types, nghost);
		amrex::MultiFab *mf_ptr = mf.get();
		owned_multifabs_.push_back(std::move(mf));

		// Map each type to its component index
		std::size_t idx = 0;
		TL::IterateTypes([&](auto t) {
			using VarType = decltype(t);
			component_map_[std::type_index(typeid(VarType))] = {mf_ptr, static_cast<int>(idx), true};
			idx++;
		});
	}

	// Constructor that wraps existing MultiFabs without copying
	template <class... TypedMFs>
	TypedMultifab(const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, int nghost, TypedMFs &&...typed_mfs) : ba_(ba), dm_(dm), nghost_(nghost)
	{
		// Process each input TypedMultifab and extract components we need
		(extractComponents(std::forward<TypedMFs>(typed_mfs)), ...);
	}

	// Default constructor
	TypedMultifab() = default;

	// Move constructor
	TypedMultifab(TypedMultifab &&other) = default;

	// Move assignment
	TypedMultifab &operator=(TypedMultifab &&other) = default;

	// Delete copy operations
	TypedMultifab(const TypedMultifab &) = delete;
	TypedMultifab &operator=(const TypedMultifab &) = delete;

	// Get Array4 for a specific variable type and MFIter
	template <class VarType> auto array(const amrex::MFIter &mfi) const
	{
		static_assert(TypeListContains_v<VarType, TL>, "Variable type not in TypeList");
		const auto &info = getComponentInfo<VarType>();
		const auto &mf_array = info.mf->array(mfi);
		return amrex::Array4<amrex::Real const>(mf_array, info.component_index);
	}

	template <class VarType> auto array(const amrex::MFIter &mfi)
	{
		static_assert(TypeListContains_v<VarType, TL>, "Variable type not in TypeList");
		auto &info = getComponentInfo<VarType>();
		auto &mf_array = info.mf->array(mfi);
		return amrex::Array4<amrex::Real>(mf_array, info.component_index);
	}

	// Get arrays for all components in a box
	template <class VarType> auto arrays(const amrex::Box &box) const
	{
		static_assert(TypeListContains_v<VarType, TL>, "Variable type not in TypeList");
		const auto &info = getComponentInfo<VarType>();
		return info.mf->arrays()[box][info.component_index];
	}

	template <class VarType> auto arrays(const amrex::Box &box)
	{
		static_assert(TypeListContains_v<VarType, TL>, "Variable type not in TypeList");
		auto &info = getComponentInfo<VarType>();
		return info.mf->arrays()[box][info.component_index];
	}

	// Get component name
	template <class VarType> static std::string component_name() { return VarType::name(); }

	// Get all component names
	std::vector<std::string> component_names() const
	{
		std::vector<std::string> names;
		TL::IterateTypes([&](auto t) {
			using VarType = decltype(t);
			names.push_back(VarType::name());
		});
		return names;
	}

	// Get number of components
	static constexpr std::size_t num_components() { return TL::n_types; }

	// Check if we have a specific component
	template <class VarType> bool hasComponent() const { return component_map_.find(std::type_index(typeid(VarType))) != component_map_.end(); }

	// Get the underlying MultiFab for a component (for AMReX interop)
	template <class VarType> amrex::MultiFab &getMultiFab()
	{
		auto &info = getComponentInfo<VarType>();
		return *info.mf;
	}

	template <class VarType> const amrex::MultiFab &getMultiFab() const
	{
		const auto &info = getComponentInfo<VarType>();
		return *info.mf;
	}

	// Get component index within its MultiFab
	template <class VarType> int getComponentIndex() const
	{
		const auto &info = getComponentInfo<VarType>();
		return info.component_index;
	}

      private:
	// Helper to extract components from another TypedMultifab
	template <class OtherTL> void extractComponents(const TypedMultifab<OtherTL> &other)
	{
		// Iterate through our types and see which ones exist in the other
		TL::IterateTypes([&](auto t) {
			using VarType = decltype(t);
			if constexpr (TypeListContains_v<VarType, OtherTL>) {
				// This type exists in the other TypedMultifab
				const auto &other_info = other.template getComponentInfo<VarType>();
				component_map_[std::type_index(typeid(VarType))] = {
				    other_info.mf, other_info.component_index,
				    false // We don't own this
				};
			}
		});
	}

	template <class OtherTL> void extractComponents(TypedMultifab<OtherTL> &&other)
	{
		// For rvalue references, we can potentially take ownership
		TL::IterateTypes([&](auto t) {
			using VarType = decltype(t);
			if constexpr (TypeListContains_v<VarType, OtherTL>) {
				auto &other_info = other.template getComponentInfo<VarType>();
				if (other_info.owned && other.owned_multifabs_.size() == 1) {
					// If the other owns a single MultiFab and we're moving from it,
					// we can take ownership
					owned_multifabs_.push_back(std::move(other.owned_multifabs_[0]));
					component_map_[std::type_index(typeid(VarType))] = {owned_multifabs_.back().get(), other_info.component_index, true};
				} else {
					// Otherwise just reference it
					component_map_[std::type_index(typeid(VarType))] = {other_info.mf, other_info.component_index, false};
				}
			}
		});
	}
};

// Helper function to create a TypedMultifab from multiple sources
template <class TargetTL, class... SourceTMs>
auto makeTypedMultifab(const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, int nghost, SourceTMs &&...sources)
{
	return TypedMultifab<TargetTL>(ba, dm, nghost, std::forward<SourceTMs>(sources)...);
}

} // namespace quokka

#endif // TYPED_MULTIFAB_HPP_