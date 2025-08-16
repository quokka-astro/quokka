#ifndef TYPEDMULTIFAB_HPP_ // NOLINT
#define TYPEDMULTIFAB_HPP_

#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "TypeList.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace quokka
{

// Base class for variable names/types
namespace variable_names
{
template <bool IsMassScalar> struct base_t {
	// Default constructor
	base_t() = default;

	// Perfect forwarding constructor to support any constructor arguments
	template <class... Ts> explicit base_t(Ts &&.../*args*/) {}

	// Virtual destructor for polymorphism
	virtual ~base_t() = default;

	// Static method to get variable name - must be overridden in derived classes
	static std::string name()
	{
		return "unnamed_variable";
	}

	// Whether this is a mass scalar
	static constexpr bool is_mass_scalar = IsMassScalar;
};
} // namespace variable_names

// Macro to define strongly-typed variables
#define VARIABLE(ns, varname)                                                                                                                              \
	struct varname : public variable_names::base_t<false> {                                                                                           \
		template <class... Ts> AMREX_GPU_HOST_DEVICE AMREX_INLINE varname(Ts &&...args) : variable_names::base_t<false>(std::forward<Ts>(args)...) \
		{                                                                                                                                          \
		}                                                                                                                                          \
		static std::string name()                                                                                                                  \
		{                                                                                                                                          \
			return #ns "." #varname;                                                                                                           \
		}                                                                                                                                          \
	}

// TypedMultifab: A strongly-typed wrapper around amrex::MultiFab
template <typename TypeListT> class TypedMultifab {
      public:
	using TypeList = TypeListT;
	static constexpr int ncomp = TypeList::n_types;

	// Constructors
	TypedMultifab() = default;

	TypedMultifab(amrex::BoxArray const &ba, amrex::DistributionMapping const &dm, int nghost, amrex::IntVect const &ngrow_vect,
		      amrex::MFInfo const &info = amrex::MFInfo(), amrex::FabFactory<amrex::FArrayBox> const &factory = amrex::FArrayBoxFactory())
	    : multifab_(std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost, ngrow_vect, info, factory))
	{
		setComponentNames();
	}

	TypedMultifab(amrex::BoxArray const &ba, amrex::DistributionMapping const &dm, int nghost, amrex::MFInfo const &info = amrex::MFInfo(),
		      amrex::FabFactory<amrex::FArrayBox> const &factory = amrex::FArrayBoxFactory())
	    : multifab_(std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost, info, factory))
	{
		setComponentNames();
	}

	// Move constructor
	TypedMultifab(TypedMultifab &&) = default;

	// Move assignment
	TypedMultifab &operator=(TypedMultifab &&) = default;

	// Delete copy operations to prevent unintended copies
	TypedMultifab(TypedMultifab const &) = delete;
	TypedMultifab &operator=(TypedMultifab const &) = delete;

	// Access to underlying MultiFab
	amrex::MultiFab &get()
	{
		return *multifab_;
	}
	amrex::MultiFab const &get() const
	{
		return *multifab_;
	}
	amrex::MultiFab *operator->()
	{
		return multifab_.get();
	}
	amrex::MultiFab const *operator->() const
	{
		return multifab_.get();
	}

	// Get array access with type safety
	template <typename VarType> auto arrays() const
	{
		constexpr std::size_t comp = TypeList::template GetIdx<VarType>();
		return multifab_->arrays(comp);
	}

	// Get array access for a specific box with type safety
	template <typename VarType> auto array(amrex::MFIter const &mfi) const
	{
		constexpr std::size_t comp = TypeList::template GetIdx<VarType>();
		return multifab_->array(mfi, comp);
	}

	// Get the component index for a given type
	template <typename VarType> static constexpr int comp()
	{
		return static_cast<int>(TypeList::template GetIdx<VarType>());
	}

	// Set component names based on TypeList
	void setComponentNames()
	{
		if (!multifab_) {
			return;
		}

		componentNames_.clear();
		componentNames_.reserve(ncomp);

		// Helper lambda to add component names
		auto addName = [this, idx = 0]<typename T>(T /*unused*/) mutable {
			componentNames_.push_back(T::name());
			++idx;
		};

		// Iterate through all types and set their names
		TypeList::IterateTypes(addName);
	}

	// Get component names
	std::vector<std::string> const &componentNames() const
	{
		return componentNames_;
	}

	// Get a specific component name
	template <typename VarType> static std::string componentName()
	{
		return VarType::name();
	}

      private:
	std::unique_ptr<amrex::MultiFab> multifab_;
	std::vector<std::string> componentNames_;
};

// Helper function to create a TypedMultifab
template <typename TypeListT>
auto makeTypedMultifab(amrex::BoxArray const &ba, amrex::DistributionMapping const &dm, int nghost, amrex::MFInfo const &info = amrex::MFInfo(),
		       amrex::FabFactory<amrex::FArrayBox> const &factory = amrex::FArrayBoxFactory())
{
	return TypedMultifab<TypeListT>(ba, dm, nghost, info, factory);
}

} // namespace quokka

#endif // TYPEDMULTIFAB_HPP_