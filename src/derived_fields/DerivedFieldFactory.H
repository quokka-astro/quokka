#ifndef DERIVED_FIELD_FACTORY_H
#define DERIVED_FIELD_FACTORY_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "Factory.H"

class DerivedFieldFactory : public quokka::Factory<DerivedFieldFactory>
{
      public:
	~DerivedFieldFactory() override = default;

	static auto base_identifier() -> std::string { return "DerivedFieldFactory"; }

	// Initialize the derived field with parameters from ParmParse
	virtual void init(const std::string &fieldName, const amrex::ParmParse &pp) = 0;

	// Compute the derived field and store it in the provided MultiFab
	virtual void compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const = 0;

	// Get the field name
	[[nodiscard]] virtual auto getFieldName() const -> std::string = 0;

	// Get the number of components for this field
	[[nodiscard]] virtual auto getNumComponents() const -> int = 0;

	// Get component names (for multi-component fields)
	[[nodiscard]] virtual auto getComponentNames() const -> std::vector<std::string> = 0;

	// Check if this field requires specific state variables
	[[nodiscard]] virtual auto getRequiredStateVars() const -> std::vector<std::string> = 0;

      protected:
	std::string fieldName_;
	int numComponents_{1};
	std::vector<std::string> componentNames_;
	std::vector<std::string> requiredStateVars_;
};

// Factory manager for derived fields
class DerivedFieldManager
{
      public:
	static auto getInstance() -> DerivedFieldManager &;

	// Initialize derived fields from ParmParse
	void initFromParmParse();

	// Get list of active derived field names
	[[nodiscard]] auto getActiveFields() const -> const std::vector<std::string> &;

	// Compute a specific derived field
	void computeField(const std::string &fieldName, amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time,
			  int ncomp) const;

	// Get the field factory for a specific field
	[[nodiscard]] auto getFieldFactory(const std::string &fieldName) const -> std::shared_ptr<DerivedFieldFactory>;

	// Get total number of components needed for all derived fields
	[[nodiscard]] auto getTotalComponents() const -> int;

	// Get mapping of field names to component indices
	[[nodiscard]] auto getFieldComponentMapping() const -> std::map<std::string, std::pair<int, int>>;

      private:
	DerivedFieldManager() = default;
	std::vector<std::string> activeFields_;
	std::map<std::string, std::shared_ptr<DerivedFieldFactory>> fieldFactories_;
	std::map<std::string, std::pair<int, int>> fieldComponentMapping_; // fieldName -> (startComp, numComps)
	int totalComponents_{0};
};

#endif // DERIVED_FIELD_FACTORY_H