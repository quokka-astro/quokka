#include "DerivedFieldFactory.H"
#include "AMReX_Print.H"

auto DerivedFieldManager::getInstance() -> DerivedFieldManager &
{
	static DerivedFieldManager instance;
	return instance;
}

void DerivedFieldManager::initFromParmParse()
{
	amrex::ParmParse pp("derived_fields");

	// Read derived field names from input
	amrex::Vector<std::string> fieldNames;
	pp.queryarr("fields", fieldNames);

	activeFields_.clear();
	fieldFactories_.clear();
	fieldComponentMapping_.clear();
	totalComponents_ = 0;

	int currentComponent = 0;

	for (const auto &fieldName : fieldNames) {
		try {
			// Create field factory
			auto factory = DerivedFieldFactory::create(fieldName);

			// Initialize with field-specific parameters
			amrex::ParmParse fieldPP(fieldName);
			factory->init(fieldName, fieldPP);

			// Store factory
			fieldFactories_[fieldName] = factory;
			activeFields_.push_back(fieldName);

			// Update component mapping
			int numComps = factory->getNumComponents();
			fieldComponentMapping_[fieldName] = {currentComponent, numComps};
			currentComponent += numComps;
			totalComponents_ += numComps;

			amrex::Print() << "Registered derived field: " << fieldName << " with " << numComps << " components" << std::endl;
		} catch (const std::exception &e) {
			amrex::Print() << "Warning: Failed to create derived field '" << fieldName << "': " << e.what() << std::endl;
		}
	}
}

auto DerivedFieldManager::getActiveFields() const -> const std::vector<std::string> & { return activeFields_; }

void DerivedFieldManager::computeField(const std::string &fieldName, amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom,
				       amrex::Real time, int ncomp) const
{
	auto it = fieldFactories_.find(fieldName);
	if (it != fieldFactories_.end()) {
		it->second->compute(mf, state, geom, time, ncomp);
	} else {
		amrex::Abort("DerivedFieldManager::computeField: Field '" + fieldName + "' not found");
	}
}

auto DerivedFieldManager::getFieldFactory(const std::string &fieldName) const -> std::shared_ptr<DerivedFieldFactory>
{
	auto it = fieldFactories_.find(fieldName);
	if (it != fieldFactories_.end()) {
		return it->second;
	}
	return nullptr;
}

auto DerivedFieldManager::getTotalComponents() const -> int { return totalComponents_; }

auto DerivedFieldManager::getFieldComponentMapping() const -> std::map<std::string, std::pair<int, int>> { return fieldComponentMapping_; }