#include "QuokkaSimulation.hpp"
#include "derived_fields/DerivedFieldFactory.H"

template <typename problem_t>
void QuokkaSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const
{
	// Try to compute using the derived field factory first
	auto &manager = DerivedFieldManager::getInstance();
	auto factory = manager.getFieldFactory(dname);
	
	if (factory != nullptr) {
		// Use the factory to compute the derived field
		factory->compute(mf, state_new_cc_[lev], this->geom[lev], this->t_new[lev], ncomp);
		amrex::Gpu::streamSynchronizeAll();
		return;
	}
	
	// Fall back to problem-specific implementation if factory doesn't handle this field
	// This allows for backward compatibility with existing problem-specific derived fields
	ComputeDerivedVarImpl(lev, dname, mf, ncomp);
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::ComputeDerivedVarImpl(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const
{
	// Default implementation - problems can override this for custom derived fields
	// that are not handled by the factory system
	amrex::Abort("ComputeDerivedVar: Unknown derived variable '" + dname + "'");
}

// Note: Explicit template instantiation is handled by each problem's implementation
// since QuokkaSimulation is a template class that gets specialized for each problem type
