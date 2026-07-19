#ifndef IO_UTILS_HPP_
#define IO_UTILS_HPP_
// ABOUTME: RAII utilities for managing AMReX I/O settings
// ABOUTME: Ensures global I/O settings are properly restored after use

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_VisMF.H"

namespace quokka
{

// RAII helper class to temporarily set AMReX IO settings
class ScopedVisMFNOutFiles
{
      private:
	int originalNOutFiles_;

      public:
	explicit ScopedVisMFNOutFiles(int nfiles) : originalNOutFiles_(amrex::VisMF::GetNOutFiles())
	{
		const int nfiles_to_set = (nfiles == -1) ? amrex::ParallelDescriptor::NProcs() : nfiles;
		amrex::VisMF::SetNOutFiles(nfiles_to_set);
	}

	~ScopedVisMFNOutFiles()
	{
		// Restore original value
		amrex::VisMF::SetNOutFiles(originalNOutFiles_);
	}

	// Delete copy and move operations to ensure RAII semantics
	ScopedVisMFNOutFiles(const ScopedVisMFNOutFiles &) = delete;
	auto operator=(const ScopedVisMFNOutFiles &) -> ScopedVisMFNOutFiles & = delete;
	ScopedVisMFNOutFiles(ScopedVisMFNOutFiles &&) = delete;
	auto operator=(ScopedVisMFNOutFiles &&) -> ScopedVisMFNOutFiles & = delete;
};

} // namespace quokka

#endif // IO_UTILS_HPP_
