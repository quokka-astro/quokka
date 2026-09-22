#ifndef INITIAL_STELLAR_PARTICLES_HPP_
#define INITIAL_STELLAR_PARTICLES_HPP_

#include "AMReX_ParallelDescriptor.H"
#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"

#include <vector>

namespace quokka::testing
{
template <typename problem_t> auto initialStellarParticles(QuokkaSimulation<problem_t> &sim) -> std::vector<std::vector<double>>
{
	auto records = sim.particleRegister_.getParticleDescriptor(ParticleType::StochasticStellarPop)->getParticleDataAtLevel(0).first;
	int count = static_cast<int>(records.size());
	const int root = amrex::ParallelDescriptor::IOProcessorNumber();
	amrex::ParallelDescriptor::Bcast(&count, 1, root);
	records.resize(static_cast<std::size_t>(count));
	constexpr int components = AMREX_SPACEDIM + StochasticStellarPopParticleRealComps<problem_t>;
	for (auto &record : records) {
		record.resize(components);
		amrex::ParallelDescriptor::Bcast(record.data(), components, root);
	}
	return records;
}
} // namespace quokka::testing

#endif
