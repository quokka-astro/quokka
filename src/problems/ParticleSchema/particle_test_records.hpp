#ifndef QUOKKA_PARTICLE_TEST_RECORDS_HPP_
#define QUOKKA_PARTICLE_TEST_RECORDS_HPP_

#include "particles/particle_IO.hpp"

namespace particle_test
{
using Record = std::tuple<int64_t, int, std::vector<double>, std::vector<int>>;

inline auto Records(const std::vector<int64_t> &ids, const std::vector<int> &cpus, const std::vector<std::vector<double>> &reals,
		    const std::vector<std::vector<int>> &integers) -> std::vector<Record>
{
	AMREX_ALWAYS_ASSERT(ids.size() == cpus.size() && ids.size() == reals.size());
	AMREX_ALWAYS_ASSERT(integers.empty() || integers.size() == ids.size());
	std::vector<Record> rows;
	for (size_t i = 0; i < ids.size(); ++i) {
		rows.emplace_back(ids[i], cpus[i], reals[i], integers.empty() ? std::vector<int>{} : integers[i]);
	}
	std::sort(rows.begin(), rows.end());
	return rows;
}

inline auto CaptureRecords(quokka::ParticleContainer &pc) -> std::vector<Record>
{
	const auto [ids, reals, integers] = quokka::particle_io::getParticleDataAtAllLevels(&pc);
	std::vector<int> cpus;
	for (int lev = 0; lev <= pc.finestLevel(); ++lev) {
		for (const auto &[key, tile] : pc.GetParticles(lev)) {
			amrex::ignore_unused(key);
			const auto data = tile.getConstParticleTileData();
			const auto count = tile.numParticles();
			amrex::Gpu::DeviceVector<int> packed(2 * count);
			auto *out = packed.data();
			amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int i) {
				out[2 * i] = data.id(i).is_valid();
				out[2 * i + 1] = data.cpu(i);
			});
			std::vector<int> host(packed.size());
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, packed.begin(), packed.end(), host.begin());
			for (amrex::Long i = 0; i < count; ++i) {
				if (host[2 * i] != 0) {
					cpus.push_back(host[2 * i + 1]);
				}
			}
		}
	}
	return Records(ids, quokka::particle_io::gatherBuffer(cpus), reals, integers);
}

// AoS access is restricted to fixtures that write the pre-migration format.
template <typename Legacy> auto CaptureLegacyRecords(const Legacy &pc) -> std::vector<Record>
{
	std::vector<int64_t> ids;
	std::vector<int> cpus, integers;
	std::vector<double> reals;
	constexpr int nr = AMREX_SPACEDIM + Legacy::ParticleType::NReal;
	constexpr int ni = Legacy::ParticleType::NInt;
	for (int lev = 0; lev <= pc.finestLevel(); ++lev) {
		for (const auto &[key, tile] : pc.GetParticles(lev)) {
			amrex::ignore_unused(key);
			amrex::Gpu::HostVector<typename Legacy::ParticleType> host(tile.numParticles());
			const auto &aos = tile.GetArrayOfStructs();
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, aos().begin(), aos().begin() + tile.numParticles(), host.begin());
			for (const auto &p : host) {
				if (!p.id().is_valid()) {
					continue;
				}
				ids.push_back(p.id());
				cpus.push_back(p.cpu());
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					reals.push_back(p.pos(d));
				}
				for (int c = 0; c < Legacy::ParticleType::NReal; ++c) {
					reals.push_back(p.rdata(c));
				}
				for (int c = 0; c < ni; ++c) {
					integers.push_back(p.idata(c));
				}
			}
		}
	}
	ids = quokka::particle_io::gatherBuffer(ids);
	cpus = quokka::particle_io::gatherBuffer(cpus);
	reals = quokka::particle_io::gatherBuffer(reals);
	integers = quokka::particle_io::gatherBuffer(integers);
	std::vector<std::vector<double>> realRows(ids.size());
	std::vector<std::vector<int>> intRows(ni > 0 ? ids.size() : 0);
	for (size_t i = 0; i < ids.size(); ++i) {
		realRows[i].assign(reals.begin() + i * nr, reals.begin() + (i + 1) * nr);
		if constexpr (ni > 0) {
			intRows[i].assign(integers.begin() + i * ni, integers.begin() + (i + 1) * ni);
		}
	}
	return Records(ids, cpus, realRows, intRows);
}
} // namespace particle_test
#endif
