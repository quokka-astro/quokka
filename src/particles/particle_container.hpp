#ifndef QUOKKA_PARTICLE_CONTAINER_HPP_
#define QUOKKA_PARTICLE_CONTAINER_HPP_

#include "AMReX_AmrCore.H"
#include "AMReX_AmrParGDB.H"
#include "AMReX_AsyncOut.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_ParticleContainer.H"
#include "AMReX_ParticleHeader.H"
#include "particles/particle_schema.hpp"

namespace quokka
{
namespace detail
{
// CUDA extended lambdas and their captured types must be accessible to the
// kernel launcher. Keep the launch outside the container's private methods.
template <typename T> void FillParticleComponent(T *values, amrex::Long begin, amrex::Long end, T value)
{
	amrex::ParallelFor(end - begin, [=] AMREX_GPU_DEVICE(amrex::Long offset) noexcept { values[begin + offset] = value; });
}
} // namespace detail

class ParticleContainer : public amrex::ParticleContainerRTSoA<>
{
      public:
	using Base = amrex::ParticleContainerRTSoA<>;

	template <typename... Args>
	requires(!(sizeof...(Args) == 1 && (std::is_convertible_v<Args, amrex::AmrCore *> && ...))) explicit ParticleContainer(ParticleSchema schema,
															       Args &&...args)
	    : Base(std::forward<Args>(args)...), schema_(std::move(schema))
	{
		Configure();
	}
	ParticleContainer(ParticleSchema schema, amrex::AmrCore *amr) : ParticleContainer(std::move(schema), amr->GetParGDB()) {}
	ParticleContainer(const ParticleContainer &) = delete;
	ParticleContainer(ParticleContainer &&) = delete;
	auto operator=(const ParticleContainer &) -> ParticleContainer & = delete;
	auto operator=(ParticleContainer &&) -> ParticleContainer & = delete;
	~ParticleContainer() override = default;

	[[nodiscard]] auto Schema() const -> const ParticleSchema & { return schema_; }
	[[nodiscard]] auto Layout() const -> ParticleLayout { return schema_.Layout(); }
	[[nodiscard]] static auto ReserveIDs(amrex::Long count) -> amrex::Long
	{
		RequireCreationReady();
		AMREX_ALWAYS_ASSERT(count >= 0);
		if (count == 0) {
			return 0;
		}
		const auto first = ParticleType::NextID();
		AMREX_ALWAYS_ASSERT(count - 1 <= amrex::LongParticleIds::LastParticleID - first);
		ParticleType::NextID(first + count);
		return first;
	}

	// All fields are initialized before the caller receives a new particle range.
	// ID/CPU allocation is left to the caller's one allocation per new particle.
	void ResizeAndInitialize(ParticleTileType &tile, amrex::Long size)
	{
		BL_PROFILE("ParticleContainer::ResizeAndInitialize()"); // NOLINT(misc-const-correctness)
		RequireCreationReady();
		AMREX_ALWAYS_ASSERT(size >= 0);
		const auto oldSize = tile.size();
		{
			BL_PROFILE("ParticleStorage::resize()"); // NOLINT(misc-const-correctness)
			tile.resize(size);
		}
		if (size <= oldSize) {
			return;
		}
		const auto data = tile.getParticleTileData();
		const int rank = amrex::ParallelDescriptor::MyProc();
		{
			BL_PROFILE("ParticleContainer::InitializeIdentity()"); // NOLINT(misc-const-correctness)
			amrex::ParallelFor(size - oldSize, [=] AMREX_GPU_DEVICE(amrex::Long offset) noexcept {
				const auto i = offset + oldSize;
				data.id(i) = 0;
				data.cpu(i) = rank;
				for (int d = 0; d < AMREX_SPACEDIM; ++d) {
					data.pos(d, i) = 0;
				}
			});
		}
		SetDefaults(tile, oldSize, size, 0);
	}

	void InitRandom(amrex::Long count, amrex::ULong seed, const ParticleInitData &data = {}, bool serialize = false, amrex::RealBox box = amrex::RealBox())
	{
		Base::InitRandom(count, seed, CompleteDefaults(data), serialize, box);
	}
	void InitOnePerCell(amrex::Real x, amrex::Real y, amrex::Real z, const ParticleInitData &data = {})
	{
		Base::InitOnePerCell(x, y, z, CompleteDefaults(data));
	}
	void InitFromAsciiFile(const std::string &file, int extra, const amrex::IntVect *replicas = nullptr)
	{
		RequireCreationReady();
		AMREX_ALWAYS_ASSERT(extra >= 0 && extra <= Layout().numReal);
		ParticleContainer incoming(schema_, GetParGDB());
		incoming.Base::InitFromAsciiFile(file, extra, replicas);
		for (int lev = 0; lev <= incoming.finestLevel(); ++lev) {
			for (auto &[key, tile] : incoming.GetParticles(lev)) {
				amrex::ignore_unused(key);
				incoming.SetDefaults(tile, 0, tile.size(), extra);
			}
		}
		amrex::Gpu::streamSynchronize();
		Base::addParticles(incoming);
		Redistribute();
	}

	void WritePlotFile(const std::string &directory, const std::string &name) const
	{
		ValidateIO();
		Base::WritePlotFile(directory, name, schema_.RealNames(), schema_.IntNames());
	}
	void Checkpoint(const std::string &directory, const std::string &name, bool includeHeader = true) const
	{
		ValidateIO();
		Base::Checkpoint(directory, name, includeHeader, schema_.RealNames(), schema_.IntNames());
	}
	void Restart(const std::string &directory, const std::string &name)
	{
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(TotalNumberOfParticles(false) == 0, "Particle restart requires an empty destination");
		const auto header = amrex::ParticleHeader::read(directory, name);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(header.real_comp_names == schema_.RealNames() && header.int_comp_names == schema_.IntNames(),
						 "Particle restart field names/order do not match the population schema");
		AMREX_ALWAYS_ASSERT(header.dim == AMREX_SPACEDIM);
		AMREX_ALWAYS_ASSERT((header.how == "double" && sizeof(amrex::ParticleReal) == sizeof(double)) ||
				    (header.how == "single" && sizeof(amrex::ParticleReal) == sizeof(float)));
		// AMReX resets the shared counter for each population. Preserve every
		// requirement until the high-level restart has restored all populations.
		const auto previous = ParticleType::NextID();
		ParticleType::NextID(previous);
		pendingNextID_ = std::max({pendingNextID_, previous, header.next_id});
		Base::Restart(directory, name);
		restartNextID_ = header.next_id;
		amrex::ReduceOps<amrex::ReduceOpMax> op;
		amrex::ReduceData<amrex::Long> result(op);
		using Tuple = decltype(result)::Type;
		for (int lev = 0; lev <= finestLevel(); ++lev) {
			for (const auto &[key, tile] : GetParticles(lev)) {
				amrex::ignore_unused(key);
				const auto data = tile.getConstParticleTileData();
				op.eval(tile.numParticles(), result,
					[=] AMREX_GPU_DEVICE(int i) -> Tuple { return {data.id(i).is_valid() ? static_cast<amrex::Long>(data.id(i)) : 0}; });
			}
		}
		const auto maximum = std::max(static_cast<amrex::Long>(0), amrex::get<0>(result.value(op)));
		pendingNextID_ = std::max(pendingNextID_, maximum + 1);
	}
	// Register a restored population before the shared finalization step.
	void RegisterRestartNextID(amrex::Long next)
	{
		const auto previous = ParticleType::NextID();
		ParticleType::NextID(previous);
		AMREX_ALWAYS_ASSERT(next > 0 && next <= amrex::LongParticleIds::LastParticleID);
		restartNextID_ = next;
		pendingNextID_ = std::max({pendingNextID_, previous, next});
	}
	[[nodiscard]] auto RestartNextID() const -> amrex::Long { return restartNextID_; }
	// Collective: invoke once after every population, including tracers, has
	// restarted and before any initialization or physics creation is allowed.
	static void FinalizeRestarts()
	{
		amrex::ParallelDescriptor::ReduceLongMax(pendingNextID_);
		if (pendingNextID_ >= 0) {
			ParticleType::NextID(pendingNextID_);
			pendingNextID_ = -1;
		}
	}
	[[nodiscard]] static auto RestartsPending() -> bool { return pendingNextID_ >= 0; }

      private:
	static void RequireCreationReady() { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!RestartsPending(), "Finalize all particle restarts before creating particles"); }
	void Configure()
	{
		Base::SetArena(amrex::The_Arena());
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			Base::AddRealComp("__position_" + std::to_string(d));
		}
		for (const auto &name : schema_.RealNames()) {
			Base::AddRealComp(name);
		}
		for (const auto &name : schema_.IntNames()) {
			Base::AddIntComp(name);
		}
	}
	void ValidateIO() const
	{
		AMREX_ALWAYS_ASSERT(NumRealComps() == AMREX_SPACEDIM + Layout().numReal && NumIntComps() == Layout().numInt);
		AMREX_ALWAYS_ASSERT(!amrex::AsyncOut::UseAsyncOut());
	}
	[[nodiscard]] auto CompleteDefaults(const ParticleInitData &supplied) const -> ParticleInitData
	{
		RequireCreationReady();
		AMREX_ALWAYS_ASSERT(supplied.real_array_data.size() <= schema_.RealDefaults().size());
		AMREX_ALWAYS_ASSERT(supplied.int_array_data.size() <= schema_.IntDefaults().size());
		ParticleInitData result;
		result.real_array_data.assign(schema_.RealDefaults().begin(), schema_.RealDefaults().end());
		result.int_array_data = schema_.IntDefaults();
		std::copy(supplied.real_array_data.begin(), supplied.real_array_data.end(), result.real_array_data.begin());
		std::copy(supplied.int_array_data.begin(), supplied.int_array_data.end(), result.int_array_data.begin());
		return result;
	}
	void SetDefaults(ParticleTileType &tile, amrex::Long begin, amrex::Long end, int firstReal)
	{
		BL_PROFILE("ParticleContainer::SetDefaults()"); // NOLINT(misc-const-correctness)
		const auto data = tile.getParticleTileData();
		for (int c = firstReal; c < Layout().numReal; ++c) {
			const auto value = schema_.RealDefaults()[c];
			const int storage = Layout().RealStorage(c);
			detail::FillParticleComponent(data.rdata(storage), begin, end, value);
		}
		for (int c = 0; c < Layout().numInt; ++c) {
			const int value = schema_.IntDefaults()[c];
			detail::FillParticleComponent(data.idata(c), begin, end, value);
		}
	}

	// Hide unsupported entry points on the Quokka interface. Explicit casts to
	// AMReX are reserved for the implementation of this wrapper.
	using Base::AddIntComp;
	using Base::addParticles;
	using Base::AddParticlesAtLevel;
	using Base::AddRealComp;
	using Base::copyParticles;
	using Base::CreateGhostParticles;
	using Base::CreateVirtualParticles;
	using Base::InitFromBinaryFile;
	using Base::InitFromBinaryMetaFile;
	using Base::InitNRandomPerCell;
	using Base::InitRandomPerBox;
	using Base::make_alike;
	using Base::ResizeRuntimeIntComp;
	using Base::ResizeRuntimeRealComp;
	using Base::SetArena;
	using Base::SetSoACompileTimeNames;
	using Base::WriteAsciiFile;
	using Base::WriteBinaryParticleData;

	const ParticleSchema schema_;
	amrex::Long restartNextID_ = 1;
	inline static amrex::Long pendingNextID_ = -1;
};
} // namespace quokka
#endif
