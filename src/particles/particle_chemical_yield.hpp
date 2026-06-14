#ifndef PARTICLE_CHEMICAL_YIELD_HPP_
#define PARTICLE_CHEMICAL_YIELD_HPP_

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "util/DataTable.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace quokka
{

namespace ChemicalYieldLookup
{

constexpr int kMaxTrackedIsotopes = 32;
constexpr int kMaxTrackedChannels = 3;
constexpr int kChemicalYieldTableNumIsotopes = 389;
constexpr amrex::Real kFixedYieldMetallicity = 0.02;

using FullChemicalYieldDataTable = quokka::DataTable<1, kChemicalYieldTableNumIsotopes, quokka::OutOfBounds::clamp>;
using SelectedChemicalYieldDataTable = quokka::DataTable<1, kMaxTrackedIsotopes, quokka::OutOfBounds::clamp>;

struct ChemicalYieldGpuConstTables {
	std::array<quokka::DataTableGpuConst<1, kMaxTrackedIsotopes, quokka::OutOfBounds::clamp>, kMaxTrackedChannels> channels{};
};

class ChemicalYieldTables
{
      public:
	std::array<SelectedChemicalYieldDataTable, kMaxTrackedChannels> channels{};

	[[nodiscard]] auto const_tables() const -> ChemicalYieldGpuConstTables
	{
		ChemicalYieldGpuConstTables tables{};
		for (int c = 0; c < kMaxTrackedChannels; ++c) {
			tables.channels[static_cast<std::size_t>(c)] = channels[static_cast<std::size_t>(c)].const_tables();
		}
		return tables;
	}
};

inline ChemicalYieldTables *g_tables = nullptr;					       // NOLINT
AMREX_GPU_MANAGED inline bool g_loaded = false;				       // NOLINT
AMREX_GPU_MANAGED inline int g_num_tracked_isotopes = 0;		       // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<int, kMaxTrackedChannels> g_channel_enabled{}; // NOLINT

inline auto mutableTables() -> ChemicalYieldTables &
{
	if (g_tables == nullptr) {
		g_tables = new ChemicalYieldTables(); // NOLINT(cppcoreguidelines-owning-memory)
	}
	return *g_tables;
}

inline auto lowercase(std::string s) -> std::string
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
}

inline auto resolveInputPath(const std::string &filename) -> std::filesystem::path
{
	std::filesystem::path candidate(filename);
	if (std::filesystem::exists(candidate)) {
		return candidate;
	}

	std::string trimmed = filename;
	while (trimmed.rfind("../", 0) == 0) {
		trimmed.erase(0, 3);
		const std::filesystem::path alt(trimmed);
		if (std::filesystem::exists(alt)) {
			return alt;
		}
	}

	return candidate;
}

inline auto channelName(int channel_index) -> std::string
{
	static const std::array<std::string, kMaxTrackedChannels> names{"snii", "wr", "agb"};
	return names.at(static_cast<std::size_t>(channel_index));
}

inline auto channelTableName(int channel_index) -> std::string
{
	static const std::array<std::string, kMaxTrackedChannels> names{"SNII_yield_table.csv", "WR_yield_table.csv", "AGB_yield_table.csv"};
	return names.at(static_cast<std::size_t>(channel_index));
}

inline auto requestedChannelMap(const std::vector<std::string> &tracked_channels) -> std::array<bool, kMaxTrackedChannels>
{
	std::array<bool, kMaxTrackedChannels> requested{true, true, true};
	if (!tracked_channels.empty()) {
		requested = {false, false, false};
		for (const auto &channel : tracked_channels) {
			const std::string name = lowercase(channel);
			for (int c = 0; c < kMaxTrackedChannels; ++c) {
				if (name == channelName(c)) {
					requested[static_cast<std::size_t>(c)] = true;
				}
			}
		}
	}
	return requested;
}

inline auto outputIndex(const FullChemicalYieldDataTable &table, const std::string &isotope_name) -> int
{
	const std::string requested = lowercase(isotope_name);
	const auto output_names = table.output_names();
	for (int i = 0; i < kChemicalYieldTableNumIsotopes; ++i) {
		if (lowercase(output_names[static_cast<std::size_t>(i)]) == requested) {
			return i;
		}
	}
	return -1;
}

inline auto physicalCoordinateBound(amrex::Real coord, quokka::SpacingType spacing) -> amrex::Real
{
	if (spacing == quokka::SpacingType::log) {
		return std::exp(coord);
	}
	if (spacing == quokka::SpacingType::fast_log) {
		return FastMath::pow2(coord);
	}
	return coord;
}

inline auto makeOutputNames(const std::vector<std::string> &tracked_isotopes) -> std::array<std::string, kMaxTrackedIsotopes>
{
	std::array<std::string, kMaxTrackedIsotopes> names{};
	for (int i = 0; i < kMaxTrackedIsotopes; ++i) {
		if (i < static_cast<int>(tracked_isotopes.size())) {
			names[static_cast<std::size_t>(i)] = lowercase(tracked_isotopes[static_cast<std::size_t>(i)]);
		} else {
			names[static_cast<std::size_t>(i)] = "unused_" + std::to_string(i);
		}
	}
	return names;
}

inline auto makeOutputUnits() -> std::array<std::string, kMaxTrackedIsotopes>
{
	std::array<std::string, kMaxTrackedIsotopes> units{};
	units.fill("fraction");
	return units;
}

inline auto makeZeroTable() -> SelectedChemicalYieldDataTable
{
	const std::array<amrex::Real, 1> x_mins{1.0};
	const std::array<amrex::Real, 1> x_maxs{2.0};
	const std::array<int, 1> n_xs{2};
	const std::array<quokka::SpacingType, 1> spacing{quokka::SpacingType::linear};
	const std::array<std::string, 1> input_names{"mass"};
	const std::array<std::string, 1> input_units{"Msun"};
	const auto output_names = makeOutputNames({});
	const auto output_units = makeOutputUnits();
	amrex::Vector<amrex::Real> flat_data(static_cast<std::size_t>(kMaxTrackedIsotopes * n_xs[0]), 0.0);
	return SelectedChemicalYieldDataTable::FromFlatData(x_mins, x_maxs, n_xs, spacing, flat_data, input_names, output_names, input_units,
							    output_units, quokka::SpacingType::linear);
}

inline auto loadChannelTable(const std::filesystem::path &table_path, int channel_index, const std::vector<std::string> &tracked_isotopes) -> bool
{
	if (!std::filesystem::exists(table_path)) {
		return false;
	}

	auto full_table = FullChemicalYieldDataTable::CSVReader(table_path.string(), quokka::SpacingType::linear);
	if (!full_table.is_initialized()) {
		return false;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(full_table.input_name(0) == "mass", "chemical yield tables must use 'mass' as the input coordinate");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(full_table.input_unit(0) == "Msun", "chemical yield table mass coordinate must use 'Msun'");

	const auto full_const = full_table.const_tables();
	const int num_entries = full_const.sizes[0];
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_entries >= 2, "chemical yield tables must contain at least two mass points");

	const auto spacing = full_const.spacing_types[0];
	const std::array<amrex::Real, 1> x_mins{physicalCoordinateBound(full_const.coord_min[0], spacing)};
	const std::array<amrex::Real, 1> x_maxs{physicalCoordinateBound(full_const.coord_max[0], spacing)};
	const std::array<int, 1> n_xs{num_entries};
	const std::array<quokka::SpacingType, 1> spacing_types{spacing};
	const std::array<std::string, 1> input_names{"mass"};
	const std::array<std::string, 1> input_units{"Msun"};
	const auto output_names = makeOutputNames(tracked_isotopes);
	const auto output_units = makeOutputUnits();

	amrex::Vector<amrex::Real> flat_data(static_cast<std::size_t>(kMaxTrackedIsotopes * num_entries), 0.0);
	for (int isotope_index = 0; isotope_index < g_num_tracked_isotopes; ++isotope_index) {
		const int out_idx = outputIndex(full_table, tracked_isotopes[static_cast<std::size_t>(isotope_index)]);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(out_idx >= 0, ("chemical yield isotope not found in table: " + tracked_isotopes[static_cast<std::size_t>(isotope_index)]).c_str());
		const auto data = full_const.dataViewArrays[static_cast<std::size_t>(out_idx)];
		for (int i = 0; i < num_entries; ++i) {
			flat_data[static_cast<std::size_t>(isotope_index * num_entries + i)] = std::max<amrex::Real>(data(i), 0.0);
		}
	}

	mutableTables().channels[static_cast<std::size_t>(channel_index)] =
	    SelectedChemicalYieldDataTable::FromFlatData(x_mins, x_maxs, n_xs, spacing_types, flat_data, input_names, output_names, input_units, output_units,
							 quokka::SpacingType::linear);
	g_channel_enabled[channel_index] = 1;
	return true;
}

inline auto loadTable(const std::string &filename, const std::vector<std::string> &tracked_isotopes, const std::vector<std::string> &tracked_channels)
    -> bool
{
	g_loaded = false;
	g_num_tracked_isotopes = std::min(static_cast<int>(tracked_isotopes.size()), kMaxTrackedIsotopes);
	auto &tables = mutableTables();
	for (int c = 0; c < kMaxTrackedChannels; ++c) {
		g_channel_enabled[c] = 0;
		tables.channels[static_cast<std::size_t>(c)] = makeZeroTable();
	}
	if (g_num_tracked_isotopes <= 0) {
		return false;
	}

	const std::filesystem::path input_path = resolveInputPath(filename);
	const std::filesystem::path table_dir = std::filesystem::is_directory(input_path) ? input_path : input_path.parent_path();
	const auto requested_channels = requestedChannelMap(tracked_channels);

	bool loaded_any = false;
	for (int c = 0; c < kMaxTrackedChannels; ++c) {
		if (!requested_channels[static_cast<std::size_t>(c)]) {
			continue;
		}
		const std::filesystem::path table_path = table_dir / channelTableName(c);
		loaded_any = loadChannelTable(table_path, c, tracked_isotopes) || loaded_any;
	}

	g_loaded = loaded_any;
	return g_loaded;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto isLoaded() -> bool { return g_loaded && (g_num_tracked_isotopes > 0); }

inline auto constTables() -> ChemicalYieldGpuConstTables { return mutableTables().const_tables(); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto queryYieldFraction(ChemicalYieldGpuConstTables const &tables, int channel_index, int isotope_index,
								 amrex::Real mass_msun, amrex::Real /*metallicity*/) -> amrex::Real
{
	if (!isLoaded() || channel_index < 0 || isotope_index < 0 || channel_index >= kMaxTrackedChannels ||
	    isotope_index >= g_num_tracked_isotopes || g_channel_enabled[channel_index] == 0) {
		return 0.0;
	}

	std::array<amrex::Real, 1> const point{mass_msun};
	const auto values = tables.channels[static_cast<std::size_t>(channel_index)].interpolate(point);
	return std::max<amrex::Real>(values[static_cast<std::size_t>(isotope_index)], 0.0);
}

} // namespace ChemicalYieldLookup

} // namespace quokka

#endif // PARTICLE_CHEMICAL_YIELD_HPP_
