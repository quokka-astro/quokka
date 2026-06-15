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
#include <utility>
#include <vector>

namespace quokka
{

namespace ChemicalYieldLookup
{

constexpr int max_tracked_isotopes = 32;
constexpr int max_tracked_channels = 3;
constexpr int chemical_yield_table_num_isotopes = 389;

using FullChemicalYieldDataTable = quokka::DataTable<1, chemical_yield_table_num_isotopes, quokka::OutOfBounds::clamp>;
using SelectedChemicalYieldDataTable = quokka::DataTable<1, max_tracked_isotopes, quokka::OutOfBounds::clamp>;
using WRMassLossDistributionDataTable = quokka::DataTable<2, 1, quokka::OutOfBounds::clamp>;

struct ChemicalYieldGpuConstTables {
	std::array<quokka::DataTableGpuConst<1, max_tracked_isotopes, quokka::OutOfBounds::clamp>, max_tracked_channels> channels{};
	quokka::DataTableGpuConst<2, 1, quokka::OutOfBounds::clamp> wr_mass_loss_distribution{};
};

class ChemicalYieldTables
{
      public:
	std::array<SelectedChemicalYieldDataTable, max_tracked_channels> channels{};
	WRMassLossDistributionDataTable wr_mass_loss_distribution{};

	[[nodiscard]] auto const_tables() const -> ChemicalYieldGpuConstTables
	{
		ChemicalYieldGpuConstTables tables{};
		for (int c = 0; c < max_tracked_channels; ++c) {
			tables.channels[static_cast<std::size_t>(c)] = channels[static_cast<std::size_t>(c)].const_tables();
		}
		tables.wr_mass_loss_distribution = wr_mass_loss_distribution.const_tables();
		return tables;
	}
};

inline ChemicalYieldTables *tables_ptr = nullptr;				       // NOLINT
AMREX_GPU_MANAGED inline bool tables_loaded = false;				       // NOLINT
AMREX_GPU_MANAGED inline bool wr_mass_loss_distribution_loaded = false;		       // NOLINT
AMREX_GPU_MANAGED inline int num_tracked_isotopes = 0;				       // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<int, max_tracked_channels> channel_enabled{}; // NOLINT

inline auto mutableTables() -> ChemicalYieldTables &
{
	if (tables_ptr == nullptr) {
		tables_ptr = new ChemicalYieldTables(); // NOLINT(cppcoreguidelines-owning-memory)
	}
	return *tables_ptr;
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
	static const std::array<std::string, max_tracked_channels> names{"snii", "wr", "agb"};
	return names.at(static_cast<std::size_t>(channel_index));
}

inline auto channelTableName(int channel_index) -> std::string
{
	static const std::array<std::string, max_tracked_channels> names{"SNII_yield_table.csv", "WR_yield_table.csv", "AGB_yield_table.csv"};
	return names.at(static_cast<std::size_t>(channel_index));
}

inline auto requestedChannelMap(const std::vector<std::string> &tracked_channels) -> std::array<bool, max_tracked_channels>
{
	std::array<bool, max_tracked_channels> requested{true, true, true};
	if (!tracked_channels.empty()) {
		requested = {false, false, false};
		for (const auto &channel : tracked_channels) {
			const std::string name = lowercase(channel);
			for (int c = 0; c < max_tracked_channels; ++c) {
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
	for (int i = 0; i < chemical_yield_table_num_isotopes; ++i) {
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

inline auto makeOutputNames(const std::vector<std::string> &tracked_isotopes) -> std::array<std::string, max_tracked_isotopes>
{
	std::array<std::string, max_tracked_isotopes> names{};
	for (int i = 0; i < max_tracked_isotopes; ++i) {
		if (i < static_cast<int>(tracked_isotopes.size())) {
			names[static_cast<std::size_t>(i)] = lowercase(tracked_isotopes[static_cast<std::size_t>(i)]);
		} else {
			names[static_cast<std::size_t>(i)] = "unused_" + std::to_string(i);
		}
	}
	return names;
}

inline auto makeOutputUnits() -> std::array<std::string, max_tracked_isotopes>
{
	std::array<std::string, max_tracked_isotopes> units{};
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
	amrex::Vector<amrex::Real> flat_data(static_cast<std::size_t>(max_tracked_isotopes * n_xs[0]), 0.0);
	return SelectedChemicalYieldDataTable::FromFlatData(x_mins, x_maxs, n_xs, spacing, flat_data, input_names, output_names, input_units, output_units,
							    quokka::SpacingType::linear);
}

inline auto makeZeroWRMassLossDistributionTable() -> WRMassLossDistributionDataTable
{
	const std::array<amrex::Real, 2> x_mins{0.0, 1.0};
	const std::array<amrex::Real, 2> x_maxs{1.0, 2.0};
	const std::array<int, 2> n_xs{2, 2};
	const std::array<quokka::SpacingType, 2> spacing{quokka::SpacingType::linear, quokka::SpacingType::linear};
	const std::array<std::string, 2> input_names{"age", "mass"};
	const std::array<std::string, 1> output_names{"cumulative_fraction"};
	const std::array<std::string, 2> input_units{"s", "Msun"};
	const std::array<std::string, 1> output_units{"fraction"};
	amrex::Vector<amrex::Real> flat_data(4, 0.0);
	return WRMassLossDistributionDataTable::FromFlatData(x_mins, x_maxs, n_xs, spacing, flat_data, input_names, output_names, input_units, output_units,
							     quokka::SpacingType::linear);
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

	amrex::Vector<amrex::Real> flat_data(static_cast<std::size_t>(max_tracked_isotopes * num_entries), 0.0);
	for (int isotope_index = 0; isotope_index < num_tracked_isotopes; ++isotope_index) {
		const int out_idx = outputIndex(full_table, tracked_isotopes[static_cast<std::size_t>(isotope_index)]);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    out_idx >= 0, ("chemical yield isotope not found in table: " + tracked_isotopes[static_cast<std::size_t>(isotope_index)]).c_str());
		const auto data = full_const.dataViewArrays[static_cast<std::size_t>(out_idx)];
		for (int i = 0; i < num_entries; ++i) {
			flat_data[static_cast<std::size_t>(isotope_index * num_entries + i)] = std::max<amrex::Real>(data(i), 0.0);
		}
	}

	mutableTables().channels[static_cast<std::size_t>(channel_index)] = SelectedChemicalYieldDataTable::FromFlatData(
	    x_mins, x_maxs, n_xs, spacing_types, flat_data, input_names, output_names, input_units, output_units, quokka::SpacingType::linear);
	channel_enabled[channel_index] = 1;
	return true;
}

inline auto loadWRMassLossDistributionTable(const std::filesystem::path &table_path) -> bool
{
	if (!std::filesystem::exists(table_path)) {
		return false;
	}

	auto distribution_table = WRMassLossDistributionDataTable::CSVReader(table_path.string(), quokka::SpacingType::linear);
	if (!distribution_table.is_initialized()) {
		return false;
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(distribution_table.input_name(0) == "age", "WR mass-loss distribution table must use 'age' as input 0");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(distribution_table.input_name(1) == "mass", "WR mass-loss distribution table must use 'mass' as input 1");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(distribution_table.input_unit(0) == "s", "WR mass-loss distribution age coordinate must use 's'");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(distribution_table.input_unit(1) == "Msun", "WR mass-loss distribution mass coordinate must use 'Msun'");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(distribution_table.output_name(0) == "cumulative_fraction",
					 "WR mass-loss distribution table must output 'cumulative_fraction'");

	mutableTables().wr_mass_loss_distribution = std::move(distribution_table);
	wr_mass_loss_distribution_loaded = true;
	return true;
}

inline auto loadTable(const std::string &filename, const std::vector<std::string> &tracked_isotopes, const std::vector<std::string> &tracked_channels) -> bool
{
	tables_loaded = false;
	wr_mass_loss_distribution_loaded = false;
	num_tracked_isotopes = std::min(static_cast<int>(tracked_isotopes.size()), max_tracked_isotopes);
	auto &tables = mutableTables();
	for (int c = 0; c < max_tracked_channels; ++c) {
		channel_enabled[c] = 0;
		tables.channels[static_cast<std::size_t>(c)] = makeZeroTable();
	}
	tables.wr_mass_loss_distribution = makeZeroWRMassLossDistributionTable();
	if (num_tracked_isotopes <= 0) {
		return false;
	}

	const std::filesystem::path input_path = resolveInputPath(filename);
	const std::filesystem::path table_dir = std::filesystem::is_directory(input_path) ? input_path : input_path.parent_path();
	const auto requested_channels = requestedChannelMap(tracked_channels);

	bool loaded_any = false;
	for (int c = 0; c < max_tracked_channels; ++c) {
		if (!requested_channels[static_cast<std::size_t>(c)]) {
			continue;
		}
		const std::filesystem::path table_path = table_dir / channelTableName(c);
		const bool loaded_channel = loadChannelTable(table_path, c, tracked_isotopes);
		if (loaded_channel && c == 1) {
			const std::filesystem::path wr_distribution_path = table_dir / "WR_mass_loss_distribution_table.csv";
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(loadWRMassLossDistributionTable(wr_distribution_path),
							 ("failed to load WR mass-loss distribution table: " + wr_distribution_path.string()).c_str());
		}
		loaded_any = loaded_channel || loaded_any;
	}

	tables_loaded = loaded_any;
	return tables_loaded;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto isLoaded() -> bool { return tables_loaded && (num_tracked_isotopes > 0); }

inline auto constTables() -> ChemicalYieldGpuConstTables { return mutableTables().const_tables(); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto queryYieldFraction(ChemicalYieldGpuConstTables const &tables, int channel_index, int isotope_index,
								 amrex::Real mass_msun, amrex::Real /*metallicity*/) -> amrex::Real
{
	if (!isLoaded() || channel_index < 0 || isotope_index < 0 || channel_index >= max_tracked_channels || isotope_index >= num_tracked_isotopes ||
	    channel_enabled[channel_index] == 0) {
		return 0.0;
	}

	std::array<amrex::Real, 1> const point{mass_msun};
	const auto values = tables.channels[static_cast<std::size_t>(channel_index)].interpolate(point);
	return std::max<amrex::Real>(values[static_cast<std::size_t>(isotope_index)], 0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto queryWRMassLossCumulativeFraction(ChemicalYieldGpuConstTables const &tables, amrex::Real age,
										amrex::Real mass_msun) -> amrex::Real
{
	if (!isLoaded() || !wr_mass_loss_distribution_loaded || mass_msun <= 0.0) {
		return 0.0;
	}

	std::array<amrex::Real, 2> const point{std::max<amrex::Real>(0.0, age), mass_msun};
	const auto values = tables.wr_mass_loss_distribution.interpolate(point);
	return std::min<amrex::Real>(1.0, std::max<amrex::Real>(0.0, values[0]));
}

} // namespace ChemicalYieldLookup

} // namespace quokka

#endif // PARTICLE_CHEMICAL_YIELD_HPP_
