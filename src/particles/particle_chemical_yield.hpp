#ifndef PARTICLE_CHEMICAL_YIELD_HPP_
#define PARTICLE_CHEMICAL_YIELD_HPP_

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace quokka
{

// #Chuhan_start: Provides an interface for looking up, loading and querying the chemical yields of isotopes
namespace ChemicalYieldLookup
{

constexpr int kMaxTrackedIsotopes = 32;
constexpr int kMaxTrackedChannels = 5; // first version: SNII; 2nd version WR, AGB; 3rd version: SNIa, NSM
constexpr int kMaxChemicalYieldEntries = 4096;

AMREX_GPU_MANAGED inline int g_num_entries = 0; // NOLINT
AMREX_GPU_MANAGED inline bool g_loaded = false; // NOLINT
AMREX_GPU_MANAGED inline int g_num_tracked_isotopes = 0; // NOLINT
AMREX_GPU_MANAGED inline int g_num_tracked_channels = 0; // NOLINT

AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries> g_mass_msun{};     // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries> g_metallicity{};   // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries * kMaxTrackedIsotopes> g_snii_frac{}; // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries * kMaxTrackedIsotopes> g_wr_frac{};   // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries * kMaxTrackedIsotopes> g_agb_frac{};  // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries * kMaxTrackedChannels * kMaxTrackedIsotopes> g_channel_iso_frac{}; // NOLINT

inline auto lowercase(std::string s) -> std::string;

inline auto normalizeNumericToken(std::string token) -> std::string;

inline auto trackedIsotopeIndex(const std::map<std::string, int> &tracked_isotopes, const std::string &name) -> int
{
	const auto it = tracked_isotopes.find(lowercase(name));
	if (it == tracked_isotopes.end()) {
		return -1;
	}
	return it->second;
}

inline auto trackedChannelIndex(const std::map<std::string, int> &tracked_channels, const std::string &name) -> int
{
	const auto it = tracked_channels.find(lowercase(name));
	if (it == tracked_channels.end()) {
		return -1;
	}
	return it->second;
}

struct YieldEntry {
	amrex::Real mass_msun = 0.0;
	amrex::Real metallicity = 0.0;
	amrex::GpuArray<amrex::Real, kMaxTrackedIsotopes> snii_frac{};
	amrex::GpuArray<amrex::Real, kMaxTrackedIsotopes> wr_frac{};
	amrex::GpuArray<amrex::Real, kMaxTrackedIsotopes> agb_frac{};
};

struct CompactYieldEntry {
	amrex::Real mass_msun = 0.0;
	amrex::Real metallicity = 0.0;
	int channel_index = -1;
	int isotope_index = -1;
	amrex::Real yield = 0.0;
};

inline auto lowercase(std::string s) -> std::string
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
}

inline auto normalizeNumericToken(std::string token) -> std::string
{
	auto replace_all = [&token](std::string const &from, std::string const &to) {
		std::string::size_type pos = 0;
		while ((pos = token.find(from, pos)) != std::string::npos) {
			token.replace(pos, from.size(), to);
			pos += to.size();
		}
	};

	// Normalize common Unicode glyphs that appear in externally generated yield tables.
	replace_all("\xE2\x88\x92", "-"); // U+2212 minus sign
	replace_all("\xE2\x80\x93", "-"); // U+2013 en dash
	replace_all("\xE2\x80\x94", "-"); // U+2014 em dash
	replace_all("\xC2\xA0", "");      // U+00A0 non-breaking space

	return token;
}

inline auto elementFromIsotope(std::string isotope) -> std::string
{
	isotope = lowercase(isotope);
	std::string elem;
	for (const char c : isotope) {
		if (std::isalpha(static_cast<unsigned char>(c)) == 0) {
			break;
		}
		elem.push_back(c);
	}
	return elem;
}

inline auto isMetalElement(const std::string &elem_in) -> bool
{
	const std::string elem = lowercase(elem_in);
	if (elem == "h" || elem == "he" || elem == "p" || elem == "d" || elem == "n" || elem == "g") {
		return false;
	}
	return !elem.empty();
}

inline auto parseMetallicityFromFolderName(const std::string &name) -> amrex::Real
{
	// e.g. z001models -> 0.001, z004models -> 0.004, z02models -> 0.02
	if (name.size() < 7 || name.front() != 'z') {
		return -1.0;
	}
	const auto pos = name.find("models");
	if (pos == std::string::npos || pos <= 1) {
		return -1.0;
	}
	const std::string digits = name.substr(1, pos - 1);
	if (digits.empty()) {
		return -1.0;
	}
	const int numer = std::stoi(digits);
	const amrex::Real denom = std::pow(10.0, static_cast<int>(digits.size()));
	return static_cast<amrex::Real>(numer) / denom;
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

inline auto parseLegacyFiveColumnFile(const std::filesystem::path &file_path, std::vector<YieldEntry> &entries) -> bool
{
	std::ifstream file(file_path);
	if (!file.is_open()) {
		return false;
	}

	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::stringstream ss(line);
		YieldEntry e{};
	amrex::Real snii_frac = 0.0;
	amrex::Real wr_frac = 0.0;
	amrex::Real agb_frac = 0.0;
		if (!(ss >> e.mass_msun >> e.metallicity >> snii_frac >> wr_frac >> agb_frac)) {
			continue;
		}
		e.mass_msun = std::max<amrex::Real>(e.mass_msun, 1.0e-12);
		e.metallicity = std::max<amrex::Real>(e.metallicity, 1.0e-12);
		e.snii_frac[0] = std::max<amrex::Real>(snii_frac, 0.0);
		e.wr_frac[0] = std::max<amrex::Real>(wr_frac, 0.0);
		e.agb_frac[0] = std::max<amrex::Real>(agb_frac, 0.0);
		entries.push_back(e);
	}

	return !entries.empty();
}

inline auto parseSNIIFromRawDirectory(const std::filesystem::path &root, std::vector<YieldEntry> &entries) -> void
{
	const std::filesystem::path snii_root = root / "SNII_Kobayashi0611";
	if (!std::filesystem::exists(snii_root) || !std::filesystem::is_directory(snii_root)) {
		return;
	}

	const std::regex mass_re(R"(s([0-9]+(?:\.[0-9]+)?)\.yield_table)");

	for (const auto &dir_entry : std::filesystem::directory_iterator(snii_root)) {
		if (!dir_entry.is_directory()) {
			continue;
		}
		const std::string zdir_name = dir_entry.path().filename().string();
		const amrex::Real z = parseMetallicityFromFolderName(zdir_name);
		if (z <= 0.0) {
			continue;
		}

		for (const auto &file_entry : std::filesystem::directory_iterator(dir_entry.path())) {
			if (!file_entry.is_regular_file()) {
				continue;
			}
			const std::string fname = file_entry.path().filename().string();
			std::smatch m;
			if (!std::regex_match(fname, m, mass_re)) {
				continue;
			}
			const amrex::Real mass = static_cast<amrex::Real>(std::stod(m[1].str()));

			std::ifstream in(file_entry.path());
			if (!in.is_open()) {
				continue;
			}

			std::string line;
			amrex::Real metal_sum = 0.0;
			while (std::getline(in, line)) {
				if (line.empty() || line[0] == '[' || line[0] == '#') {
					continue;
				}
				std::stringstream ss(line);
				std::string isotope;
				std::string yld_token;
				if (!(ss >> isotope >> yld_token)) {
					continue;
				}
				amrex::Real yld = 0.0;
				try {
					yld = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yld_token)));
				} catch (...) {
					continue;
				}
				if (isMetalElement(elementFromIsotope(isotope))) {
					metal_sum += std::max<amrex::Real>(yld, 0.0);
				}
			}

			YieldEntry e{};
			e.mass_msun = std::max<amrex::Real>(mass, 1.0e-12);
			e.metallicity = std::max<amrex::Real>(z, 1.0e-12);
			e.snii_frac[0] = std::max<amrex::Real>(metal_sum / e.mass_msun, 0.0);
			entries.push_back(e);
		}
	}
}

inline auto parseDohertySuperAGB(const std::filesystem::path &file_path, std::vector<YieldEntry> &entries) -> void
{
	if (!std::filesystem::exists(file_path)) {
		return;
	}
	std::ifstream in(file_path);
	if (!in.is_open()) {
		return;
	}

	const std::regex header_re(R"(([0-9]+(?:\.[0-9]+)?)M\s+Z=\s*([0-9]+(?:\.[0-9]+)?))", std::regex::icase);

	std::string line;
	amrex::Real current_mass = -1.0;
	amrex::Real current_z = -1.0;
	amrex::Real metal_sum = 0.0;
	bool in_block = false;

	auto flush_block = [&]() {
		if (!in_block || current_mass <= 0.0 || current_z <= 0.0) {
			return;
		}
		YieldEntry e{};
		e.mass_msun = std::max<amrex::Real>(current_mass, 1.0e-12);
		e.metallicity = std::max<amrex::Real>(current_z, 1.0e-12);
		e.agb_frac[0] = std::max<amrex::Real>(metal_sum / e.mass_msun, 0.0);
		entries.push_back(e);
	};

	while (std::getline(in, line)) {
		std::smatch m;
		if (std::regex_search(line, m, header_re)) {
			flush_block();
			current_mass = static_cast<amrex::Real>(std::stod(m[1].str()));
			current_z = static_cast<amrex::Real>(std::stod(m[2].str()));
			metal_sum = 0.0;
			in_block = true;
			continue;
		}

		if (!in_block || line.empty() || line.find("Species") != std::string::npos) {
			continue;
		}

		std::stringstream ss(line);
		std::string isotope;
		amrex::Real yld = 0.0;
		if (!(ss >> isotope >> yld)) {
			continue;
		}
		if (isMetalElement(elementFromIsotope(isotope))) {
			metal_sum += std::max<amrex::Real>(yld, 0.0);
		}
	}

	flush_block();
}

inline auto parseKarakasAGBFile(const std::filesystem::path &file_path, std::vector<YieldEntry> &entries) -> void
{
	std::ifstream in(file_path);
	if (!in.is_open()) {
		return;
	}

	const std::regex header_re(
	    R"(#\s*Initial\s+mass\s*=\s*([0-9]+(?:\.[0-9]+)?),\s*Z\s*=\s*([0-9]+(?:\.[0-9]+)?),.*M_mix\s*=\s*([0-9eE+\-.]+))",
	    std::regex::icase);

	std::string line;
	amrex::Real current_mass = -1.0;
	amrex::Real current_z = -1.0;
	amrex::Real current_mmix = 0.0;
	amrex::Real metal_sum = 0.0;
	bool in_block = false;

	auto flush_block = [&]() {
		if (!in_block || current_mass <= 0.0 || current_z <= 0.0) {
			return;
		}
		if (std::abs(current_mmix) > 1.0e-12) {
			return;
		}
		YieldEntry e{};
		e.mass_msun = std::max<amrex::Real>(current_mass, 1.0e-12);
		e.metallicity = std::max<amrex::Real>(current_z, 1.0e-12);
		e.agb_frac[0] = std::max<amrex::Real>(metal_sum / e.mass_msun, 0.0);
		entries.push_back(e);
	};

	while (std::getline(in, line)) {
		std::smatch m;
		if (std::regex_search(line, m, header_re)) {
			flush_block();
			current_mass = static_cast<amrex::Real>(std::stod(m[1].str()));
			current_z = static_cast<amrex::Real>(std::stod(m[2].str()));
			current_mmix = static_cast<amrex::Real>(std::stod(m[3].str()));
			metal_sum = 0.0;
			in_block = true;
			continue;
		}

		if (!in_block || line.empty() || line[0] == '#') {
			continue;
		}

		std::stringstream ss(line);
		std::string species;
		int A = 0;
		std::string yld_token;
		if (!(ss >> species >> A >> yld_token)) {
			continue;
		}
		amrex::Real yld = 0.0;
		try {
			yld = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yld_token)));
		} catch (...) {
			continue;
		}
		if (isMetalElement(species)) {
			metal_sum += std::max<amrex::Real>(yld, 0.0);
		}
	}

	flush_block();
}

inline auto parseAGBFromRawDirectory(const std::filesystem::path &root, std::vector<YieldEntry> &entries) -> void
{
	const std::filesystem::path agb_root = root / "AGB_Karakas16";
	if (std::filesystem::exists(agb_root) && std::filesystem::is_directory(agb_root)) {
		for (const auto &dir_entry : std::filesystem::recursive_directory_iterator(agb_root)) {
			if (!dir_entry.is_regular_file()) {
				continue;
			}
			if (dir_entry.path().extension().string() == ".dat") {
				parseKarakasAGBFile(dir_entry.path(), entries);
			}
		}
	}

	parseDohertySuperAGB(root / "superAGB_Doherty14" / "doherty14a_table1.txt", entries);
}

inline auto finalizeAndUpload(std::vector<YieldEntry> const &raw_entries) -> bool
{
	g_num_entries = 0;
	g_loaded = false;

	if (raw_entries.empty()) {
		return false;
	}

	std::map<std::pair<int, int>, std::array<amrex::Real, 4>> acc;
	for (auto const &e : raw_entries) {
		const int mk = static_cast<int>(std::llround(e.mass_msun * 1000.0));
		const int zk = static_cast<int>(std::llround(e.metallicity * 1.0e6));
		auto &v = acc[{mk, zk}];
		v[0] += e.snii_frac[0];
		v[1] += e.wr_frac[0];
		v[2] += e.agb_frac[0];
		v[3] += 1.0;
	}

	for (auto const &[k, v] : acc) {
		if (g_num_entries >= kMaxChemicalYieldEntries) {
			break;
		}
		const int idx = g_num_entries;
		const amrex::Real cnt = std::max<amrex::Real>(v[3], 1.0);
		g_mass_msun[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(k.first) / 1000.0, 1.0e-12);
		g_metallicity[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(k.second) / 1.0e6, 1.0e-12);
		g_snii_frac[idx] = std::max<amrex::Real>(v[0] / cnt, 0.0);
		g_wr_frac[idx] = std::max<amrex::Real>(v[1] / cnt, 0.0);
		g_agb_frac[idx] = std::max<amrex::Real>(v[2] / cnt, 0.0);
		++g_num_entries;
	}

	g_loaded = (g_num_entries > 0);
	return g_loaded;
}

inline auto loadTable(const std::string &filename) -> bool
{
	std::vector<YieldEntry> entries;
	const std::filesystem::path input_path = resolveInputPath(filename);

	if (std::filesystem::exists(input_path) && std::filesystem::is_regular_file(input_path)) {
		if (!parseLegacyFiveColumnFile(input_path, entries)) {
			return false;
		}
		return finalizeAndUpload(entries);
	}

	if (std::filesystem::exists(input_path) && std::filesystem::is_directory(input_path)) {
		parseSNIIFromRawDirectory(input_path, entries);
		parseAGBFromRawDirectory(input_path, entries);
		return finalizeAndUpload(entries);
	}

	return false;
}

inline auto loadTable(const std::string &filename, const std::vector<std::string> &tracked_isotopes, const std::vector<std::string> & /*tracked_channels*/) -> bool
{
	g_num_tracked_isotopes = std::min(static_cast<int>(tracked_isotopes.size()), kMaxTrackedIsotopes);
	g_num_tracked_channels = 3;
	const std::filesystem::path input_path = resolveInputPath(filename);
	if (std::filesystem::exists(input_path) && std::filesystem::is_regular_file(input_path)) {
		std::ifstream file(input_path);
		if (!file.is_open()) {
			return false;
		}

		std::map<std::string, int> iso_map;
		for (int i = 0; i < g_num_tracked_isotopes; ++i) {
			iso_map[lowercase(tracked_isotopes[i])] = i;
		}
		std::map<std::string, int> channel_map{{"snii", 0}, {"wr", 1}, {"agb", 2}};

		std::vector<CompactYieldEntry> rows;
		std::string line;
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}
			std::stringstream ss(line);
			CompactYieldEntry row{};
			std::string channel_name;
			std::string isotope_name;
			std::string mass_token;
			std::string metallicity_token;
			std::string yield_token;
			if (!(ss >> mass_token >> metallicity_token >> channel_name >> isotope_name >> yield_token)) {
				continue;
			}
			try {
				row.mass_msun = static_cast<amrex::Real>(std::stod(normalizeNumericToken(mass_token)));
				row.metallicity = static_cast<amrex::Real>(std::stod(normalizeNumericToken(metallicity_token)));
				row.yield = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yield_token)));
			} catch (...) {
				continue;
			}
			row.channel_index = trackedChannelIndex(channel_map, channel_name);
			row.isotope_index = trackedIsotopeIndex(iso_map, isotope_name);
			if (row.channel_index < 0 || row.isotope_index < 0) {
				continue;
			}
			rows.push_back(row);
		}

		g_num_entries = 0;
		g_loaded = false;
		std::map<std::pair<int, int>, std::vector<amrex::Real>> sums;
		for (const auto &row : rows) {
			const int mk = static_cast<int>(std::llround(row.mass_msun * 1000.0));
			const int zk = static_cast<int>(std::llround(row.metallicity * 1.0e6));
			auto &vals = sums[{mk, zk}];
			if (vals.empty()) {
				const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
							static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
				vals.resize(table_size, 0.0);
			}
			vals[row.channel_index * kMaxTrackedIsotopes + row.isotope_index] += row.yield;
		}

		for (const auto &[key, vals] : sums) {
			if (g_num_entries >= kMaxChemicalYieldEntries) {
				break;
			}
			const int idx = g_num_entries;
			g_mass_msun[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(key.first) / 1000.0, 1.0e-12);
			g_metallicity[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(key.second) / 1.0e6, 1.0e-12);
			for (int c = 0; c < kMaxTrackedChannels; ++c) {
				for (int i = 0; i < g_num_tracked_isotopes; ++i) {
					const int flat = idx * kMaxTrackedChannels * kMaxTrackedIsotopes + c * kMaxTrackedIsotopes + i;
					g_channel_iso_frac[flat] = vals[c * kMaxTrackedIsotopes + i];
				}
			}
			++g_num_entries;
		}
		g_loaded = (g_num_entries > 0);
		return g_loaded;
	}

	if (std::filesystem::exists(input_path) && std::filesystem::is_directory(input_path)) {
		const std::filesystem::path snii_root = input_path / "SNII_Kobayashi0611";
		if (!std::filesystem::exists(snii_root) || !std::filesystem::is_directory(snii_root)) {
			return false;
		}

		std::map<std::string, int> iso_map;
		for (int i = 0; i < g_num_tracked_isotopes; ++i) {
			iso_map[lowercase(tracked_isotopes[i])] = i;
		}

		std::map<std::pair<int, int>, std::vector<amrex::Real>> sums;
		const std::regex mass_re(R"(s([0-9]+(?:\.[0-9]+)?)\.yield_table)");

		for (const auto &dir_entry : std::filesystem::directory_iterator(snii_root)) {
			if (!dir_entry.is_directory()) {
				continue;
			}
			const std::string zdir_name = dir_entry.path().filename().string();
			const amrex::Real z = parseMetallicityFromFolderName(zdir_name);
			if (z <= 0.0) {
				continue;
			}

			for (const auto &file_entry : std::filesystem::directory_iterator(dir_entry.path())) {
				if (!file_entry.is_regular_file()) {
					continue;
				}

				const std::string fname = file_entry.path().filename().string();
				std::smatch m;
				if (!std::regex_match(fname, m, mass_re)) {
					continue;
				}

				const amrex::Real mass_msun = static_cast<amrex::Real>(std::stod(m[1].str()));
				if (mass_msun <= 0.0) {
					continue;
				}

				std::ifstream in(file_entry.path());
				if (!in.is_open()) {
					continue;
				}

				const int mk = static_cast<int>(std::llround(mass_msun * 1000.0));
				const int zk = static_cast<int>(std::llround(z * 1.0e6));
				auto &vals = sums[{mk, zk}];
				if (vals.empty()) {
					const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
							static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
					vals.resize(table_size, 0.0);
				}

				std::string line;
				while (std::getline(in, line)) {
					if (line.empty() || line[0] == '[' || line[0] == '#') {
						continue;
					}

					std::stringstream ss(line);
					std::string isotope_name;
					std::string yield_token;
					if (!(ss >> isotope_name >> yield_token)) {
						continue;
					}
					amrex::Real yield_mass = 0.0;
					try {
						yield_mass = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yield_token)));
					} catch (...) {
						continue;
					}

					const auto iso_it = iso_map.find(lowercase(isotope_name));
					if (iso_it == iso_map.end()) {
						continue;
					}

					const int iso_index = iso_it->second;
					const amrex::Real frac = std::max<amrex::Real>(yield_mass / mass_msun, 0.0);
					vals[0 * kMaxTrackedIsotopes + iso_index] += frac;
				}
			}
		}

		// Parse WR (wind) yields from SNII_Sukhbold16 -> channel 1
		const std::filesystem::path wr_root = input_path / "SNII_Sukhbold16";
		if (std::filesystem::exists(wr_root) && std::filesystem::is_directory(wr_root)) {
			const std::regex wr_mass_re(R"(s([0-9]+(?:\.[0-9]+)?)\.yield_table)");
			// Sukhbold+16 tables are solar metallicity
			constexpr amrex::Real wr_z = 0.014;
			const int wr_zk = static_cast<int>(std::llround(wr_z * 1.0e6));

			for (const auto &file_entry : std::filesystem::directory_iterator(wr_root)) {
				if (!file_entry.is_regular_file()) {
					continue;
				}
				const std::string fname = file_entry.path().filename().string();
				std::smatch m;
				if (!std::regex_match(fname, m, wr_mass_re)) {
					continue;
				}
				const amrex::Real mass_msun = static_cast<amrex::Real>(std::stod(m[1].str()));
				if (mass_msun <= 0.0) {
					continue;
				}

				std::ifstream in(file_entry.path());
				if (!in.is_open()) {
					continue;
				}

				const int mk = static_cast<int>(std::llround(mass_msun * 1000.0));
				auto &vals = sums[{mk, wr_zk}];
				if (vals.empty()) {
					const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
								static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
					vals.resize(table_size, 0.0);
				}

				std::string line;
				while (std::getline(in, line)) {
					if (line.empty() || line[0] == '[' || line[0] == '#') {
						continue;
					}
					std::stringstream ss(line);
					std::string isotope_name;
					if (!(ss >> isotope_name)) {
						continue;
					}
					// wind yield is the last numeric token on the line
					amrex::Real wind_yield = 0.0;
					bool has_value = false;
					std::string token;
					while (ss >> token) {
						try {
							wind_yield = static_cast<amrex::Real>(std::stod(normalizeNumericToken(token)));
							has_value = true;
						} catch (...) {
						}
					}
					if (!has_value) {
						continue;
					}

					const auto iso_it = iso_map.find(lowercase(isotope_name));
					if (iso_it == iso_map.end()) {
						continue;
					}
					const int iso_index = iso_it->second;
					const amrex::Real frac = std::max<amrex::Real>(wind_yield / mass_msun, 0.0);
					vals[1 * kMaxTrackedIsotopes + iso_index] += frac;
				}
			}
		}

		// Parse AGB yields from AGB_Karakas16 -> channel 2
		const std::filesystem::path agb_root = input_path / "AGB_Karakas16";
		if (std::filesystem::exists(agb_root) && std::filesystem::is_directory(agb_root)) {
			// Filename-based mass/Z: m<mass>z<Z_digits>.<extra>.dat
			const std::regex agb_fname_re(R"(m([0-9]+(?:\.[0-9]+)?)z([0-9]+).*\.dat)", std::regex::icase);
			// Header-based mass/Z (for yield_z0001.dat style files)
			const std::regex agb_header_re(
			    R"(#\s*Initial\s+mass\s*=\s*([0-9]+(?:\.[0-9]+)?),\s*Z\s*=\s*([0-9]+(?:\.[0-9]+)?),.*M_mix\s*=\s*([0-9eE+\-.]+))",
			    std::regex::icase);

			for (const auto &dir_entry : std::filesystem::recursive_directory_iterator(agb_root)) {
				if (!dir_entry.is_regular_file()) {
					continue;
				}
				if (dir_entry.path().extension().string() != ".dat") {
					continue;
				}

				const std::string fname = dir_entry.path().filename().string();
				std::smatch fm;
				amrex::Real agb_mass = -1.0;
				amrex::Real agb_z = -1.0;

				if (std::regex_match(fname, fm, agb_fname_re)) {
					// Parse mass and Z from filename
					agb_mass = static_cast<amrex::Real>(std::stod(fm[1].str()));
					const std::string z_digits = fm[2].str();
					const int numer = std::stoi(z_digits);
					agb_z = static_cast<amrex::Real>(numer) / std::pow(10.0, static_cast<int>(z_digits.size()));
				} else {
					// Try header-based parsing for yield_z0001.dat style
					std::ifstream header_in(dir_entry.path());
					if (!header_in.is_open()) {
						continue;
					}
					std::string hdr_line;
					bool found = false;
					while (std::getline(header_in, hdr_line)) {
						std::smatch hm;
						if (std::regex_search(hdr_line, hm, agb_header_re)) {
							agb_mass = static_cast<amrex::Real>(std::stod(hm[1].str()));
							agb_z = static_cast<amrex::Real>(std::stod(hm[2].str()));
							found = true;
							break;
						}
					}
					if (!found) {
						continue;
					}
				}

				if (agb_mass <= 0.0 || agb_z <= 0.0) {
					continue;
				}

				std::ifstream in(dir_entry.path());
				if (!in.is_open()) {
					continue;
				}

				const int mk = static_cast<int>(std::llround(agb_mass * 1000.0));
				const int zk = static_cast<int>(std::llround(agb_z * 1.0e6));
				auto &vals = sums[{mk, zk}];
				if (vals.empty()) {
					const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
								static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
					vals.resize(table_size, 0.0);
				}

				std::string line;
				while (std::getline(in, line)) {
					if (line.empty() || line[0] == '#') {
						continue;
					}

					std::stringstream ss(line);
					std::string species;
					int A = 0;
					std::string yld_token;
					if (!(ss >> species >> A >> yld_token)) {
						continue;
					}
					// species column already holds the isotope name, e.g. "c12", "o16", "ar40"
					const auto iso_it = iso_map.find(lowercase(species));
					if (iso_it == iso_map.end()) {
						continue;
					}
					amrex::Real yld = 0.0;
					try {
						yld = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yld_token)));
					} catch (...) {
						continue;
					}
					const amrex::Real frac = std::max<amrex::Real>(yld / agb_mass, 0.0);
					vals[2 * kMaxTrackedIsotopes + iso_it->second] += frac;
				}
			}
		}

		g_num_entries = 0;
		g_loaded = false;
		for (const auto &[key, vals] : sums) {
			if (g_num_entries >= kMaxChemicalYieldEntries) {
				break;
			}
			const int idx = g_num_entries;
			g_mass_msun[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(key.first) / 1000.0, 1.0e-12);
			g_metallicity[idx] = std::max<amrex::Real>(static_cast<amrex::Real>(key.second) / 1.0e6, 1.0e-12);
			for (int c = 0; c < kMaxTrackedChannels; ++c) {
				for (int i = 0; i < g_num_tracked_isotopes; ++i) {
					const int flat = idx * kMaxTrackedChannels * kMaxTrackedIsotopes + c * kMaxTrackedIsotopes + i;
					g_channel_iso_frac[flat] = vals[c * kMaxTrackedIsotopes + i];
				}
			}
			++g_num_entries;
		}

		g_loaded = (g_num_entries > 0);
		return g_loaded;
	}

	return loadTable(filename);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto isLoaded() -> bool { return g_loaded && (g_num_entries > 0); }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto queryYieldFraction(int channel_index, int isotope_index, amrex::Real mass_msun, amrex::Real metallicity)
    -> amrex::Real
{
	if (!isLoaded() || channel_index < 0 || isotope_index < 0 || channel_index >= g_num_tracked_channels || isotope_index >= g_num_tracked_isotopes) {
		return 0.0;
	}

	const amrex::Real safe_mass = amrex::max<amrex::Real>(mass_msun, 1.0e-12);
	const amrex::Real safe_z = amrex::max<amrex::Real>(metallicity, 1.0e-12);
	const amrex::Real log_mass = std::log10(safe_mass);
	const amrex::Real log_z = std::log10(safe_z);

	amrex::Real wsum = 0.0;
	amrex::Real frac_sum = 0.0;
	constexpr amrex::Real eps = 1.0e-20;
	constexpr amrex::Real exact_tol = 1.0e-10;

	for (int i = 0; i < g_num_entries; ++i) {
		const int flat = i * kMaxTrackedChannels * kMaxTrackedIsotopes + channel_index * kMaxTrackedIsotopes + isotope_index;
		const amrex::Real frac_i = g_channel_iso_frac[flat];
		// Skip entries that have no data for this channel+isotope (e.g. SNII-only entries when querying WR)
		if (frac_i <= 0.0) {
			continue;
		}
		const amrex::Real dm = std::abs(std::log10(amrex::max<amrex::Real>(g_mass_msun[i], 1.0e-12)) - log_mass);
		const amrex::Real dz = std::abs(std::log10(amrex::max<amrex::Real>(g_metallicity[i], 1.0e-12)) - log_z);
		const amrex::Real dist2 = dm * dm + dz * dz;
		if (dist2 < exact_tol) {
			return frac_i;
		}
		const amrex::Real w = 1.0 / (dist2 + eps);
		wsum += w;
		frac_sum += w * frac_i;
	}

	return (wsum > 0.0) ? (frac_sum / wsum) : 0.0;
}

// Returns [SNII total fraction, WR total fraction, AGB total fraction] from inverse-distance interpolation in (logM, logZ).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto queryFractions(amrex::Real mass_msun, amrex::Real metallicity)
    -> amrex::GpuArray<amrex::Real, 3>
{
	if (!isLoaded()) {
		return {0.0, 0.0, 0.0};
	}

	const amrex::Real safe_mass = amrex::max<amrex::Real>(mass_msun, 1.0e-12);
	const amrex::Real safe_z = amrex::max<amrex::Real>(metallicity, 1.0e-12);
	const amrex::Real log_mass = std::log10(safe_mass);
	const amrex::Real log_z = std::log10(safe_z);

	amrex::Real wsum = 0.0;
	amrex::Real snii_sum = 0.0;
	amrex::Real wr_sum = 0.0;
	amrex::Real agb_sum = 0.0;
	constexpr amrex::Real eps = 1.0e-20;
	constexpr amrex::Real exact_tol = 1.0e-10;

	for (int i = 0; i < g_num_entries; ++i) {
		const amrex::Real dm = std::abs(std::log10(amrex::max<amrex::Real>(g_mass_msun[i], 1.0e-12)) - log_mass);
		const amrex::Real dz = std::abs(std::log10(amrex::max<amrex::Real>(g_metallicity[i], 1.0e-12)) - log_z);
		const amrex::Real dist2 = dm * dm + dz * dz;
		if (dist2 < exact_tol) {
			return {g_snii_frac[i], g_wr_frac[i], g_agb_frac[i]};
		}
		const amrex::Real w = 1.0 / (dist2 + eps);
		wsum += w;
		snii_sum += w * g_snii_frac[i];
		wr_sum += w * g_wr_frac[i];
		agb_sum += w * g_agb_frac[i];
	}

	if (wsum <= 0.0) {
		return {0.0, 0.0, 0.0};
	}

	return {snii_sum / wsum, wr_sum / wsum, agb_sum / wsum};
}

} // namespace ChemicalYieldLookup
// #Chuhan_end: Centralise the preparation of chemical yield data and onfly query capabilities.

} // namespace quokka

#endif // PARTICLE_CHEMICAL_YIELD_HPP_
