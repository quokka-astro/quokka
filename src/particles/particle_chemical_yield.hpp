#ifndef PARTICLE_CHEMICAL_YIELD_HPP_
#define PARTICLE_CHEMICAL_YIELD_HPP_

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include <algorithm>
#include <cctype>
#include <cmath>
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

namespace ChemicalYieldLookup
{

constexpr int kMaxTrackedIsotopes = 32;
constexpr int kMaxTrackedChannels = 3;
constexpr int kMaxChemicalYieldEntries = 4096;

AMREX_GPU_MANAGED inline int g_num_entries = 0;		 // NOLINT
AMREX_GPU_MANAGED inline bool g_loaded = false;		 // NOLINT
AMREX_GPU_MANAGED inline int g_num_tracked_isotopes = 0; // NOLINT
AMREX_GPU_MANAGED inline int g_num_tracked_channels = 0; // NOLINT

AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries> g_mass_msun{};							  // NOLINT
AMREX_GPU_MANAGED inline amrex::GpuArray<amrex::Real, kMaxChemicalYieldEntries> g_metallicity{};						  // NOLINT
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
	replace_all("\xC2\xA0", "");	  // U+00A0 non-breaking space

	return token;
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

inline auto loadTable(const std::string &filename, const std::vector<std::string> &tracked_isotopes, const std::vector<std::string> & /*tracked_channels*/)
    -> bool
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
		std::map<std::string, int> iso_map;
		for (int i = 0; i < g_num_tracked_isotopes; ++i) {
			iso_map[lowercase(tracked_isotopes[i])] = i;
		}

		std::map<std::pair<int, int>, std::vector<amrex::Real>> sums;

		const std::filesystem::path sukhbold_root = input_path / "SNII_Sukhbold16";
		if (std::filesystem::exists(sukhbold_root) && std::filesystem::is_directory(sukhbold_root)) {
			const std::regex sukhbold_mass_re(R"(s([0-9]+(?:\.[0-9]+)?)\.yield_table)");
			constexpr amrex::Real sukhbold_z = 0.014;
			const int sukhbold_zk = static_cast<int>(std::llround(sukhbold_z * 1.0e6));

			for (const auto &file_entry : std::filesystem::directory_iterator(sukhbold_root)) {
				if (!file_entry.is_regular_file()) {
					continue;
				}
				const std::string fname = file_entry.path().filename().string();
				std::smatch m;
				if (!std::regex_match(fname, m, sukhbold_mass_re)) {
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
				auto &vals = sums[{mk, sukhbold_zk}];
				if (vals.empty()) {
					const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
								static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
					vals.resize(table_size, 0.0);
				}

				std::string line;
				bool has_ejecta_col = false;
				bool has_wind_col = false;
				while (std::getline(in, line)) {
					if (line.empty() || line[0] == '#') {
						continue;
					}
					if (line[0] == '[') {
						has_ejecta_col = (line.find("[ejecta]") != std::string::npos);
						has_wind_col = (line.find("[wind]") != std::string::npos);
						continue;
					}
					std::stringstream ss(line);
					std::string isotope_name;
					std::string ejecta_token;
					std::string wind_token;
					if (!(ss >> isotope_name)) {
						continue;
					}
					if (has_ejecta_col && !(ss >> ejecta_token)) {
						continue;
					}
					if (has_wind_col && !(ss >> wind_token)) {
						continue;
					}
					const auto iso_it = iso_map.find(lowercase(isotope_name));
					if (iso_it == iso_map.end()) {
						continue;
					}
					try {
						const int iso_index = iso_it->second;
						if (has_ejecta_col) {
							const amrex::Real ejecta_yield = static_cast<amrex::Real>(std::stod(normalizeNumericToken(ejecta_token)));
							vals[0 * kMaxTrackedIsotopes + iso_index] += std::max<amrex::Real>(ejecta_yield / mass_msun, 0.0);
						}
						if (has_wind_col) {
							const amrex::Real wind_yield = static_cast<amrex::Real>(std::stod(normalizeNumericToken(wind_token)));
							vals[1 * kMaxTrackedIsotopes + iso_index] += std::max<amrex::Real>(wind_yield / mass_msun, 0.0);
						}
					} catch (...) {
						continue;
					}
				}
			}
		}

		const std::filesystem::path agb_root = input_path / "AGB_Karakas16";
		if (std::filesystem::exists(agb_root) && std::filesystem::is_directory(agb_root)) {
			const std::regex agb_fname_re(R"(m([0-9]+(?:\.[0-9]+)?)z([0-9]+).*\.dat)", std::regex::icase);
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
					agb_mass = static_cast<amrex::Real>(std::stod(fm[1].str()));
					const std::string z_digits = fm[2].str();
					const int numer = std::stoi(z_digits);
					agb_z = static_cast<amrex::Real>(numer) / std::pow(10.0, static_cast<int>(z_digits.size()));
				} else {
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
					const auto iso_it = iso_map.find(lowercase(species));
					if (iso_it == iso_map.end()) {
						continue;
					}
					try {
						const amrex::Real yld = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yld_token)));
						vals[2 * kMaxTrackedIsotopes + iso_it->second] += std::max<amrex::Real>(yld / agb_mass, 0.0);
					} catch (...) {
						continue;
					}
				}
			}
		}

		const std::filesystem::path doherty_root = input_path / "superAGB_Doherty14";
		if (std::filesystem::exists(doherty_root) && std::filesystem::is_directory(doherty_root)) {
			const std::regex doherty_header_re(R"(\s*([0-9]+(?:\.[0-9]+)?)M\s+Z=([0-9eE+\-.]+).*)", std::regex::icase);

			for (const auto &file_entry : std::filesystem::directory_iterator(doherty_root)) {
				if (!file_entry.is_regular_file()) {
					continue;
				}
				std::ifstream in(file_entry.path());
				if (!in.is_open()) {
					continue;
				}

				amrex::Real doherty_mass = -1.0;
				std::vector<amrex::Real> *vals_ptr = nullptr;
				std::string line;
				while (std::getline(in, line)) {
					std::smatch hm;
					if (std::regex_match(line, hm, doherty_header_re)) {
						doherty_mass = static_cast<amrex::Real>(std::stod(hm[1].str()));
						const amrex::Real doherty_z = static_cast<amrex::Real>(std::stod(hm[2].str()));
						const int mk = static_cast<int>(std::llround(doherty_mass * 1000.0));
						const int zk = static_cast<int>(std::llround(doherty_z * 1.0e6));
						auto &vals = sums[{mk, zk}];
						if (vals.empty()) {
							const auto table_size = static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedChannels) *
										static_cast<std::vector<amrex::Real>::size_type>(kMaxTrackedIsotopes);
							vals.resize(table_size, 0.0);
						}
						vals_ptr = &vals;
						continue;
					}
					if (vals_ptr == nullptr || doherty_mass <= 0.0 || line.empty() || line[0] == '#') {
						continue;
					}

					std::stringstream ss(line);
					std::string species;
					std::string yld_token;
					if (!(ss >> species >> yld_token)) {
						continue;
					}
					const auto iso_it = iso_map.find(lowercase(species));
					if (iso_it == iso_map.end()) {
						continue;
					}
					try {
						const amrex::Real yld = static_cast<amrex::Real>(std::stod(normalizeNumericToken(yld_token)));
						(*vals_ptr)[2 * kMaxTrackedIsotopes + iso_it->second] += std::max<amrex::Real>(yld / doherty_mass, 0.0);
					} catch (...) {
						continue;
					}
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

	return false;
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

} // namespace ChemicalYieldLookup

} // namespace quokka

#endif // PARTICLE_CHEMICAL_YIELD_HPP_
