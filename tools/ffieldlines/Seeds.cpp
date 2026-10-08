/// \file Seeds.cpp
/// \brief Seed point input, generation, and validation.

#include "Seeds.hpp"

#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>

#include <AMReX.H>

namespace ffieldlines
{

auto ReadSeedFile(std::string const &path) -> std::vector<Point3>
{
	std::ifstream in(path);
	if (!in.good()) {
		amrex::Abort("ffieldlines: cannot open seed file '" + path + "'");
	}
	std::vector<Point3> seeds;
	std::string line;
	int lineNumber = 0;
	while (std::getline(in, line)) {
		++lineNumber;
		const auto hash = line.find('#');
		if (hash != std::string::npos) {
			line.erase(hash);
		}
		std::istringstream fields(line);
		Point3 p{};
		if (!(fields >> p[0])) {
			continue; // blank or comment-only line
		}
		std::string extra;
		if (!(fields >> p[1] >> p[2]) || (fields >> extra)) {
			amrex::Abort("ffieldlines: seed file '" + path + "' line " + std::to_string(lineNumber) + ": expected 'x y z'");
		}
		seeds.push_back(p);
	}
	return seeds;
}

auto SeedLine(Point3 const &p0, Point3 const &p1, const int n) -> std::vector<Point3>
{
	std::vector<Point3> seeds;
	for (int i = 0; i < n; ++i) {
		const double t = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
		seeds.push_back({p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2])});
	}
	return seeds;
}

auto SeedDisk(Point3 const &center, Point3 const &normal, const double radius, const int n) -> std::vector<Point3>
{
	const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
	if (!(norm > 0.0) || !(radius >= 0.0)) {
		amrex::Abort("ffieldlines: --seed-disk needs a non-zero normal and a non-negative radius");
	}
	const Point3 nhat{normal[0] / norm, normal[1] / norm, normal[2] / norm};

	// orthonormal basis (u, v) of the disk plane: cross nhat with the axis it is least aligned with
	int least = 0;
	for (int d = 1; d < 3; ++d) {
		if (std::abs(nhat[d]) < std::abs(nhat[least])) {
			least = d;
		}
	}
	Point3 axis{0.0, 0.0, 0.0};
	axis[least] = 1.0;
	Point3 u{nhat[1] * axis[2] - nhat[2] * axis[1], nhat[2] * axis[0] - nhat[0] * axis[2], nhat[0] * axis[1] - nhat[1] * axis[0]};
	const double unorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
	for (auto &c : u) {
		c /= unorm;
	}
	const Point3 v{nhat[1] * u[2] - nhat[2] * u[1], nhat[2] * u[0] - nhat[0] * u[2], nhat[0] * u[1] - nhat[1] * u[0]};

	const double goldenAngle = std::numbers::pi * (3.0 - std::sqrt(5.0));
	std::vector<Point3> seeds;
	for (int i = 0; i < n; ++i) {
		const double r = radius * std::sqrt((static_cast<double>(i) + 0.5) / static_cast<double>(n));
		const double theta = goldenAngle * static_cast<double>(i);
		const double a = r * std::cos(theta);
		const double b = r * std::sin(theta);
		seeds.push_back({center[0] + a * u[0] + b * v[0], center[1] + a * u[1] + b * v[1], center[2] + a * u[2] + b * v[2]});
	}
	return seeds;
}

auto MakeSeeds(Options const &opts) -> std::vector<Point3>
{
	auto count = [](double value, std::string const &option) -> int {
		if (value != std::floor(value) || value < 1.0 || value > static_cast<double>(maxSeeds)) {
			amrex::Abort("ffieldlines: " + option + " point count must be an integer in [1, " + std::to_string(maxSeeds) + "]");
		}
		return static_cast<int>(value);
	};

	if (!opts.seedFile.empty()) {
		return ReadSeedFile(opts.seedFile);
	}
	if (!opts.seedLine.empty()) {
		auto const &a = opts.seedLine;
		return SeedLine({a[0], a[1], a[2]}, {a[3], a[4], a[5]}, count(a[6], "--seed-line"));
	}
	auto const &a = opts.seedDisk;
	return SeedDisk({a[0], a[1], a[2]}, {a[3], a[4], a[5]}, a[6], count(a[7], "--seed-disk"));
}

void ValidateSeeds(std::vector<Point3> const &seeds, Point3 const &probLo, Point3 const &probHi)
{
	if (seeds.empty()) {
		amrex::Abort("ffieldlines: no seed points given");
	}
	if (std::ssize(seeds) > maxSeeds) {
		amrex::Abort("ffieldlines: " + std::to_string(seeds.size()) + " seeds given; the limit is " + std::to_string(maxSeeds));
	}

	std::ostringstream bad;
	int nBad = 0;
	for (size_t i = 0; i < seeds.size(); ++i) {
		bool ok = true;
		for (int d = 0; d < 3; ++d) {
			ok = ok && std::isfinite(seeds[i][d]) && seeds[i][d] >= probLo[d] && seeds[i][d] < probHi[d];
		}
		if (!ok) {
			if (nBad < 10) {
				bad << "\n  seed " << i << ": (" << seeds[i][0] << ", " << seeds[i][1] << ", " << seeds[i][2] << ")";
			}
			++nBad;
		}
	}
	if (nBad > 0) {
		std::ostringstream msg;
		msg << "ffieldlines: " << nBad << " seed(s) are non-finite or outside the domain [(" << probLo[0] << ", " << probLo[1] << ", " << probLo[2]
		    << "), (" << probHi[0] << ", " << probHi[1] << ", " << probHi[2] << ")):" << bad.str();
		amrex::Abort(msg.str());
	}
}

} // namespace ffieldlines
