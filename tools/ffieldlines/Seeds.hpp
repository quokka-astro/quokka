#ifndef FFIELDLINES_SEEDS_HPP_
#define FFIELDLINES_SEEDS_HPP_
/// \file Seeds.hpp
/// \brief Seed point input, generation, and validation.

#include <array>
#include <string>
#include <vector>

#include "Options.hpp"

namespace ffieldlines
{

using Point3 = std::array<double, 3>;

/// Hard cap on the number of seeds. More lines than this cannot be usefully visualized.
constexpr int maxSeeds = 10000;

/// Read seeds from a text file: one "x y z" per line; blank lines and '#' comments are ignored.
auto ReadSeedFile(std::string const &path) -> std::vector<Point3>;

/// `n` evenly spaced points from `p0` to `p1` inclusive (`p0` alone when n == 1).
auto SeedLine(Point3 const &p0, Point3 const &p1, int n) -> std::vector<Point3>;

/// `n` points on a disk with the given center, normal and radius, using
/// sunflower (Fibonacci) spacing so the points cover the disk evenly.
auto SeedDisk(Point3 const &center, Point3 const &normal, double radius, int n) -> std::vector<Point3>;

/// Build the seed list described by `opts` (seed file, line, or disk).
auto MakeSeeds(Options const &opts) -> std::vector<Point3>;

/// Abort unless 1 <= count <= maxSeeds and every seed is finite and inside
/// [probLo, probHi). The message lists the first offending seeds.
void ValidateSeeds(std::vector<Point3> const &seeds, Point3 const &probLo, Point3 const &probHi);

} // namespace ffieldlines

#endif // FFIELDLINES_SEEDS_HPP_
