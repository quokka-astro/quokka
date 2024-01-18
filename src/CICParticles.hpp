#ifndef CICPARTICLES_HPP_ // NOLINT
#define CICPARTICLES_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CICParticles.hpp
/// \brief Implements the particle container for gravitationally-interacting particles
///

#include "AMReX_AmrParticles.H"

namespace quokka
{

constexpr int CICParticleReals = 4; // mass vx vy vz
constexpr int CICParticleInts = 0;

enum ParticleDataIdx {ParticleMass = 0, ParticleVx, ParticleVy, ParticleVz};

using CICParticleContainer = amrex::AmrParticleContainer<0, 0, CICParticleReals, CICParticleInts>;
} // namespace quokka

#endif // CICPARTICLES_HPP_