# Particle Deposition Utilities

This document describes the per-species particle deposition utilities added to Quokka for creating derived fields from particle data.

## Overview

The particle deposition system allows depositing particle properties (mass, momentum, energy, number density) onto the computational grid for visualization and analysis. This is particularly useful for:

- Creating 2D slice plots with particle data using yt
- Generating derived fields for high-cadence output without full particle plotfiles
- Analyzing particle distributions in post-processing

## Components

### 1. DiagParticleDeposition

A diagnostic class that deposits particle properties into grid fields and outputs them.

**Configuration:**
```ini
# Enable particle deposition diagnostic
diag.particle_deposition.type = DiagParticleDeposition
diag.particle_deposition.int = 10
diag.particle_deposition.particle_types = CIC StochasticStellarPop Sink
diag.particle_deposition.deposit_fields = mass momentum energy number
diag.particle_deposition.output_format = plotfile
```

**Parameters:**
- `particle_types`: Space-separated list of particle types to deposit
- `deposit_fields`: Space-separated list of fields to deposit
- `output_format`: "plotfile" or "ascii"

### 2. Particle Deposition Utilities

Located in `src/particles/particle_deposition_utils.hpp`, these utilities provide:

#### Deposition Functors:
- `ParticleMassDensityDeposition`: Deposits particle mass density
- `ParticleMomentumDensityDeposition`: Deposits particle momentum density
- `ParticleKineticEnergyDensityDeposition`: Deposits particle kinetic energy density
- `ParticleNumberDensityDeposition`: Deposits particle number density

#### Type-Specific Functions:
- `depositCICParticleProperties()`
- `depositStochasticStellarPopParticleProperties()`
- `depositSinkParticleProperties()`
- `depositTestParticleProperties()`

#### Generic Interface:
- `depositParticlePropertiesByType()`: Runtime particle type selection

## Usage

### In Derived Fields

Add to your problem's `ComputeDerivedVar` method:

```cpp
void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override
{
    if (dname == "CIC_mass_density") {
        mf.setVal(0.0);
        auto *cicContainer = particleRegister_.getCICParticleContainer();
        depositParticleMassDensity(cicContainer, mf, lev, CICParticleMassIdx, 0);
    }
    // ... other derived fields
}
```

### Direct Usage

```cpp
// Create MultiFab for deposition
amrex::MultiFab massField(grids[lev], dmap[lev], 1, 0);
massField.setVal(0.0);

// Deposit CIC particle mass density
auto *cicContainer = particleRegister_.getCICParticleContainer();
depositParticleMassDensity(cicContainer, massField, lev, CICParticleMassIdx, 0);

// Use the deposited field
amrex::Real totalMass = massField.sum(0);
```

## Particle Types Supported

- **CIC**: Cloud-In-Cell gravitating particles
- **StochasticStellarPop**: Stellar population particles
- **Sink**: Sink particles for accretion
- **Test**: Test particles with all features
- **Rad**: Radiation particles (number density only)
- **CICRad**: Combined gravitating-radiating particles

## Fields Available

- **mass**: Particle mass density
- **momentum**: Particle momentum density (3 components)
- **energy**: Particle kinetic energy density
- **number**: Particle number density

## Implementation Details

- Uses AMReX's `ParticleInterpolator::Linear` for Cloud-In-Cell deposition
- Supports all AMReX particle container types
- GPU-accelerated using AMReX's GPU abstractions
- Integrates with existing diagnostic system using Factory pattern
- Supports AMR with proper level handling

## Testing

The implementation includes a comprehensive test in `src/problems/ParticleDeposition/` that verifies:
- Mass conservation during deposition
- Number conservation during deposition
- Momentum and energy deposition accuracy
- Integration with diagnostic system

Run the test with:
```bash
cd build
ctest -R ParticleDeposition
```

## Future Enhancements

- Support for higher-order interpolation schemes
- Integration with slice diagnostic system
- Support for custom particle properties
- Performance optimizations for large particle datasets