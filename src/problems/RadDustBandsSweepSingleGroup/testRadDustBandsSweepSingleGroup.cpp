/// \file testRadDustBandsSweepSingleGroup.cpp
/// \brief The RadDustBandsSweep parameter sweep for a single dust-absorption group.
///
/// A single group takes its opacities from the single-group hooks, a separate path from several groups, so
/// it is swept on its own. The sweep itself is shared with RadDustBandsSweep.

#define DUST_SWEEP_ONE_GROUP
#include "../RadDustBandsSweep/testRadDustBandsSweep.cpp"
