#ifndef EOS_OVERRIDE_H
#define EOS_OVERRIDE_H

#include <eos_type.H>

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void eos_override (T& /*state*/) {}

#endif

