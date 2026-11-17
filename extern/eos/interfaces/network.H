#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <network_properties.H>

void network_init();

AMREX_INLINE int network_spec_index(const std::string& name) {
  int idx = -1;
  for (int n = 0; n < NumSpec; n++) {
    if (name == spec_names_cxx[n]) {
      idx = n;
      break;
    }
  }
  return idx;
}

// As network_spec_index(), but matching the network's raw (unsanitized) species
// names -- jaff's own spelling, "H+" and "e-" rather than "H_p" and "e".
//
// short_spec_names_cxx and spec_names_cxx are built from the same species list
// in the same order, so an index found in one addresses the same species in the
// other. Looking up by raw name lets a configuration file name a species the way
// the network file does, without having to know how
// write_jaff_network_vars.py:sanitize_species_name() spells it as a C++
// identifier.
AMREX_INLINE int network_spec_index_raw(const std::string& name) {
  int idx = -1;
  for (int n = 0; n < NumSpec; n++) {
    if (name == short_spec_names_cxx[n]) {
      idx = n;
      break;
    }
  }
  return idx;
}

#if NAUX_NET > 0
AMREX_INLINE int network_aux_index(const std::string& name) {
  int idx = -1;
  for (int n = 0; n < NumAux; n++) {
    if (name == aux_names_cxx[n]) {
      idx = n;
      break;
    }
  }
  return idx;
}
#endif

#endif
