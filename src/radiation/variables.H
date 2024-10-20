#include <vector>
#include <string>

#ifndef VARIABLES_H
#define VARIABLES_H

#include <AMReX_Vector.H>

#include <network.H>

class plot_t {

public:

  int irho = -1;
  int itemp = -1;
  int ispec = -1;
  int ispec_old = -1;
#if NAUX_NET > 0
  int iaux = -1;
  int iaux_old = -1;
#endif
  int irodot = -1;
  int irho_hnuc = -1;

  int n_plot_comps = 0;

  int next_index(const int num) {
    int next = n_plot_comps;
    n_plot_comps += num;
    return next;
  }

};

plot_t init_variables();

void get_varnames(const plot_t& p, amrex::Vector<std::string>& names);

#endif
