#ifndef BDS_HPP_
#define BDS_HPP_

#include "AMReX_MultiFab.H"

void ComputeBDSReconstructionOptimized(amrex::MultiFab const &input_mf, amrex::MultiFab &x_L, amrex::MultiFab &x_R, amrex::MultiFab &y_L,
				     amrex::MultiFab &y_R, amrex::MultiFab &z_L, amrex::MultiFab &z_R, int num_ghost);

#endif // BDS_HPP_
