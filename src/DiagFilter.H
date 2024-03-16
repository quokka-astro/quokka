#ifndef DIAGFILTERS_H
#define DIAGFILTERS_H

#include <AMReX.H>
#include <AMReX_Utility.H>

struct DiagFilterData {
	int m_filterVarIdx{-1};
	amrex::Real m_low_val{0.0};
	amrex::Real m_high_val{0.0};
};

struct DiagFilter {
	std::string m_filterVar;

	void init(const std::string &a_prefix);
	void setup(const amrex::Vector<std::string> &a_varNames);

	DiagFilterData m_fdata;
};
#endif
