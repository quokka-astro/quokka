#ifndef MF_INTERPOLATER_H_
#define MF_INTERPOLATER_H_
#include <AMReX_Config.H>

#include <AMReX_Extension.H>
#include <AMReX_InterpBase.H>

namespace amrex {

class MultiFab;
class Geometry;

/**
* \brief Linear conservative interpolation on cell centered data
*
* Linear conservative interpolation on cell centered data, i.e, conservative
* interpolation with a limiting scheme that preserves the value of any
* linear combination of the fab components. If sum_ivar
* a(ic,jc,ivar)*fab(ic,jc,ivar) = 0, then sum_ivar
* a(ic,jc,ivar)*fab(if,jf,ivar) = 0 is satisfied in all fine cells if,jf
* covering coarse cell ic,jc.
*/
class MFQuokkaInterp
    : public MFInterpolater
{
public:
    explicit MFQuokkaInterp () {}

    virtual ~MFQuokkaInterp () = default;

    virtual Box CoarseBox (Box const& fine, int ratio) override;
    virtual Box CoarseBox (Box const& fine, IntVect const& ratio) override;

    virtual void interp (MultiFab const& crsemf, int ccomp, MultiFab& finemf, int fcomp, int ncomp,
                         IntVect const& ng, Geometry const& cgeom, Geometry const& fgeom,
                         Box const& dest_domain, IntVect const& ratio,
                         Vector<BCRec> const& bcs, int bcscomp) override;
};

extern AMREX_EXPORT MFQuokkaInterp mf_linear_slope_minmax_interp;

}

#endif
