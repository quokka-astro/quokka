#include "Castro_gravity.H"
#include "Gravity.H"

using namespace amrex;

void construct_gravity_source(MultiFab &source, MultiFab &state_in, Real time,
                              Real dt) {
  BL_PROFILE("Castro::construct_old_gravity_source()");

  // Gravitational source term for the time-level n+1 data.
  const MultiFab &grav_new = get_new_data(Gravity_Type);

  for (MFIter mfi(state_in); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();

    Array4<Real const> const uold = state_in.array(mfi);
    Array4<Real const> const grav = grav_new.array(mfi);
    Array4<Real> const source_arr = source.array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
      // Temporary array for seeing what the new state would be if the update
      // were applied here.

      GpuArray<Real, NUM_STATE> snew;
      for (int n = 0; n < NUM_STATE; ++n) {
        snew[n] = 0.0;
      }

      // Temporary array for holding the update to the state.

      GpuArray<Real, NSRC> src;
      for (int n = 0; n < NSRC; ++n) {
        src[n] = 0.0;
      }

      Real rho = uold(i, j, k, URHO);
      Real rhoInv = 1.0 / rho;

      for (int n = 0; n < NUM_STATE; ++n) {
        snew[n] = uold(i, j, k, n);
      }

      Real old_ke = 0.5 *
                    (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] +
                     snew[UMZ] * snew[UMZ]) *
                    rhoInv;

      GpuArray<Real, 3> Sr;
      for (int n = 0; n < 3; ++n) {
        Sr[n] = rho * grav(i, j, k, n);

        src[UMX + n] = Sr[n];

        snew[UMX + n] += dt * src[UMX + n];
      }

      Real SrE;

      Real new_ke = 0.5 *
                    (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] +
                     snew[UMZ] * snew[UMZ]) *
                    rhoInv;
      SrE = new_ke - old_ke;

      src[UEDEN] = SrE;

      snew[UEDEN] += dt * SrE;

      // Add to the outgoing source array.

      for (int n = 0; n < NSRC; ++n) {
        source_arr(i, j, k, n) += src[n];
      }
    });
  }
}
