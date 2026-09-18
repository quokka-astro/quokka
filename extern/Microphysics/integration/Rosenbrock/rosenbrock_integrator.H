#ifndef ROSENBROCK_INTEGRATOR_H
#define ROSENBROCK_INTEGRATOR_H

#include <AMReX_Algorithm.H>

#include <cmath>
#include <limits>

#include <network.H>
#include <burn_type.H>
#include <linpack.H>

#include <integrator_rhs_strang.H>

#include <integrator_data.H>
#include <rosenbrock_type.H>
#include <rosenbrock_tableau.H>

namespace rosenbrock {

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real& stage (rosenbrock_t<int_neqs>& rstate, const int i, const int n)
{
    if (i == 1) {
        return rstate.ak1(n);
    }
    if (i == 2) {
        return rstate.ak2(n);
    }
    if (i == 3) {
        return rstate.ak3(n);
    }
    if (i == 4) {
        return rstate.ak4(n);
    }
    if (i == 5) {
        return rstate.ak5(n);
    }
    if (i == 6) {
        return rstate.ak6(n);
    }
    if (i == 7) {
        return rstate.ak7(n);
    }
    if (i == 8) {
        return rstate.ak8(n);
    }
    return rstate.work(n);
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stage (const rosenbrock_t<int_neqs>& rstate, const int i, const int n)
{
    if (i == 1) {
        return rstate.ak1(n);
    }
    if (i == 2) {
        return rstate.ak2(n);
    }
    if (i == 3) {
        return rstate.ak3(n);
    }
    if (i == 4) {
        return rstate.ak4(n);
    }
    if (i == 5) {
        return rstate.ak5(n);
    }
    if (i == 6) {
        return rstate.ak6(n);
    }
    if (i == 7) {
        return rstate.ak7(n);
    }
    if (i == 8) {
        return rstate.ak8(n);
    }
    return rstate.work(n);
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RArray1D& stage_vector (rosenbrock_t<int_neqs>& rstate, const int i)
{
    if (i == 1) {
        return rstate.ak1;
    }
    if (i == 2) {
        return rstate.ak2;
    }
    if (i == 3) {
        return rstate.ak3;
    }
    if (i == 4) {
        return rstate.ak4;
    }
    if (i == 5) {
        return rstate.ak5;
    }
    if (i == 6) {
        return rstate.ak6;
    }
    if (i == 7) {
        return rstate.ak7;
    }
    if (i == 8) {
        return rstate.ak8;
    }
    return rstate.work;
}

template <typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rtol_for (const RosenbrockT& rstate, const int n)
{
    if (n == net_ienuc) {
        return rstate.rtol_enuc;
    }
#ifdef PHOTOCHEMISTRY
    if (n > net_ienuc) {
        const int group_offset = (n - net_ienuc - 1) % MicrophysicsNumRadVarsPerGroup;
        return group_offset == 0 ? rstate.rtol_rad_num : 1.0e30;
    }
#endif
    return rstate.rtol_spec;
}

template <typename RosenbrockT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real atol_for (const RosenbrockT& rstate, const int n)
{
    if (n == net_ienuc) {
        return rstate.atol_enuc;
    }
#ifdef PHOTOCHEMISTRY
    if (n > net_ienuc) {
        const int group_offset = (n - net_ienuc - 1) % MicrophysicsNumRadVarsPerGroup;
        return group_offset == 0 ? rstate.atol_rad_num : 1.0e30;
    }
#endif
    return rstate.atol_spec;
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real error_norm (const rosenbrock_t<int_neqs>& rstate)
{
    amrex::Real err = 0.0_rt;
    for (int n = 1; n <= int_neqs; ++n) {
        const amrex::Real sk = atol_for(rstate, n) +
            rtol_for(rstate, n) * amrex::max(std::abs(rstate.y(n)), std::abs(rstate.ynew(n)));
#ifdef PHOTOCHEMISTRY
    const int group_offset = (n - net_ienuc - 1) % MicrophysicsNumRadVarsPerGroup;
    amrex::Real term;
    if (n > net_ienuc && group_offset != 0) {
        term = 0.0_rt;
    } else {
        term = rstate.work(n) / sk;
    }
#else
    const amrex::Real term = rstate.work(n) / sk;
#endif
    err += term * term;
    }
    return std::sqrt(err / static_cast<amrex::Real>(int_neqs));
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool species_change_valid (const rosenbrock_t<int_neqs>& rstate)
{
    constexpr amrex::Real increase_change_factor = 4.0_rt;
    constexpr amrex::Real decrease_change_factor = 0.25_rt;

    for (int i = 1; i <= NumSpec; ++i) {
        if (std::abs(rstate.y(i)) > integrator_rp::X_reject_buffer * rstate.atol_spec &&
            std::abs(rstate.ynew(i)) > integrator_rp::X_reject_buffer * rstate.atol_spec &&
            (std::abs(rstate.ynew(i)) > increase_change_factor * std::abs(rstate.y(i)) ||
             std::abs(rstate.ynew(i)) < decrease_change_factor * std::abs(rstate.y(i)))) {
            return false;
        }
    }

    return true;
}

template <typename MatrixType, int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool matrix_is_finite (const MatrixType& matrix)
{
    for (int j = 1; j <= int_neqs; ++j) {
        for (int i = 1; i <= int_neqs; ++i) {
            const amrex::Real value = matrix(i, j);
            if (std::isnan(value) || std::isinf(value)) {
                return false;
            }
        }
    }

    return true;
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool valid_integrator_state (const amrex::Array1D<amrex::Real, 1, int_neqs>& y)
{
    for (int n = 1; n <= int_neqs; ++n) {
        const amrex::Real yn = y(n);
        if (std::isnan(yn) || std::isinf(yn)) {
            return false;
        }
    }

    for (int n = 1; n <= NumSpec; ++n) {
        if (y(n) < -integrator_rp::atol_spec) {
            return false;
        }

        if (! integrator_rp::use_number_densities && y(n) > 1.0_rt) {
            return false;
        }
    }

#ifdef PHOTOCHEMISTRY
    for (int n = net_ienuc+1; n <= net_ienuc+NumRadEqs; n += MicrophysicsNumRadVarsPerGroup) {
        if (y(n) < -integrator_rp::atol_rad_num) {
            return false;
        }
    }
#endif

    if (y(net_ienuc) <= 0.0_rt) {
        return false;
    }

    return true;
}

template <typename BurnT, int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void evaluate_jacobian (BurnT& state, rosenbrock_t<int_neqs>& rstate, const amrex::Real time)
{
    constexpr bool in_jacobian = true;

    rhs(time, state, rstate, rstate.ak1, in_jacobian);
    rstate.n_rhs += 1;

    if (rstate.jacobian_type == 1) {
        jac(time, state, rstate, rstate.jac);
        rstate.n_jac += 1;

        if (matrix_is_finite<decltype(rstate.jac), int_neqs>(rstate.jac)) {
            return;
        }
    }

    constexpr amrex::Real UROUND = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Array1D<amrex::Real, 1, int_neqs> ewt;
    amrex::Array1D<amrex::Real, 1, int_neqs> ybase;

    // rhs() may clean or renormalize all components of rstate.y.
    for (int i = 1; i <= int_neqs; ++i) {
        ybase(i) = rstate.y(i);
    }

    amrex::Real fac = 0.0_rt;
    for (int i = 1; i <= int_neqs; ++i) {
        ewt(i) = 1.0_rt / (rtol_for(rstate, i) * std::abs(rstate.y(i)) + atol_for(rstate, i));
        fac += (rstate.ak1(i) * ewt(i)) * (rstate.ak1(i) * ewt(i));
    }
    fac = std::sqrt(fac / static_cast<amrex::Real>(int_neqs));

    amrex::Real R0 = 1000.0_rt * std::abs(rstate.dt) *
        UROUND * static_cast<amrex::Real>(int_neqs) * fac;
    if (R0 == 0.0_rt) {
        R0 = 1.0_rt;
    }

    for (int j = 1; j <= int_neqs; ++j) {
        const amrex::Real yj = ybase(j);
        const amrex::Real R = amrex::max(std::sqrt(UROUND) * std::abs(yj), R0 / ewt(j));
        for (int n = 1; n <= int_neqs; ++n) {
            rstate.y(n) = ybase(n);
        }
        rstate.y(j) += R;

        rhs(time, state, rstate, rstate.work, in_jacobian);

        fac = 1.0_rt / R;
        for (int i = 1; i <= int_neqs; ++i) {
            rstate.jac.set(i, j, (rstate.work(i) - rstate.ak1(i)) * fac);
        }

        for (int n = 1; n <= int_neqs; ++n) {
            rstate.y(n) = ybase(n);
        }
    }

    rstate.n_rhs += int_neqs;
    rstate.n_jac += 1;
}

template <typename BurnT, int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rhs_at (const amrex::Real time, BurnT& state, rosenbrock_t<int_neqs>& rstate,
             const amrex::Array1D<amrex::Real, 1, int_neqs>& y,
             amrex::Array1D<amrex::Real, 1, int_neqs>& ydot)
{
    amrex::Array1D<amrex::Real, 1, int_neqs> ysave;

    for (int n = 1; n <= int_neqs; ++n) {
        ysave(n) = rstate.y(n);
        rstate.y(n) = y(n);
    }

    rhs(time, state, rstate, ydot);

    for (int n = 1; n <= int_neqs; ++n) {
        rstate.y(n) = ysave(n);
    }
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int decompose (rosenbrock_t<int_neqs>& rstate, const amrex::Real fac)
{
    for (int j = 1; j <= int_neqs; ++j) {
        for (int i = 1; i <= int_neqs; ++i) {
            rstate.jac(i, j) = -rstate.jac(i, j);
        }
        rstate.jac(j, j) += fac;
    }

    int ierr_linpack = 0;
    if (integrator_rp::linalg_do_pivoting == 1) {
        constexpr bool allow_pivot{true};
        dgefa<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, ierr_linpack);
    } else {
        constexpr bool allow_pivot{false};
        dgefa<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, ierr_linpack);
    }

    return ierr_linpack;
}

template <int int_neqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void solve (rosenbrock_t<int_neqs>& rstate, RArray1D& b)
{
    if (integrator_rp::linalg_do_pivoting == 1) {
        constexpr bool allow_pivot{true};
        dgesl<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, b);
    } else {
        constexpr bool allow_pivot{false};
        dgesl<int_neqs, allow_pivot>(rstate.jac, rstate.pivot, b);
    }
}

} // namespace rosenbrock

template <typename BurnT, typename Tableau>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int rosenbrock_integrator (BurnT& state, rosenbrock_t<integrator_neqs<BurnT>()>& rstate)
{
    using C = Tableau;
    constexpr int int_neqs = integrator_neqs<BurnT>();
    static_assert(C::stages <= 8, "Rosenbrock integrator stores up to 8 stages");

    if (rstate.tout == rstate.t) {
        return IERR_SUCCESS;
    }

    for (int n = 1; n <= int_neqs; ++n) {
        if (rosenbrock::atol_for(rstate, n) <= 0.0_rt ||
            rosenbrock::rtol_for(rstate, n) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
            return IERR_TOO_MUCH_ACCURACY_REQUESTED;
        }
    }

    rstate.n_rhs = 0;
    rstate.n_jac = 0;
    rstate.n_step = 0;

    rstate.dt = rstate.tout - rstate.t;

    const amrex::Real posneg = rstate.tout >= rstate.t ? 1.0_rt : -1.0_rt;
    const amrex::Real hmax = amrex::min(integrator_rp::ode_max_dt,
                                        std::abs(rstate.tout - rstate.t));
    amrex::Real h = amrex::min(std::abs(rstate.dt), hmax) * posneg;
    if (std::abs(h) <= 10.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
        h = 1.e-6_rt * posneg;
    }

    bool reject = false;
    bool last = false;
    int nsing = 0;
    int n_reject = 0;
    amrex::Real facold = 1.0_rt;
    amrex::Real errold = 1.0_rt;
    amrex::Real hopt = h;
    amrex::Real x = rstate.t;
    amrex::Array1D<amrex::Real, 1, int_neqs> rhs_tmp;

    // main evolution loop

    // within this loop, x will be the current time and h will be the
    // step size we are attempting.  The solution at the current time
    // is rstate.y().

    while (true) {

        if (rstate.n_step > integrator_rp::ode_max_steps) {
            rstate.t = x;
            rstate.dt = h;
            return IERR_TOO_MANY_STEPS;
        }

        if (0.1_rt * std::abs(h) <= std::abs(x) * std::numeric_limits<amrex::Real>::epsilon()) {
            rstate.t = x;
            rstate.dt = h;
            return IERR_DT_UNDERFLOW;
        }

        if (last) {
            rstate.t = x;
            rstate.dt = hopt;
            return IERR_SUCCESS;
        }

        hopt = h;
        if ((x + h * (1.0_rt + timestep_safety_factor) - rstate.tout) * posneg >= 0.0_rt) {
            h = rstate.tout - x;
            last = true;
        }


        // take a step -- this loop will keep trying until a step is
        // successful or a catastrophic error is encountered.

        while (true) {
            rstate.dt = h;

            if (0.1_rt * std::abs(h) <= std::abs(x) * std::numeric_limits<amrex::Real>::epsilon()) {
                rstate.t = x;
                rstate.dt = h;
                return IERR_DT_UNDERFLOW;
            }

            const amrex::Real fac = 1.0_rt / (h * C::gamma);
            rosenbrock::evaluate_jacobian(state, rstate, x);
            int ierr_linpack = rosenbrock::decompose(rstate, fac);

            if (ierr_linpack != 0) {
                nsing += 1;
                if (nsing >= 5) {
                    rstate.t = x;
                    rstate.dt = h;
                    return IERR_LU_DECOMPOSITION_ERROR;
                }
                h *= 0.5_rt;
                reject = true;
                last = false;
                continue;
            }

            rosenbrock::solve(rstate, rstate.ak1);

            bool valid_stage_state = true;
            for (int istage = 2; istage <= C::stages; ++istage) {
                for (int n = 1; n <= int_neqs; ++n) {
                    rstate.ynew(n) = rstate.y(n);
                    rosenbrock::stage(rstate, istage, n) = 0.0_rt;

                    for (int jstage = 1; jstage < istage; ++jstage) {
                        const amrex::Real akj = rosenbrock::stage(rstate, jstage, n);
                        rstate.ynew(n) += C::a(istage, jstage) * akj;
                        rosenbrock::stage(rstate, istage, n) +=
                            (C::c(istage, jstage) / h) * akj;
                    }
                }

                if (! rosenbrock::valid_integrator_state(rstate.ynew)) {
                    h *= 0.25_rt;
                    reject = true;
                    n_reject += 1;
                    last = false;
                    valid_stage_state = false;
                    break;
                }

                rosenbrock::rhs_at(x + C::ct(istage) * h, state, rstate, rstate.ynew, rhs_tmp);
                rstate.n_rhs += 1;

                for (int n = 1; n <= int_neqs; ++n) {
                    rosenbrock::stage(rstate, istage, n) += rhs_tmp(n);
                }
                rosenbrock::solve(rstate, rosenbrock::stage_vector(rstate, istage));
            }

            if (! valid_stage_state) {
                continue;
            }

            for (int n = 1; n <= int_neqs; ++n) {
                amrex::Real y_update = 0.0_rt;
                amrex::Real err = 0.0_rt;

                for (int istage = 1; istage <= C::stages; ++istage) {
                    const amrex::Real aki = rosenbrock::stage(rstate, istage, n);
                    y_update += C::b(istage) * aki;
                    err += C::e(istage) * aki;
                }

                rstate.ynew(n) = rstate.y(n) + y_update;
                rstate.work(n) = err;
            }

            rstate.n_step += 1;

            const bool valid_state = rosenbrock::valid_integrator_state(rstate.ynew);
            if (! valid_state) {
                reject = true;
                n_reject += 1;
                last = false;
                h *= 0.25_rt;
                continue;
            }

            const bool valid_update = rosenbrock::species_change_valid(rstate);
            constexpr amrex::Real err_min = 1.e-10_rt;
            amrex::Real err = rosenbrock::error_norm(rstate);
            amrex::Real controller_fac = amrex::Clamp(
                std::pow(1.0_rt / amrex::max(err, err_min),
                         1.0_rt / (integrator_rp::h211b_b * integrator_rp::h211b_k)) *
                std::pow(1.0_rt / amrex::max(errold, err_min),
                         1.0_rt / (integrator_rp::h211b_b * integrator_rp::h211b_k)) *
                std::pow(facold, -1.0_rt / integrator_rp::h211b_b),
                integrator_rp::h211b_fac_min, integrator_rp::h211b_fac_max);
            facold = controller_fac;
            errold = err;

            amrex::Real hnew = h * controller_fac;

            // update the starting time and state for the next step
            // and predict the next timestep

            if (err <= 1.0_rt && valid_update) {
                for (int n = 1; n <= int_neqs; ++n) {
                    rstate.y(n) = rstate.ynew(n);
                }
                x += h;

                if (std::abs(hnew) > hmax) {
                    hnew = posneg * hmax;
                }
                if (reject) {
                    hnew = posneg * amrex::min(std::abs(hnew), std::abs(h));
                }
                reject = false;
                n_reject = 0;
                h = hnew;
                break;
            }

            // if we made it here, then we've failed
            // cut the timestep in prep for trying again

            reject = true;
            n_reject += 1;
            last = false;
            if (n_reject >= 2) {
                hnew *= integrator_rp::h211b_reduction_fac;
            }
            if (valid_update) {
                h = posneg * amrex::min(std::abs(hnew),
                                         integrator_rp::h211b_reduction_fac * std::abs(h));
            } else {
                h *= 0.25_rt;
            }
        }
    }
}

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int rosenbrock_integrator (BurnT& state, rosenbrock_t<integrator_neqs<BurnT>()>& rstate)
{
    if (integrator_rp::rosenbrock_tableau == 1) {
        return rosenbrock_integrator<BurnT, rosenbrock::rodas4p_tableau>(state, rstate);
    }
    if (integrator_rp::rosenbrock_tableau == 2) {
        return rosenbrock_integrator<BurnT, rosenbrock::rodas3p_tableau>(state, rstate);
    }
    if (integrator_rp::rosenbrock_tableau == 3) {
        return rosenbrock_integrator<BurnT, rosenbrock::ros2s_tableau>(state, rstate);
    }

    return rosenbrock_integrator<BurnT, rosenbrock::rodas5p_tableau>(state, rstate);
}

#endif
