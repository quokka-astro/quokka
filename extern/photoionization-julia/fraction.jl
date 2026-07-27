using Pkg
PATH = dirname(PROGRAM_FILE)
Pkg.activate(PATH)
Pkg.instantiate()

using DifferentialEquations
using Sundials
using Plots
using CSV
using DataFrames

const m_e = 9.1093837139e-28
const m_HI = 1.67353286432139e-24
const m_HII = 1.67262192595e-24
const c   = 2.99792458e10
const k_B = 1.3806490000000002e-16
const σ_v = 2.1596876715103067e-18
const E_ion = 6.4e-12

struct State
    n_spec::Vector{Float64}
    n_rad::Vector{Float64}
    γ_vec::Vector{Float64}
    spec_mass::Vector{Float64}
    E::Float64
end

function get_ρ(state::State)
    ρ = 0.0
    for i in 1:length(state.n_spec)
        ρ += state.n_spec[i] * state.spec_mass[i]
    end
    return ρ
end

function get_γ_inv(state::State)
    γ = 0.0
    for i in 1:length(state.n_spec)
        γ += state.n_spec[i] / (state.γ_vec[i] - 1.0)
    end
    return γ / sum(state.n_spec)
end

function get_T(state::State)
    γ_inv = get_γ_inv(state)
    ρ = get_ρ(state)
    sum_ni = sum(state.n_spec)
    T = state.E * ρ / (k_B * γ_inv * sum_ni)
    return T
end

function get_E(state::State, T::Float64)
    # The state should contain a dummy value for energy
    γ_inv = get_γ_inv(state)
    sum_ni = sum(state.n_spec)
    E = γ_inv * k_B * T * sum_ni / get_ρ(state)
    return E
end

function rhs!(df::Vector{Float64}, f::Vector{Float64}, params, t)
    energy_switch = params[1]
    γ_e, γ_HI, γ_HII = params[2:4]
    m_e, m_HI, m_HII = params[5:7]

    # Unpack the state variables
    state = State(f[1:3], f[4:5], [γ_e, γ_HI, γ_HII], [m_e, m_HI, m_HII], f[6])
    n_spec = state.n_spec
    n_rad = state.n_rad
    ρ = get_ρ(state)
    T = get_T(state)

    n_e = n_spec[1]
    n_HI = n_spec[2]
    n_HII = n_spec[3]
    n_photon = n_rad[1]
    flux_photon = n_rad[2]

    α_rec = 2.63e-13 * (1 + energy_switch * ((T / 1.0e4)^(-0.69999999999999996) - 1.0))
    ionization_term = n_HI * c * σ_v * n_photon
    recombination_term = α_rec * n_e * n_HII

    df[1] =  ionization_term - recombination_term
    df[2] = -ionization_term + recombination_term
    df[3] =  ionization_term - recombination_term
    df[4] = -ionization_term
    df[5] = -ionization_term * flux_photon / n_photon
    df[6] = energy_switch * (ionization_term * E_ion - recombination_term * k_B * T * (0.684 - 0.0416 * log(T / 1.0e4))) / ρ
end

function make_problem(state::State, tend::Float64, params::Tuple)
    f = vcat(state.n_spec, state.n_rad, state.E)
    prob = ODEProblem(rhs!, f, (0.0, tend), params)
    return prob
end

function parse_fraction(s)
    if occursin("/", s)
        num, den = split(s, "/")
        return parse(Float64, num) / parse(Float64, den)
    else
        return parse(Float64, s)
    end
end

n_photon0 = parse(Float64, ARGS[1])
n_e, n_HI, n_HII = parse.(Float64, ARGS[2:4])
γ_e, γ_HI, γ_HII = parse_fraction.(ARGS[5:7])
T0 = parse(Float64, ARGS[8])
energy_switch = parse(Int64, ARGS[9])
tend = parse(Float64, ARGS[10])
constant_dt = parse(Float64, ARGS[11])
abstol, reltol = parse.(Float64, ARGS[12:13])
save_path = ARGS[14]

E0 = get_E(State([n_e, n_HI, n_HII], [n_photon0, 0], [γ_e, γ_HI, γ_HII], [m_e, m_HI, m_HII], 0.0), T0)
state = State(
    [n_e, n_HI, n_HII],     # n_spec: [n_e, n_HI, n_HII]
    [n_photon0, 0],         # n_rad: [n_photon, flux_photon]
    [γ_e, γ_HI, γ_HII],     # γ_vec
    [m_e, m_HI, m_HII],     # spec_mass
    E0                      # E
)
params = (energy_switch, γ_e, γ_HI, γ_HII, m_e, m_HI, m_HII)
prob = make_problem(state, tend, params)
sol = solve(prob, saveat=constant_dt, CVODE_BDF(); reltol=reltol, abstol=abstol)

T_values = [get_T(State(sol[i][1:3], sol[i][4:5], [γ_e, γ_HI, γ_HII], [m_e, m_HI, m_HII], sol[i][6])) for i in eachindex(sol)]
rho_values = [get_ρ(State(sol[i][1:3], sol[i][4:5], [γ_e, γ_HI, γ_HII], [m_e, m_HI, m_HII], sol[i][6])) for i in eachindex(sol)]

results = DataFrame(time=sol.t, n_e=sol[1, :], n_HI=sol[2, :], n_HII=sol[3, :], n_gamma=sol[4, :], Egas=sol[6, :] .* rho_values, gas_temp=T_values)
CSV.write(save_path, results)
