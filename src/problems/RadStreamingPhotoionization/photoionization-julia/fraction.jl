using Pkg
PATH = "./src/problems/RadStreamingPhotoionization/photoionization-julia"
Pkg.activate(PATH)
Pkg.instantiate()

using DifferentialEquations
using Plots
using CSV
using DataFrames

const m_p = 1.6726219e-24
const m_e = 9.10938356e-28
const c   = 2.99792458e10
const k_B = 1.380649e-16
const h   = 6.62606957e-27
const R   = 8.314462618e7
const σ_v = 1.5e-18
const E_ion = 6.4e-12

struct State
    n_spec::Vector{Float64}
    n_rad::Vector{Float64}
    γ_vec::Vector{Float64}
    spec_mass::Vector{Float64}
    E::Float64
end

function get_mean_μ(state::State)
    μ = 0.0
    for i in 1:length(state.n_spec)
        μ += state.n_spec[i] * state.spec_mass[i]
    end
    return μ / sum(state.n_spec)
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
    ρ = get_ρ(state)
    μ = get_mean_μ(state)
    for i in 1:length(state.n_spec)
        γ += (state.n_spec[i] * m_p / ρ) * (1.0 / (state.γ_vec[i] - 1.0))
    end
    return γ * μ
end

function get_T(state::State)
    μ_mean = get_mean_μ(state)
    γ_inv = get_γ_inv(state)
    T = state.E * μ_mean / (R * γ_inv)
    return T
end

function get_E(state::State, T::Float64)
    # The state should contain a dummy value for energy
    μ_mean = get_mean_μ(state)
    γ_inv = get_γ_inv(state)
    E = R * T * γ_inv / μ_mean
    return E
end

function rhs!(df::Vector{Float64}, f::Vector{Float64}, params, t)
    # Unpack the state variables
    state = State(f[1:3], f[4:5], [5/3, 5/3, 5/3], [m_e, m_p, m_p], f[6])
    n_spec = state.n_spec
    n_rad = state.n_rad
    ρ = get_ρ(state)
    T = get_T(state)

    n_e = n_spec[1]
    n_HI = n_spec[2]
    n_HII = n_spec[3]
    n_photon = n_rad[1]
    flux_photon = n_rad[2]

    α_rec = 2.6e-13 * (T / 1.0e4)^(-0.7)
    ionization_term = n_HI * c * σ_v * n_photon
    recombination_term = α_rec * n_e * n_HII

    df[1] =  ionization_term - recombination_term
    df[2] = -ionization_term + recombination_term
    df[3] =  ionization_term - recombination_term
    df[4] = -ionization_term
    df[5] = -ionization_term * flux_photon / n_photon
    df[6] = (ionization_term * E_ion - recombination_term * k_B * T * (0.684 - 0.0416 * log(T / 1.0e4))) / ρ
end

function make_problem(state::State, tend::Float64)
    f = vcat(state.n_spec, state.n_rad, state.E)
    prob = ODEProblem(rhs!, f, (0.0, tend), nothing)
    return prob
end

E_rad0 = 2.93e-7
freq_low, freq_high = 3.29e15, 1.50e16
avg_freq = 0.5 * (freq_low + freq_high)
n_photon0 = E_rad0 / (h * avg_freq)
n_e, n_HI, n_HII = 0.0, 1.0e2, 0.0
γ_e, γ_HI, γ_HII = 5/3, 5/3, 5/3
T0 = 1.0e3
E0 = get_E(State([n_e, n_HI, n_HII], [n_photon0, 0], [γ_e, γ_HI, γ_HII], [m_e, m_p, m_p], 0.0), T0)
state = State(
    [0.0, 1.0e2, 0.0],   # n_spec: [n_e, n_HI, n_HII]
    [n_photon0, 0],      # n_rad: [n_photon, flux_photon]
    [5/3, 5/3, 5/3],     # γ_vec
    [m_e, m_p, m_p],     # spec_mass
    E0                   # E
)
prob = make_problem(state, 8.0e3)
sol = solve(prob, saveat=50.0)

T_values = [get_T(State(sol[i][1:3], sol[i][4:5], [5/3, 5/3, 5/3], [m_e, m_p, m_p], sol[i][6])) for i in eachindex(sol)]

results = DataFrame(time=sol.t, n_e=sol[1, :], n_HI=sol[2, :], n_HII=sol[3, :], n_photon=sol[4, :], flux_photon=sol[5, :], E=sol[6, :], T=T_values)
CSV.write(PATH*"/photoionization-julia.csv", results)
