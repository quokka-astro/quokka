import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.interpolate
from scikits.odes import ode
from math import sqrt
import pdb

def shock_jump(gamma, P0, M0):
    """Compute shock jump conditions for a radiative shock
    assuming an ideal gas, rho0 = 1, and T0 = 1."""

    def f1(T1): return 3.0*(gamma + 1.0)*(T1 - 1.0) - \
        P0*gamma*(gamma - 1.0)*(7.0 + T1**4)

    def f2(T1): return 12.0 * (gamma - 1.0)**2 * \
        T1 * (3.0 + gamma*P0*(1.0 + 7.0*T1**4))

    def rho(T1): return (f1(T1) + sqrt(f1(T1)**2 + f2(T1))) / \
        (6.0*(gamma - 1.0)*T1)

    def func(T1):
        rho1 = rho(T1)
        lhs = 3.0*rho1*(rho1*T1 - 1.0) + gamma*P0*rho1*(T1**4 - 1.0)
        rhs = 3.0*gamma*(rho1 - 1.0) * M0**2
        return lhs - rhs

    T1_guess = M0**2
    T1 = scipy.optimize.newton(func, T1_guess, tol=1e-7, rtol=1e-14)
    rho1 = rho(T1)
    v1 = M0 / rho1
    return rho1, T1, v1

def dTrad_dx_fun(rho, T, Trad, gamma=np.NaN, M0=np.NaN, P0=np.NaN, kappa=np.NaN):
    C_p = 1.0 / (gamma - 1.0)
    v = M0 / rho
    d_dx = v * ( 6*C_p*rho*(T-1) + 3*rho*(v**2 - M0**2) + 8 *
            P0*(Trad**4 - rho) ) / (24*P0*kappa*Trad**3)
    return d_dx

def Trad_fun(rho, T, gamma=np.NaN, M0=np.NaN, P0=np.NaN):
    Km = 3*(gamma*M0**2 + 1) + gamma*P0
    return ( (Km - 3*gamma*(M0**2 / rho) - 3*T*rho) / (gamma*P0) )**(1./4.)

def shock(M, y, dy_dM, gamma=np.NaN, M0=np.NaN, P0=np.NaN, kappa=np.NaN):
    """ rhs for the problem"""
    T = y[1]
    rho = M0 / (M*sqrt(T))
    Trad = Trad_fun(rho, T, gamma=gamma, M0=M0, P0=P0)
    dTrad_dx = dTrad_dx_fun(rho, T, Trad, gamma=gamma, M0=M0, P0=P0, kappa=kappa)

    r = 3*rho*sigma_a * (Trad**4 - T**4)
    Zd = 4*M0*Trad**3 * dTrad_dx + ( (gamma-1)/(gamma+1) ) * (gamma*M**2 + 1) * r
    Zn = 4*M0*Trad**3 * dTrad_dx + (gamma*M**2 - 1) * r

    dx_dM = -6 * ( (M0*rho*T) / ((gamma+1)*P0*M) ) * ((M**2 - 1)/Zd)
    dT_dM = -2 * ( (gamma-1)/(gamma+1) ) * (T/M) * (Zn/Zd)

    dy_dM[0] = dx_dM
    dy_dM[1] = dT_dM

    #pdb.set_trace()


def dT_dTrad(rho, T, Trad, gamma, M0, P0, sign=-1.0, sign2=1.0):
    C_p = 1.0/(gamma - 1.0)
    v = M0 / rho
    M = v / sqrt(T)

    Km = 3*(gamma*M0**2 + 1) + gamma*P0
    b = Km - gamma*P0*Trad**4
    d = sqrt(b**2 - 36*gamma*M0**2 * T)

    drho_T = -(rho + sign*3*gamma*M0**2 / d) / T
    drho_Trad = (-2./3.)*(P0*gamma*Trad**3 / T)*(1 + sign*(Km - gamma*P0*Trad**4)/d)

    c1 = M0 / (24*P0*kappa*rho**2 * Trad**3)
    c2 = P0 / (3*C_p*M0*(M**2 - 1))

    dG_T = c1*(6*C_p*rho*(2*drho_T*(T-1) + rho) - 6*M0**2 *
               rho*drho_T + 8*P0*drho_T*(Trad**4 - 2*rho))
    dG_Trad = c1*(12*C_p*drho_Trad*rho*(T-1) - 6*M0**2 * rho *
                  drho_Trad + 8*P0*(drho_Trad*(Trad**4 - 2*rho) + 4*rho*Trad**3))
    dF_T = c2*(4*v*Trad**3 * dG_T - 12*sigma_a*(gamma*M**2 - 1)*T**3)
    dF_Trad = c2*(4*v*Trad**3 * dG_Trad + 12*sigma_a*(gamma*M**2 - 1)*Trad**3)

    sign2 = 1.0
    result = (dF_T - dG_Trad + sign2*sqrt((dF_T - dG_Trad)**2 + 4*dG_T*dF_Trad)) / (2*dG_T)
    if (result > 0.):
        return result
    else:
        sign2 = -1.0
        return (dF_T - dG_Trad + sign2*sqrt((dF_T - dG_Trad)**2 + 4*dG_T*dF_Trad)) / (2*dG_T)

## compute asymptotic states
gamma = 5./3.
P0 = 1.0e-4
M0 = 30.0

T0 = 1.0
Trad0 = T0
rho0 = 1.0
rho1, T1, v1 = shock_jump(gamma, P0, M0)
Trad1 = T1
M1 = v1
P1 = P0 * T1**3 / rho1

print(f"rho1 = {rho1}")
print(f"T1 = {T1}")
print(f"v1 = {v1}")
print(f"post-shock radiation-to-gas pressure ratio = {P1}")

## define absorption coefficients, diffusivity
sigma_a = 1e6           # absorption coefficient
c = 100.0 * (M0 + 1.0)  # dimensionless speed of light
#kappa = sigma_a / c     # diffusivity
kappa = 1

## compute solution
# eps is slightly problem dependent -- if too small (for high Mach numbers), will cause CVODE errors:
#   [CVODE ERROR]  CVode
#     At t = 4.99999, mxstep steps taken before reaching tout.
eps = 1e-5
assert(eps <= 1e-3) # but don't make it larger than 1e-3, otherwise solutions are quite wrong
epsA = eps
epsB = -eps
Trad_epsA = Trad0 + epsA
Trad_epsB = Trad1 + epsB
sign_A = -1.0
sign_B = 1.0    # problem dependent!!
T_epsA = T0 + epsA * dT_dTrad(rho0, T0, Trad0, gamma, M0, P0, sign=sign_A)
T_epsB = T1 - epsB * dT_dTrad(rho1, T1, Trad1, gamma, M0, P0, sign=sign_B)

def rho(T, Trad, sign): 
    Km = 3*(gamma*M0**2 + 1) + gamma*P0
    return ((Km - gamma*P0*Trad**4) + sign *
            sqrt((Km - gamma*P0*Trad**4)**2 - 36*gamma*M0**2 * T)) / (6*T)

rho_epsA = rho(T_epsA, Trad_epsA, sign_A)
rho_epsB = rho(T_epsB, Trad_epsB, sign_B)
assert(np.allclose(rho1,rho_epsB,rtol=2*eps))  # if triggered, this indicates the wrong sign_B was chosen
print(f"rho_epsA = {rho_epsA}")
print(f"rho_epsB = {rho_epsB}")
v_epsA = M0 / rho_epsA
v_epsB = M0 / rho_epsB
Traddot_epsA = dTrad_dx_fun(rho_epsA, T_epsA, Trad_epsA, gamma=gamma, M0=M0, P0=P0, kappa=kappa)
Traddot_epsB = dTrad_dx_fun(rho_epsB, T_epsB, Trad_epsB, gamma=gamma, M0=M0, P0=P0, kappa=kappa)
print(f"Traddot_epsA = {Traddot_epsA}")
print(f"Traddot_epsB = {Traddot_epsB}")

y0_A = np.array([-epsA/Traddot_epsA, T_epsA])  # initial conditions
y0_B = np.array([ epsB/Traddot_epsB, T_epsB])

x0_A = v_epsA / sqrt(T_epsA)
x0_B = v_epsB / sqrt(T_epsB)

# avoid the Adiabatic Sonic Point (only needed for *continuous* solutions w/out hydro shock)
#eps_ASP = 1.0e-5
eps_ASP = 0.
x1_A = 1 + eps_ASP
x1_B = 1 - eps_ASP

print(f"Left-side initial conditions = ({x0_A}, {y0_A})")
print(f"Right-side initial conditions = ({x0_B}, {y0_B})")

## integrate ODE
xsol_A = np.linspace(x0_A, x1_A, 1024, endpoint=True)
xsol_B = np.linspace(x0_B, x1_B, 1024, endpoint=True)
fun = lambda x,y,ydot: shock(x,y,ydot, gamma=gamma, P0=P0, M0=M0, kappa=kappa)

options = {'max_steps': 50000, 'rtol': 1e-8, 'atol': 1e-12}
print("Integrating precursor region...")
solution_A = ode('cvode', fun, old_api=False, **options).solve(xsol_A, y0_A)
print("Integrating relaxation region...")
solution_B = ode('cvode', fun, old_api=False, **options).solve(xsol_B, y0_B)

## plot solution
M_A = solution_A.values.t
M_B = solution_B.values.t
x_A = solution_A.values.y[:,0]
x_B = solution_B.values.y[:,0]
T_A = solution_A.values.y[:, 1]
T_B = solution_B.values.y[:, 1]
vel_A = M_A * np.sqrt(T_A)
vel_B = M_B * np.sqrt(T_B)
rho_A = M0 / vel_A
rho_B = M0 / vel_B
Trad_A = Trad_fun(rho_A, T_A, gamma=gamma, P0=P0, M0=M0)
Trad_B = Trad_fun(rho_B, T_B, gamma=gamma, P0=P0, M0=M0)

print(f"Mach_A ranges from [{np.min(M_A)}, {np.max(M_A)}] = {np.max(M_A) - np.min(M_A):.3e}")
print(f"Mach_B ranges from [{np.min(M_B)}, {np.max(M_B)}] = {np.max(M_B) - np.min(M_B):.3e}")

## connect solutions
## -> solve 2d root-finding problem:
##      f(\Delta x_A, \Delta x_B) = 0, where
##      f(...) = | j_p(x=0) - j_s(x=0) |^2 + [Trad_p(x=0) - Trad_s(x=0)]^2
##      j_p = ( (\rho v)_p, (\rho v^2 + P)_p, [v(\rho E + P)]_p )
##      j_s = ( (\rho v)_s, (\rho v^2 + P)_s, [v(\rho E + P)]_s )

interp = lambda x,y: scipy.interpolate.interp1d(x,y,kind='cubic',fill_value='extrapolate')

rho_Afun = interp(x_A, rho_A)
vel_Afun = interp(x_A, vel_A)
T_Afun   = interp(x_A, T_A)
Trad_Afun= interp(x_A, Trad_A)

rho_Bfun = interp(x_B, rho_B)
vel_Bfun = interp(x_B, vel_B)
T_Bfun   = interp(x_B, T_B)
Trad_Bfun= interp(x_B, Trad_B)

def objective(dx):
    """compute the figure of merit f, defined above."""
    x0 = 0.
    xA = x0 - dx[0]
    xB = x0 - dx[1]

    rhoA = rho_Afun(xA)
    velA = vel_Afun(xA)
    TmatA = T_Afun(xA)
    TradA = Trad_Afun(xA)
    P_a = rhoA*TmatA / gamma
    E_a = TmatA / (gamma*(gamma-1)) + 0.5*(velA**2)

    rhoB = rho_Bfun(xB)
    velB = vel_Bfun(xB)
    TmatB = T_Bfun(xB)
    TradB = Trad_Bfun(xB)
    P_b = rhoA*TmatA / gamma
    E_b = TmatB / (gamma*(gamma-1)) + 0.5*(velB**2)

    j_p = np.array([rhoA*velA, rhoA*velA**2 + P_a, velA*(rhoA*E_a + P_a)])
    j_s = np.array([rhoB*velB, rhoB*velB**2 + P_b, velB*(rhoB*E_b + P_b)])

    #norm = np.sum((j_p - j_s)**2) + (TradA - TradB)**2     # bad
    norm = (rhoA*velA - rhoB*velB)**2 + (TradA - TradB)**2 + 0.1*dx[0]**2  # good
    return norm

dx_guess = np.array([-np.max(x_A), -np.min(x_B)])
print(f"dx_guess = {dx_guess}")

sol = scipy.optimize.minimize(objective, dx_guess, method='powell', tol=1e-10)
dx_A, dx_B = sol.x
print(f"objective = {sol.fun} after {sol.nit} iterations.")
#dx_A, dx_B = dx_guess
print(f"dx_A = {dx_A}")
print(f"dx_B = {dx_B}")

x_A += dx_A
x_B += dx_B
print(f"x_A ranges from [{np.min(x_A)}, {np.max(x_A)}] = {np.max(x_A) - np.min(x_A):.3e}")
print(f"x_B ranges from [{np.min(x_B)}, {np.max(x_B)}] = {np.max(x_B) - np.min(x_B):.3e}")
A_mask = (x_A <= 0.)
B_mask = (x_B >= 0.)

plt.plot(x_A[A_mask], rho_A[A_mask], color='blue', label='density')
plt.plot(x_A[A_mask], T_A[A_mask], color='black', label='gas temperature')
plt.plot(x_A[A_mask], Trad_A[A_mask], '-.', color='black', label='radiation temperature')
#plt.plot(x_A[A_mask], vel_A[A_mask], color='red', label='velocity')

plt.plot(x_B[B_mask], rho_B[B_mask], color='blue')
plt.plot(x_B[B_mask], T_B[B_mask], color='black')
plt.plot(x_B[B_mask], Trad_B[B_mask], '-.', color='black')
#plt.plot(x_B[B_mask], vel_B[B_mask], color='red')

plot_jump = True
if plot_jump:
    # plot temperature shock jump
    plt.scatter(x_A[A_mask][-1], T_A[A_mask][-1], color='black')
    plt.scatter(x_B[B_mask][-1], T_B[B_mask][-1], color='black')

    # plot density shock jump
    plt.scatter(x_A[A_mask][-1], rho_A[A_mask][-1], color='black')
    plt.scatter(x_B[B_mask][-1], rho_B[B_mask][-1], color='black')

# plot discarded (unphysical) regions of solutions
plot_discarded = True
if plot_discarded:
    plt.plot(x_A[~A_mask], rho_A[~A_mask], '--', color='blue', alpha=0.5)
    plt.plot(x_A[~A_mask], T_A[~A_mask], '--', color='black',  alpha=0.5)
    plt.plot(x_A[~A_mask], Trad_A[~A_mask], '--', color='black', alpha=0.5)
    #plt.plot(x_A[~A_mask], vel_A[~A_mask], '--', color='red', alpha=0.5)

    plt.plot(x_B[~B_mask], rho_B[~B_mask], '--', color='blue',  alpha=0.5)
    plt.plot(x_B[~B_mask], T_B[~B_mask], '--', color='black', alpha=0.5)
    plt.plot(x_B[~B_mask], Trad_B[~B_mask], '--', color='black', alpha=0.5)
    #plt.plot(x_B[~B_mask], vel_B[~B_mask], '--', color='red', alpha=0.5)

plt.legend(loc='best')
plt.title(f"M0 = {M0}, P0 = {P0}, kappa = {kappa}, sigma_a = {sigma_a:.1e}")

# Mach 1.05 plot
#plt.xlim(-0.015, 0.015)
#plt.ylim(1.0, 1.08)

# Mach 1.2 plot
#plt.xlim(-0.01, 0.01)
#plt.ylim(1.0, 1.3)

# Mach 1.4 plot
#plt.xlim(-0.01, 0.01)
#plt.ylim(1.0, 1.6)

# Mach 2 plot
#plt.xlim(-0.01, 0.01)
#plt.ylim(1, 2.5)

# Mach 3 plot
#plt.xlim(-0.01, 0.005)
#plt.ylim(1, 4.5)

# Mach 5 plot
#plt.xlim(-0.04, 0.01)
#plt.ylim(1, 11)

# Mach 27 plot
plt.xlim(-0.25, 0.05)
plt.ylim(1., 70.)

plt.tight_layout()
plt.savefig('ode_solution.pdf')
