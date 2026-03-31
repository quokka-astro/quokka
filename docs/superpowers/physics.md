
## Equations of radiation-gas-dust coupling

We solve the radiation-matter coupling equations:

```
d/dt Egas = c G
d/dt Erad[n] = - chat G[n]         // Eq. 1
```

where 

```
c G[n] = c (xidE[n] Erad[n] - xidB[n] planck[n])
```

and `planck[n]` is Planck function at dust temperature `Td` multiplied by `(4 pi / c)` integrated over the n'th radiation group.

We consider dust imperfectly coupled to gas via thermal conduction, and assume dust is always in local thermal equilibrium between gas and radiation. Let `Λgd[n]` be the radiative cooling rate of dust at frequency group `n` : 

```
Λgd[n] = - c G[n] = - c ( xidE[n] Erad[n] - xidB[n] planck[n] )      // Eq. 2
```

and the thermal exchange from gas to dust, `Λgd`, should balance with dust's radiative cooling, i.e. `Λgd = sum( Λgd[n] )`, and we use an empirical relation for `Λgd` :

```
Λgd = Θgd n_H^2 sqrt(T) (T - Td)      // Eq. 3
```

## Numerical method

We want to solve Eq. 1 using a single Backward Euler step, by solving the implicit equation

```
Egas - Egas0 = c G dt = - dt sum( Λgd[n] )
Erad[n] - Erad0[n] = - chat G[n] dt = (chat / c) Λgd[n] dt
```

where `Egas0` and `Erad0` denotes the gas internal energy and radiation energy at time `t`, and other variables are at time `t + dt`. We define the residuals:

```
F[0] = Egas - Egas0 + dt sum( Λgd[n] )
F[n] = Erad[n] - Erad0[n] - (chat / c) dt Λgd[n]
```

Solving the implicit equation becomes finding the root of the residuals.

To simplify the problem, we define 

```
R[n] = - chat G[n] dt = (chat / c) Λgd[n] dt      // Eq. 4
```

Then, we can rewrite Eq. 3 as

```
sum( R[n] ) = (chat / c) dt Θgd n_H^2 sqrt(T) (T - Td)      // Eq. 5
```

Then, the residuals become

```
F[0] = Egas - Egas0 + (c / chat) sum( R[n] )
F[n] = Erad[n] - Erad0[n] - R[n]                    // Eq. 6
```

Our goal becomes finding the root of the new residual equations `F[0]=0; F[n]=0` using the Newton-Raphson iteration method on the base `(Egas, R[n])`. Working on this base has huge advantages over working on the basis `(Egas, Erad[n])` because the dust temperature `Td` is trivially derivable, as we will see later. 

For convenience, we define the following variables:

```
Nd = (chat / c) Θgd n_H^2 dt  # dust-gas interaction constants
tau[n] = chat dt xidB[n]      # characteristic optical depth
X[n] = xidB[n] / xidE[n]      # ratio of Planck-mean to energy-mean absorption coefficient of the dust
```

then, we can show from Eq. 5

```
sum( R[n] ) = Nd sqrt(T) (T - Td)      // Eq. 7
```

We can also express `Erad[n]` in terms of `R[n]`:

```
Erad[n] = - 1 / tau[n] * X[n] * R[n] + X[n] planck[n]
```

Then, we can write down the Jacobian matrix of Eq. 6:

```
J[0][0] = 1
J[0][n] = c / chat
J[n][0] = d Erad[n] / d Egas = (1/CV) d Erad[n] / d T = (1/CV) X[n] (d planck[n] / d Td) (d Td / d T)
J[n][n] = - 1 / tau[n] * X[n] + X[n] (d planck[n] / d Td) (d Td / d R[n]) - 1
```

Note that we have assumed that `d xidE[n] / d T = 0` and `d xidB[n] / d T = 0`. 

What is `d Td / d T` ? From Eq. 7 one can easily show that, at fixed `R[n]`,

```
d Td / d T = 3/2 - Td / (2 T)
```

What is `d Td / d R[n]` ? From Eq. 7 one can show that, at fixed `T`, 

```
d Td / d R[n] = - 1 / (Nd sqrt(T))
```

for all `n`. 

How to solve for `Td` given `Egas` and `Erad[n]`? Eq. 7 implies

```
Td = T - sum( R[n] ) / (Nd sqrt(T))
```

### A special case: dust perfectly coupled to gas

In most test problems in the Quokka code, when dust is not enabled, we assume that dust is perfectly coupled to gas, thus

```
Td = T
d Td / d T = 1
At fixed T: d Td / d R[n] = 0
```

and the Jacobian is simplified:


```
J[0][0] = 1
J[0][n] = c / chat
J[n][0] = (1/CV) X[n] (d planck[n] / d Td)
J[n][n] = - 1 / tau[n] * X[n] - 1
```

