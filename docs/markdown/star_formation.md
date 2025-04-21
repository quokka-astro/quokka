# Star Formation Recipe 

## Checking for Jeans Violation
The star formation recipe is implemented via the Stochastic Stellar Population specialisation. Every cell is checked for Jeans violation and star particles are added if $\lambda_J = c_s/\sqrt{G\rho} < J \cdot dx$, where J is $0.5$. 

Adding star particle to every Jeans violating cell can lead to a large number of small mass stars which can be cumbersome. To alleviate this issue, we control the star formation rate through $\epsilon_{\rm ff}$ and $\epsilon_{*}$. $\epsilon_{\rm ff}$ is the efficiency per freefall time given by $\left( \frac{\dot{M}_{*}$ t $_{\rm ff}}{M_{\rm cell} dt} \right) $
, while $\epsilon_{*}$ is the fraction of cell mass that gets converted into stars. 

There are two equivalent ways of converting a certain mass from cell into stars. 

In the first, we assume that every cell that is Jeans unstable definitely forms a star. In this scenario, $M_{*} = \epsilon_{\rm ff} \cdot dt \cdot M/t_{\rm ff}$ is the mass that gets converted into stars from a cell mass of M. This will lead to the issue mentioned earlier, that is of a large number of stars being created in a galaxy-scale simulation.

We circumvent this by associating a probability P to the process of spawning a star in the cell. A choice for P $=\epsilon_{\rm ff} \cdot dt/\epsilon_{*}$, which is ratio of the expected from the star formation rate in the cell over the timestep dt ($\epsilon_{\rm ff} dt/t_{\rm ff}$) to the fraction of mass available for star formation ($\epsilon_{\rm ff} M$). It should be noted that this is the probability for $\epsilon_{\rm ff}$ averaged over the timestep rather than for the instantaneous star formation rate, which should be integrated over the timestep dt. 

Both the descriptions of P, from assuming a star formation rate averaged over timestep or an instantaneous star formation rate, lead to identical expression of P if $dt<t_{\rm ff}$. However if we are in the regime of $dt>t_{\rm ff}$ neither prescriptions of P offer the correct answer because they rely on conditions at the beginning of the timestep. 


## Estimating number and type of stars

Once we establish that a star will form in a cell, we need to find out the stellar population looks like. In our implementation, the population comprises at least particle representing low mass star ($M<8M_{\odot}$) population and a random number of particles, each represeting a high mass star.

The number of high mass stars is determined by the initial mass function, taken to be Chabrier03. The IMF is represented by a log normal for $M<M_{\odot}$ and a power law with a slope of $2.35$ beyond that. From the IMF, we can estimate the fraction of high mass stars, $f_{*,\rm{high}}$, and the average mass of high mass stars, $\langle m \rangle _{*,\rm{high}}$. The expectation value of the number of high mass stars, *num_high_mass_stars*, then is the ratio of the total mass in high mass stars and $\langle m \rangle _{*,\rm{high}}$. 

The total number of star particles to be spwaned in a cell is $1+$ Poission distribution with an expectation value of *num_high_mass_stars*. A fraction $1-f_{*,\rm{high}}$ goes into the particle representing low mass star while the mass of the high mass star particles is randomly drawn from the high mass part of the IMF.

In the case there are no high mass stars in the cell, the low mass star particle gets spawned at the centre of the cell with a velocity identical to the gas velocity. 



## Estimating the Velocity of the High Mass Star(s)

In the presence of high mass stars, we first assign momentum to these stars. The velocity of these particles is derived from a log normal distribution. The mean of this distribution is the cell velocity while its dispersion is obtained by mass-averaging over the neighbouring cells. 

We track the total momentum of the high mass star particles in the three directions and in order to conserve momentum we impart an equal and opposite momentum to the low mass star. 