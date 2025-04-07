# Star Formation Recipe 

The star formation recipe is implemented via the Stochastic Stellar Population specialisation. Every cell is checked for Jeans violation and star particles are added if $\lambda_J = c_s/\sqrt{G\rho} < J \times dx$, where J is $0.5$. 

Adding star particle to every Jeans violating cell can lead to a large number of small mass stars which can be cumbersome. To alleviate this issue, we control the star formation rate through $\epsilon_{\rm ff}$ and $\epsilon_{*}$. $\epsilon_{\rm ff}$ is the efficiency per freefall time given by $=\dot{M}_* \cdot t_{\rm ff} / M_{\rm cell} \cdot dt $, while $\epsilon_{*}$ is the fraction of cell mass that gets converted into stars. 

There are two equivalent ways of converting a certain mass from cell into stars. 

In the first, we assume that every cell that is Jeans unstable definitely forms a star. In this scenario, $M_{*} = \epsilon_{\rm ff} \cdot dt \cdot M/t_{\rm ff}$ is the mass that gets converted into stars from a cell mass of M. This will lead to the issue mentioned earlier, that is of a large number of stars being created in a galaxy-scale simulation.

We circumvent this by associating a probability P to the process of spawning a star in the cell. A choice for P $=\epsilon_{\rm ff} \cdot dt/\epsilon_{*}$, which is ratio of the expected from the star formation rate in the cell over the timestep dt ($\epsilon_{\rm ff} dt/t_{\rm ff}$) to the fraction of mass available for star formation ($\epsilon_{\rm ff} M$). It should be noted that this is the probability for $\epsilon_{\rm ff}$ averaged over the timestep rather than for the instantaneous star formation rate, which should be integrated over the timestep dt. 

Both the descriptions of P, from assuming a star formation rate averaged over timestep or an instantaneous star formation rate, lead to identical expression of P if $dt<t_{\rm ff}$. However if we are in the regime of $dt>t_{\rm ff}$ neither prescriptions of P offer the correct answer because they rely on conditions at the beginning of the timestep. 

