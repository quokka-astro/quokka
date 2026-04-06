We solve the following equations for the density profile:

$$
\sigma_1^2 \frac{d \rho_1}{d z}=\rho_1\left(g_1+g_{\mathrm{ext}}\right) \\
\frac{d g_1}{d z}=4 \pi G \rho_1,
$$

where $\sigma_1 = 7$ km/s is the velocity dispersion, and $\rho_1$ and $g_1$ are the density and gravitational acceleration due to gas component, and $g_{\mathrm{ext}}=-d \Phi_{\mathrm{ext}} / \mathrm{dz}$ is the gravitational acceleration due to external potential (from stars plus dark matter). We solve these equations subject to the constraints that $g_1(z=0)=0$ and that $\Sigma_{\text {gas }}= 2 \int_0^{\infty} \rho_1 \mathrm{dz}$. The second constraint requires an iterative approach, so we take an initial guess value for $\rho_1(z=0) \equiv \rho_{1,0} = 1 m_H~\mathrm{cm}^{-3}$, solve the equations and integrate to find $\boldsymbol{\Sigma}_{\text {gas }}$, and then iteratively adjust the guess until we find the value of $\rho_{1,0}$ that yields the desired value of $\Sigma_{\text {gas }}$.

The external gravitational potential, $\Phi_{\rm ext}$, is set by the dark matter halo potential and a stellar disc. The dark matter potential is adapted from Kuijken \& Gilmore (1989), and the total potential from the dark matter halo and the stellar disc (reproduced from Kim \& Ostriker 2017) is,

$$
\begin{aligned}
\Phi_{\mathrm{ext}} & =2 \pi G \Sigma_* z_*\left[\left(1+\frac{z^2}{z_*^2}\right)^{1 / 2}-1\right] \\
& +2 \pi G \rho_{\mathrm{dm}} R_0^2 \ln \left(1+\frac{\mathrm{z}^2}{\mathrm{R}_0^2}\right) .
\end{aligned}
$$

Here, $\Sigma_*=42 \mathrm{M}_{\odot} \mathrm{pc}^{-2}, z_*=245 \mathrm{pc}, \rho_{\mathrm{dm}}=6.4 \times 10^{-3} \mathrm{M}_{\odot} \mathrm{pc}^{-3}$ and $R_0$ is the Galactocentric radius of our simulation box, which we set to be 8 kpc. $g_{\mathrm{ext}}=-d \Phi_{\mathrm{ext}} / d \mathrm{z}$

Requirement:

1. Keep all the variables as free parameters that I can change later.
2. Write a final solution in a CSV file, expressed with dimensionless parameters: $\xi = \xi(\theta)$, where $\xi \equiv \rho / \rho_{1,0}$ and $\theta \equiv z/z_*$, $\theta \in [0, 20]$.
3. Validate the code by first running `source ~/rc/yt.rc` to load the yt environment, then executing the python script in its folder.
