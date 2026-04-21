import yt
import numpy as np
import matplotlib.pyplot as plt

ds = yt.load("plt0000130")

# Column density projection along z-axis
proj = ds.proj(("boxlib", "gasDensity"), "z")
frb = proj.to_frb((20, "kpc"), 512)
density_proj = np.array(frb[("boxlib", "gasDensity")])

# Normalise by max (peak of ring at R~3 kpc)
Sigma_c0 = density_proj.max()
density_norm = density_proj / Sigma_c0

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    density_norm,
    origin="lower",
    norm=plt.matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e0),
    cmap="viridis",
    extent=[-10, 10, -10, 10],
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\Sigma/\Sigma_{c0}$", fontsize=12)
ax.set_xlabel("x (kpc)", fontsize=12)
ax.set_ylabel("y (kpc)", fontsize=12)

t_Myr = float(ds.current_time.to("Myr"))
ax.set_title(f"t = {t_Myr:.1f} Myr", fontsize=12)

plt.tight_layout()
plt.savefig("column_density.png", dpi=150)
plt.show()