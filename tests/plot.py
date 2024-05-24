import numpy as np
import matplotlib.pyplot as plt
import yt

# N = 0
# ds = yt.load(f"plt{N:05d}")
# yt.SlicePlot(ds, "z", ('boxlib', 'gasDensity')).save(f"pulse-den-snap{N}.pdf")
# yt.SlicePlot(ds, "z", ('boxlib', 'x-GasMomentum')).save(f"pulse-vx-snap{N}.pdf")

N = 113
ds = yt.load(f"plt{N:05d}")
p = yt.SlicePlot(ds, "z", ('boxlib', 'gasDensity')).save(f"pulse-den-snap{N}.pdf")
# set colorbar units
p = yt.SlicePlot(ds, "z", ('boxlib', 'x-GasMomentum'))
p.set_unit(('boxlib', 'x-GasMomentum'), 'cm/s')
p.save(f"pulse-vx-snap{N}.pdf")
# yt.SlicePlot(ds, "z", ('boxlib', 'y-GasMomentum')).save(f"pulse-vy-snap{N}.pdf")

