"""Calculate X-ray luminosities."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.io.ascii as at


# ROSAT =======================================================================
# Calculate Lx & Lx_sd ------------------------------------
# flux is in erg s^-1 cm^-2; DISTANCE is in pc
Lx = 4 * np.pi * xfluxr * (DISTANCE * pc_to_cm)**2
parderfX = 4 * np.pi * (DISTANCE * pc_to_cm)**2 # partial deriv. wrt f_X
parderD = 8 * np.pi * (DISTANCE * pc_to_cm) * xfluxr
Lx_sd = np.sqrt(parderfX**2 * xfluxr_sd**2 + parderD**2 * (DISTANCE_SD * pc_to_cm)**2)

# Calculate LL & LL_sd ------------------------------------
LL = np.zeros(num)
LL_sd = np.zeros(num)
LL[irange] = Lx[irange] / Lbol[irange]
LL_sd[irange] = np.sqrt(Lbol[irange]**-2 * Lx_sd[irange]**2 \
                         + Lx[irange]**2 * Lbol[irange]**-4 * Lbol_sd[irange]**2)

outdata[:,7] = Lx
outdata[:,8] = Lx_sd
outdata[:,9] = LL
outdata[:,10] = LL_sd

