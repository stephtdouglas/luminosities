"""Calculate X-ray luminosities."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.io.ascii as at


def calc_lx_rosat(xflux,distance=DISTANCE,unc_distance=distance_SD,
    null_val=-9999.):
    """Calculate Lx & unc_Lx. Flux is in erg s^-1 cm^-2; distance is in pc."""
    Lx = 4 * np.pi * xflux * (distance * pc_to_cm)**2
    Lx[xflux=null_val] = null_val

    # Propagate uncertainties to get uncertainty on Lx
    parderfX = 4 * np.pi * (distance * pc_to_cm)**2 # partial deriv. wrt f_X
    parderD = 8 * np.pi * (distance * pc_to_cm) * xfluxr
    unc_Lx = np.sqrt(parderfX**2 * xfluxr_sd**2 + 
                     parderD**2 * (distance_SD * pc_to_cm)**2)
    unc_Lx[xflux=null_val] = null_val

    return Lx, unc_Lx

def calc_lx_lbol(Lx,unc_Lx,Lbol,unc_Lbol,null_val=-9999):
    """Calculate Lx/Lbol (LL) and uncertainty (unc_LL)."""

    # Calculate the luminosity ratio
    LL = Lx / Lbol
    LL[(Lx==null_val) | (Lbol==null_val)] = null_val

    # Propagate uncertainties to get uncertainty on Lx/Lbol
    unc_LL = np.sqrt(Lbol**-2 * Lx_sd**2 + Lx**2 * Lbol**-4 * Lbol_sd**2)
    unc_LL[(Lx==null_val) | (Lbol==null_val)] = null_val

    return LL, unc_LL
