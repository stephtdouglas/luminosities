"""Read in relevant files: SED table, bolometric corrections."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.io.ascii as at

from constants import *

def read_SEDs():
    """Read SEDs table (Adam's table)."""
    kh = at.read(model_dir+'kraushillenbrand5.dat')

    # Save relevant arrays as variables
    coltemp = kh["Teff"]
    gmag = kh["Mg"]
    rmag = kh["Mr"]
    imag = kh["Mi"]
    numrows = len(rmag)

    # Interpolation functions for Teff as a function of Absolute Magnitude
    gfunc = interp1d(gmag, coltemp, kind='linear')
    rfunc = interp1d(rmag, coltemp, kind='linear')
    ifunc = interp1d(imag, coltemp, kind='linear')

    # Save the slopes separately, for computing uncertainties later
    slopes = np.zeros((numrows - 1, 3))
    slopes[:, 0] = np.abs(np.diff(coltemp) / np.diff(gmag))
    slopes[:, 1] = np.abs(np.diff(coltemp) / np.diff(rmag))
    slopes[:, 2] = np.abs(np.diff(coltemp) / np.diff(imag))

    # Magnitude ranges where the interpolation functions are valid
    # (for g,r,i)
    magranges = {"g":[-0.39,20.98], "r":[-0.04,18.48], "i":[0.34,15.85]}
    mags = {"g":gmag,"r":rmag,"i":imag}
    funcs = {"g":gfunc,"r":rfunc,"i":ifunc}
    slopes_dict = {"g":slopes[:, 0],"r":slopes[:, 1],"i":slopes[:, 2]}

    return mags, funcs, magranges, slopes_dict


def read_BCs(log_g=LOG_G):
   """Read BCs table (Girardi 2004), compute interp1d functions at log_g."""

   table_gir = at.read(model_dir+BCTABLE)

    # Save relevant arrays as variables
   colTeff = table_gir["Teff"]
   collogg = table_gir["logg"]
   colBCg =  table_gir["g"]
   colBCr =  table_gir["r"]
   colBCi =  table_gir["i"]

   # Only keep log_g for dwarfs
   iM37g = np.where(collogg==log_g)[0]

   # Compute interpolation functions
   bcfuncg = interp1d(colTeff[iM37g], colBCg[iM37g], kind='linear')
   bcfuncr = interp1d(colTeff[iM37g], colBCr[iM37g], kind='linear')
   bcfunci = interp1d(colTeff[iM37g], colBCi[iM37g], kind='linear')

   # Save the slopes separately, for computing uncertainties later
   slopesBC = np.zeros((len(colTeff[iM37g]) - 1,3))
   slopesBC[:,0]= np.abs(np.diff(colBCg[iM37g]) / np.diff(colTeff[iM37g]))
   slopesBC[:,1]= np.abs(np.diff(colBCr[iM37g]) / np.diff(colTeff[iM37g]))
   slopesBC[:,2]= np.abs(np.diff(colBCi[iM37g]) / np.diff(colTeff[iM37g]))

    # Teff ranges where the interpolation functions are valid
    # (for g,r,i)
    teffrange = [min(colTeff[iM37g])*1.00001,max(colTeff[iM37g])*0.99999]
    bcs = {"g":colBCg,"r":colBCr,"i":colBCi}
    funcs = {"g":bcfuncg,"r":bcfuncr,"i":bcfunci}
    slopes_dict = {"g":slopesBC[:, 0],"r":slopesBC[:, 1],"i":slopesBC[:, 2]}
    teff_bins = colTeff[iM37g]

    return funcs, teffrange, slopes_dict, teff_bins
