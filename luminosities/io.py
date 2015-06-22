"""Read in relevant files: SED table, bolometric corrections."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.io.ascii as at

def read_SEDs():
    """Read SEDs table (Adam's table)."""
    kh = at.read('/home/stephanie/code/python/luminosities/models/kraushillenbrand5.dat')

    # Save relevant arrays as variables
    coltemp = kh["Teff"]
    gmag =kh["Mg"]
    rmag =kh["Mr"]
    imag =kh["Mi"]
    numrows = len(rmag)

    # Interpolation functions for Teff as a function of Absolute Magnitude
    gfunc = interp1d(gmag, coltemp, kind='linear')
    rfunc = interp1d(rmag, coltemp, kind='linear')
    ifunc = interp1d(imag, coltemp, kind='linear')

    # I don't know what the following is doing
    slopes = np.zeros((numrows - 1, 3))
    slopes[:, 0] = np.abs(np.diff(coltemp) / np.diff(gmag))
    slopes[:, 1] = np.abs(np.diff(coltemp) / np.diff(rmag))
    slopes[:, 2] = np.abs(np.diff(coltemp) / np.diff(imag))

    # Magnitude ranges where the interpolation functions are valid
    # (for g,r,i)
    mags = {"g":gmag,"r":rmag,"i":imag}
    magranges = {"g":[-0.39,20.98], "r":[-0.04,18.48], "i":[0.34,15.85]}
    slopes_dict = {"g":slopes[:, 0],"r":slopes[:, 1],"i":slopes[:, 2]}

    return mags, magranges, slopes_dict


def read_BCs()
   """Read BCs table (Girardi 2004)."""
   table_gir = tools.read_table(FOLDER_DATA + BCTABLE, raw=True)
   colTeff = np.array(table_gir[1])
   collogg = np.array(table_gir[2])
   colBCg = np.array(table_gir[4])
   colBCr = np.array(table_gir[5])
   colBCi = np.array(table_gir[6])
   iM37g = np.where(collogg == LOG_G)[0]
   bcfuncg = interp1d(colTeff[iM37g], colBCg[iM37g], kind='linear')
   bcfuncr = interp1d(colTeff[iM37g], colBCr[iM37g], kind='linear')
   bcfunci = interp1d(colTeff[iM37g], colBCi[iM37g], kind='linear')
   slopesBC = np.zeros((len(colTeff[iM37g]) - 1,3))
   for it,te in enumerate(colTeff[iM37g]):
       if it == len(colTeff[iM37g]) - 1: continue
       slopesBC[it,0]= np.abs((colBCg[iM37g][it+1] - colBCg[iM37g][it]) / 
                              (colTeff[iM37g][it+1] - te))
       slopesBC[it,1]= np.abs((colBCr[iM37g][it+1] - colBCr[iM37g][it]) / 
                              (colTeff[iM37g][it+1] - te))
       slopesBC[it,2]= np.abs((colBCi[iM37g][it+1] - colBCi[iM37g][it]) / 
                              (colTeff[iM37g][it+1] - te))

