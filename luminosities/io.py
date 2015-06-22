"""Read in relevant files: SED table, bolometric corrections."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.io.ascii as at

def read_SEDs():
    """Read SEDs table (Adam's table)."""
    kh = tools.read_table(FOLDER_DATA + KHTABLE5, raw=True)
    coltemp = np.array(kh[10])
    gmag = np.array(kh[2])
    rmag = np.array(kh[3])
    imag = np.array(kh[4])
    numrows = len(rmag)
    gfunc = interp1d(gmag, coltemp, kind='linear')
    rfunc = interp1d(rmag, coltemp, kind='linear')
    ifunc = interp1d(imag, coltemp, kind='linear')
    slopes = np.zeros((numrows - 1, 3))
    for ir in range(numrows):
        if ir == numrows - 1: 
            continue
        slopes[ir, 0] = np.abs((coltemp[ir+1] - coltemp[ir]) / 
                               (gmag[ir+1] - gmag[ir]))
        slopes[ir, 1] = np.abs((coltemp[ir+1] - coltemp[ir]) / 
                               (rmag[ir+1] - rmag[ir]))
        slopes[ir, 2] = np.abs((coltemp[ir+1] - coltemp[ir]) / 
                               (imag[ir+1] - imag[ir]))
    magranges = [[-0.39,20.98], [-0.04,18.48], [0.34,15.85]] # gri; outside of these, func not valid!


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

