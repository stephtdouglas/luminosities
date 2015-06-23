'''
Uses Adam Kraus (2007) stellar SEDs table to convert magnitudes into temperatures. Then it uses DUSTYAGB07 tables to obtain bolometric corrections. Then it calculates Lx_r, Lx_f, and Lbol.
'''

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tools


# Read my consolidated photometry file ----------------------------------------
table_photo = tools.read_table(FOLDER_DATA + 'allphotometry.txt',raw=True)
ids = np.array(table_photo[0])
num = len(ids)

# Initialize some arrays
temps = np.zeros(num)
temps_sd = np.zeros(num)
BCs = np.zeros(num)
BCs_sd = np.zeros(num)
Mbol = np.array([np.nan] * num)
Mbol_sd = np.array([np.nan] * num)
Lbol = np.zeros(num)
Lbol_sd = np.zeros(num)
outdata = np.zeros([num, 11])
outdata[:,0] = ids

magsg = np.array(table_photo[14])
magsg_sd = np.array(table_photo[17])
magsr = np.array(table_photo[15])
magsr_sd = np.array(table_photo[18])
magsi = np.array(table_photo[16])
magsi_sd = np.array(table_photo[19])
xfluxr = np.array(table_photo[8])
xfluxr_sd = np.array(table_photo[21])

# r mag =======================================================================
izeros = np.where(magsr != 0)[0]

# Calculate Mr and Mr_sd ----------------------------------
mags_abs = np.array([np.nan] * num)
mags_abs_sd = np.array([np.nan] * num)
mags_abs[izeros] = (magsr[izeros] - EXTINCTIONr) - 5 * (np.log10(DISTANCE) - 1)
arrDfac = np.array([5 * np.log10(1 + 10**(np.log10(DISTANCE_SD) - np.log10(DISTANCE)))] * num)
mags_abs_sd[izeros] = np.sqrt(magsr_sd[izeros]**2 + arrDfac[izeros]**2)
irange = np.where((mags_abs >= magranges[1][0]) & (mags_abs <= magranges[1][1]))[0]

# Convert abs mag to T and calculate T_sd -----------------
temps[irange] = rfunc(mags_abs[irange])
digitized = np.digitize(mags_abs[irange], rmag[1:]) # Bin Mr using Adam's table as bins
for ib,slp in enumerate(slopes[:,1]):
    ibin = np.where(digitized == ib)[0]
    temps_sd[irange[ibin]] = slp * mags_abs_sd[irange[ibin]]

# Get BC and BC_sd using T --------------------------------
BCs[irange] = bcfuncr(temps[irange])
digitized = np.digitize(temps[irange], colTeff[iM37g][1:]) # Bin T using Girardi's table as bins
for ib,slp in enumerate(slopesBC[:,1]):
    ibin = np.where(digitized == ib)[0]
    BCs_sd[irange[ibin]] = slp * temps_sd[irange[ibin]]

# Calculate Mbol and Mbol_sd ------------------------------
Mbol[irange] = mags_abs[irange] + BCs[irange]
Mbol_sd[irange] = np.sqrt(mags_abs_sd[irange]**2 + BCs_sd[irange]**2)

# g mag =======================================================================
# Calculate Mg and Mg_sd for sources missing r mag --------
izeros = np.where((magsr == 0) & (magsg != 0))[0]

mags_abs = np.array([np.nan] * num)
mags_abs_sd = np.array([np.nan] * num)
mags_abs[izeros] = (magsg[izeros] - EXTINCTIONg) - 5 * (np.log10(DISTANCE) - 1)
mags_abs_sd[izeros] = np.sqrt(magsg_sd[izeros]**2 + arrDfac[izeros]**2)
irange = np.where((mags_abs >= magranges[0][0]) & (mags_abs <= magranges[0][1]))[0]

# Convert abs mag to T and calculate T_sd -----------------
temps[irange] = gfunc(mags_abs[irange])
digitized = np.digitize(mags_abs[irange], gmag[1:])
for ib,slp in enumerate(slopes[:,0]):
    ibin = np.where(digitized == ib)[0]
    temps_sd[irange[ibin]] = slp * mags_abs_sd[irange[ibin]]

# Get BC and BC_sd using T --------------------------------
BCs[irange] = bcfuncg(temps[irange])
digitized = np.digitize(temps[irange], colTeff[iM37g][1:]) # Bin T using Girardi's table as bins
for ib,slp in enumerate(slopesBC[:,0]):
    ibin = np.where(digitized == ib)[0]
    BCs_sd[irange[ibin]] = slp * temps_sd[irange[ibin]]

# Calculate Mbol and Mbol_sd ------------------------------
Mbol[irange] = mags_abs[irange] + BCs[irange]
Mbol_sd[irange] = np.sqrt(mags_abs_sd[irange]**2 + BCs_sd[irange]**2)

# i mag =======================================================================
# Calculate Mi and Mi_sd for sources missing gr mag -------
izeros = np.where(((magsr == 0) & (magsg == 0)) & (magsi != 0))[0]

mags_abs = np.array([np.nan] * num)
mags_abs_sd = np.array([np.nan] * num)
mags_abs[izeros] = (magsi[izeros] - EXTINCTIONi) - 5 * (np.log10(DISTANCE) - 1)
mags_abs_sd[izeros] = np.sqrt(magsi_sd[izeros]**2 + arrDfac[izeros]**2)
irange = np.where((mags_abs >= magranges[2][0]) & (mags_abs <= magranges[2][1]))[0]

# Convert abs mag to T and calculate T_sd -----------------
temps[irange] = ifunc(mags_abs[irange])
digitized = np.digitize(mags_abs[irange], imag[1:])
for ib,slp in enumerate(slopes[:,2]):
    ibin = np.where(digitized == ib)[0]
    temps_sd[irange[ibin]] = slp * mags_abs_sd[irange[ibin]]

# Get BC and BC_sd using T --------------------------------
BCs[irange] = bcfunci(temps[irange])
digitized = np.digitize(temps[irange], colTeff[iM37g][1:]) # Bin T using Girardi's table as bins
for ib,slp in enumerate(slopesBC[:,2]):
    ibin = np.where(digitized == ib)[0]
    BCs_sd[irange[ibin]] = slp * temps_sd[irange[ibin]]

# Calculate Mbol and Mbol_sd ------------------------------
Mbol[irange] = mags_abs[irange] + BCs[irange]
Mbol_sd[irange] = np.sqrt(mags_abs_sd[irange]**2 + BCs_sd[irange]**2)



# Calculate Lbol and Lbol_sd ==================================================
irange = np.where(np.isfinite(Mbol))[0]
Lbol[irange] = LBOL_SUN * 10**((MBOL_SUN - Mbol[irange]) / 2.5)
Lbol_sd[irange] = Lbol[irange] * np.log(10.) * Mbol_sd[irange]

outdata[:,1] = temps
outdata[:,2] = temps_sd
outdata[:,3] = Mbol
outdata[:,4] = Mbol_sd
outdata[:,5] = Lbol
outdata[:,6] = Lbol_sd



# Save file -------------------------------------------------------------------
fmts = ['%s'] + (['%.6f'] * 11)
np.savetxt(FOLDER_OUT_TBL + 'luminosities.txt', outdata, fmt=fmts)
