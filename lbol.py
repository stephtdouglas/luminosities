"""Calculate bolometric luminosities."""

import numpy as np
import matplotlib.pyplot as plt

from constants import *
import read_files

def calc_abs_mags(app_mags,unc_mags,distance,unc_distance,extinction,
    null_mag=-9999.):
    """Calculate absolute magnitues. 

    All inputs (except null_mag) can be arrays.
    """
    
    # Only do the calculation where the magnitudes are valid
    bad_app = np.where(app_mags==null_mag)[0]
    good_app = np.where(app_mags!=null_mag)[0]
    
    # Calculate absolute mags
    abs_mags = ((app_mags - extinction) - 5.0 * (np.log10(distance) - 1.0))
    abs_mags[bad_app] = null_mag

    # Calculate uncertainty on abs_mags (???)
    arrDfac = 5 * np.log10(1 + 10**(np.log10(unc_distance) - np.log10(distance)))
    unc_abs_mags = np.sqrt(unc_mags**2 + arrDfac**2)
    unc_abs_mags[bad_app] = null_mag

    return abs_mags, unc_abs_mags

def calc_bol_mags(abs_mags,unc_mags,bol_corr,unc_bol_corr,null_mag=-9999.):
    """Calculate bolometric luminosities."""
    
    mbol = abs_mags + bol_corr
    mbol[abs_mags==null_mag] = null_mag

    unc_mbol = np.sqrt(unc_mags**2 + unc_bol_corr**2)
    unc_mbol[abs_mags==null_mag] = null_mag

    return mbol, unc_mbol

def calc_lbol(mbol,unc_mbol,mbol_sun=MBOL_SUN,lbol_sun=LBOL_SUN,null_mag=-9999.):
    """Calculate bolometric magnitudes using solar values as reference."""

    lbol = lbol_sun * 10**((mbol_sun - mbol) / 2.5)
    lbol[(mbol==null_mag) | (np.isinf(mbol))] = null_mag

    unc_lbol = lbol * np.log(10.) * unc_mbol
    unc_lbol[(mbol==null_mag) | (np.isinf(mbol))] = null_mag

    return lbol, unc_lbol

def calc_teff(abs_mags,unc_abs_mags,which_mag="r",null_mag=-9999.):
    """Convert abs mag (Mg,Mr,Mi) to T and calculate uncertainty on T."""
    
    mags, funcs, magranges, slopes = read_files.read_SEDs()

    # Find the range where the interpolation is valid
    interp_range = np.where((abs_mags >= magranges[which_mag][0]) & 
                            (abs_mags <= magranges[which_mag][1]))[0]

    # Interpolate where magnitudes are in the right range
    teff = np.ones(len(abs_mags))*null_mag
    interp_function = funcs[which_mag]
    teff[interp_range] = interp_function(abs_mags[interp_range])

    # Calculate uncertainty on Teff
    # By binning abs_mags using Adam's table as bins and using the interp slopes
    # (y=mx+b, so sigma_y^2 = dy/dx ^2 * sigma_x^2 = m^2 * sigma_x^2)
    interp_slopes = slopes[which_mag]
    digitized_mags = np.digitize(abs_mags,mags[which_mag][1:])
    unc_teff = np.ones(len(abs_mags))*null_mag
    for i,slp in enumerate(interp_slopes):
        ibin = np.where(digitized_mags == i)[0]
        unc_teff[ibin] = slp * unc_abs_mags[ibin]
    unc_teff[abs(abs_mags-null_mag)<1e-4] = null_mag
    unc_teff[abs(teff-null_mag)<1e-4] = null_mag

    return teff, unc_teff

def calc_bc(teff,unc_teff,which_mag="r",null_teff=-9999.):
    """Convert T to bolometric correction in (g,r,i) with uncertainty."""
    
    funcs, teffrange, slopes, teff_bins = read_files.read_BCs()

    # Find the range where the interpolation is valid
    interp_range = np.where((teff >= teffrange[0]) & 
                            (teff <= teffrange[1]))[0]

    # Interpolate where Teffs are in the right range
    bol_corr = np.ones(len(teff))*null_teff
    interp_function = funcs[which_mag]
    bol_corr[interp_range] = interp_function(teff[interp_range])

    # Calculate uncertainty on the BCs
    # By binning Teffs using the Girardi table as bins and the interp slopes
    interp_slopes = slopes[which_mag]
    digitized_teffs = np.digitize(teff[interp_range], teff_bins[1:])
    unc_bol_corr = np.ones(len(teff))*null_teff
    for i,slp in enumerate(interp_slopes):
        ibin = np.where(digitized_teffs==i)[0]
        unc_bol_corr[interp_range][ibin] = slp * unc_teff[interp_range][ibin]
    unc_bol_corr[teff==null_teff] = null_teff

    return bol_corr, unc_bol_corr

def lbol_wrapper(app_mags,unc_mags,distance,unc_distance,extinction,
    which_mag="r",null_val=9999.):
    """Compute bolometric corrections using other functions from lbol.py."""

    abs_mags, unc_abs_mags = calc_abs_mags(app_mags,unc_mags,distance,
                                           unc_distance,extinction,
                                           null_mag=null_val)

    teff, unc_teff = calc_teff(abs_mags,unc_abs_mags,which_mag=which_mag,
                               null_mag=null_val)

    bol_corr, unc_bol_corr = calc_bc(teff,unc_teff,which_mag=which_mag,
                                     null_teff=null_val)

    mbol, unc_mbol = calc_bol_mags(abs_mags,unc_abs_mags,bol_corr,
                                   unc_bol_corr,null_mag=null_val)

    lbol, unc_lbol = calc_lbol(mbol,unc_mbol,null_mag=null_val)

    return lbol, unc_lbol

