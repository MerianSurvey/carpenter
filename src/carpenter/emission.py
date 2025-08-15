import os
import time
import importlib.resources

import numpy as np
import pandas as pd
from scipy import interpolate

from astropy import constants as co
from astropy import cosmology
from astropy import table
from astropy import units as u
from astropy.io import fits

from ekfphys import observer
from ekfstats import fit, math

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

def load_transmission (fname=None, band=None):
    if fname is None:
        if band is None:
            band = 'n708'
        fname = importlib.resources.files("carpenter").joinpath(f"../../data/transmission_curves/mer_{band.lower()}.txt")
    transmission = table.Table.read(
        fname,
        comment='#',
        format='ascii.basic',
        names=['wv','transmission_lambda'],
        units=[u.AA,None]    
    )
    transmission['freq'] = (co.c/transmission['wv']).to(u.Hz)
    transmission = transmission[np.argsort(transmission['wv'])]
    transmission['transmission_nu'] = transmission['transmission_lambda']/transmission['freq']**2
    return transmission

def excess_to_lineflux ( flux_freq, band):
    wv_eff_mb  = {'n540':5400.,'n708':7080.}[band]
    transmission = load_transmission(None, band=band)
    #flux_freq = float(table.Table(emcat)['flux'])*conversion*u.nJy
    flux_wv = observer.fnu_to_flambda(wv_eff_mb * u.AA, flux_freq).to(u.erg/u.s/u.cm**2/u.AA)

    # Calculate filter transmission properties
    tc_integrated = math.trapz(
        transmission['transmission_lambda'], 
        transmission['wv'].value
    ) * transmission['wv'].unit
    trans_atline = np.interp(
        wv_eff_mb, 
        transmission['wv'], 
        transmission['transmission_lambda']
    )
    return (flux_wv * tc_integrated / trans_atline).to(u.erg/u.s/u.cm**2)

def estimate_av (merian, dirstem=None):
    """
    Estimate V-band attenuation (A_V) for a sample of galaxies using their broadband photometry.

    This function computes an empirical estimate of dust attenuation (A_V) and its uncertainty 
    for galaxies based on their rest-frame optical color (g−r) and absolute magnitude in the r-band (M_r). 
    It uses a polynomial fit calibrated on SAGAbg galaxies.

    Parameters
    ----------
    merian : pandas.DataFrame
        A DataFrame containing the required photometric measurements from the Merian survey:
        - 'i_cModelFlux_Merian': i-band flux for computing apparent i magnitude
        - 'g_gaap1p0Flux_aperCorr_Merian', 'r_gaap1p0Flux_aperCorr_Merian', 
          'i_gaap1p0Flux_aperCorr_Merian': GAaP aperture-corrected fluxes for color computation

    dirstem : pathlib.Path or str, optional
        Path to the directory containing the polynomial coefficients for the A_V fits.
        If None, defaults to the internal `pieridae/data/` resource.

    Returns
    -------
    av : numpy.ndarray
        Estimated V-band attenuation (A_V) for each galaxy.
        Values >4 are considered unphysical and set to NaN.

    u_av : numpy.ndarray
        Estimated uncertainty on A_V. Values corresponding to A_V > 4 are set to infinity.
    """
    
    if dirstem is None:        
        dirstem = importlib.resources.files("carpenter").joinpath(f"../../data/SAGA_dust_predictions/")
    mi = -2.5*np.log10(merian['i_cModelFlux_Merian']*1e-9/3631.)
    Mi = mi - cosmo.distmod(0.08).value
    ri = -2.5*np.log10(merian['r_gaap1p0Flux_aperCorr_Merian']/merian['i_gaap1p0Flux_aperCorr_Merian'])
    gi = -2.5*np.log10(merian['g_gaap1p0Flux_aperCorr_Merian']/merian['i_gaap1p0Flux_aperCorr_Merian'])
    gr = -2.5*np.log10(merian['g_gaap1p0Flux_aperCorr_Merian']/merian['r_gaap1p0Flux_aperCorr_Merian'])
    Mr = Mi + ri

    saga_coeffs = np.load(f'{dirstem}/SAGA_Mr_gr_to_AV.npy')
    saga_u_coeffs = np.load(f'{dirstem}/SAGA_Mr_gr_to_u_AV.npy')
    n = len(saga_coeffs)
    deg = int((-3 + np.sqrt ( 9 - 4*(2-2*n) )) // 2 )

    av = 10.**fit.poly2d(Mr, gr, saga_coeffs, deg )
    u_av = fit.poly2d(Mr, gr, saga_u_coeffs, deg)
    u_av[av>4] = np.inf
    av[av>4] = np.NaN
    
    return av, u_av


def correct_NIISII(z, mass, source = "DESI"):
    """
    Calculate correction factors for [NII] and [SII] line contamination in N708 filter.
    
    This function computes multiplicative correction factors to account for contamination
    from [NII]6548,6583 and [SII]6716,6731 emission lines in the N708 medium-band filter,
    which is primarily designed to measure H-alpha. The corrections are based on empirical
    relationships from Mintz+24 and depend on both redshift and stellar mass.
    
    Parameters
    ----------
    z : float or array-like
        Redshift(s) of the source(s)
    mass : float or array-like
        Log10 stellar mass(es) in solar masses, aperture-corrected
    
    Returns
    -------
    float or array-like
        Multiplicative correction factor(s). Values > 1 indicate that the observed
        N708 flux is higher than pure H-alpha due to line contamination.
        
    Notes
    -----
    - Correction factors are binned by stellar mass: low (< 9.2), mid (9.2-9.8), high (> 9.8)
    - Redshift dependence is piecewise: low (< 0.074), mid (0.074-0.083), high (> 0.083)
    - In the mid-redshift range, corrections vary linearly with redshift
    - Based on empirical fitting from Mintz+24
    - Only valid for N708 filter measurements
    
    Examples
    --------
    >>> # Single source
    >>> correction = correct_NIISII(0.08, 9.5)
    >>> 
    >>> # Multiple sources
    >>> corrections = correct_NIISII([0.07, 0.08, 0.09], [9.0, 9.5, 10.0])
    """
    
    # SDSS
    if source == "SDSS":
        c0 = (1.39, 1.22, -18.8)
        c1 = (1.77, 1.39, -41.29)
        c2 = (1.85, 1.48, -41.97)
    elif source == "DESI":
        c0 = (1.47, 1.06, -45.70)
        c1 = (1.77, 1.29, -53.29)
        c2 = (1.86, 1.40, -50.89)
    
    # Handle both scalar and array inputs
    if hasattr(mass, 'all'):  # Array input
        correction = np.zeros_like(mass)
        
        # Define mass bins
        lowmass = mass < 9.2
        midmass = (mass >= 9.2) & (mass < 9.8)
        highmass = (mass >= 9.8)
        
        # Define redshift bins
        lowz = z < 0.074
        midz = (z >= 0.074) & (z < 0.083)
        highz = z > 0.083
        
        # Low mass corrections
        correction[lowmass & lowz] = c0[0]
        correction[lowmass & midz] = c0[2] * (z[lowmass & midz] - 0.074) + c0[0]
        correction[lowmass & highz] = c0[1]

        # Mid mass corrections
        correction[midmass & lowz] = c1[0]
        correction[midmass & midz] = c1[2] * (z[midmass & midz] - 0.074) + c1[0]
        correction[midmass & highz] = c1[1]

        # High mass corrections
        correction[highmass & lowz] = c2[0]
        correction[highmass & midz] = c2[2] * (z[highmass & midz] - 0.074) + c2[0]
        correction[highmass & highz] = c2[1]

        return correction
        
    else:  # Scalar input
        # Low mass regime (log M* < 9.2)
        if mass < 9.2:
            if z < 0.074:
                return c0[0]
            elif z > 0.083:
                return c0[1]
            else:
                return c0[2] * (z - 0.074) + c0[0]

        # Mid mass regime (9.2 <= log M* < 9.8)
        elif mass < 9.8:
            if z < 0.074:
                return c1[0]
            elif z > 0.083:
                return c1[1]
            else:
                return c1[2] * (z - 0.074) + c1[0]

        # High mass regime (log M* >= 9.8)
        else:
            if z < 0.074:
                return c2[0]
            elif z > 0.083:
                return c2[1]
            else:
                return c2[2] * (z - 0.074) + c2[0]
            
def compute_galactic_extinction(ra, dec, verbose=1, dirstem=None,  rv=3.1,):
    if dirstem is None:        
        dirstem = importlib.resources.files("carpenter").joinpath(f"../../data/galactic_extinction_maps/")
    
    # Use precomputed A_V map for faster computation
    if verbose > 0:
        print('Loading precomputed Galactic extinction map...')
        
    # Load A_V map and coordinate grids
    avmap = np.load(f'{dirstem}/avmap.npz')['arr_0']
    ragrid = np.load(f'{dirstem}/avmap_ragrid.npz')['arr_0']
    decgrid = np.load(f'{dirstem}/avmap_decgrid.npz')['arr_0']
    
    # Handle RA coordinate wrapping by padding the map
    ra_padded = np.concatenate([ragrid - 360, ragrid, ragrid + 360])
    avmap_padded = np.hstack([avmap, avmap, avmap])
    
    # Create interpolation function
    ifn = interpolate.RegularGridInterpolator([ra_padded, decgrid], avmap_padded.T)
    
    # Extract coordinates and interpolate A_V values
    #mra = mcat[rakey]
    #mdec = mcat[deckey]
    mcoords = np.stack([ra, dec]).T
    direct_geav = ifn(mcoords)
    
    if verbose > 0:
        print('Interpolated Galactic extinction values from map.')

    # Apply Galactic extinction correction to photometry
    wv_eff = np.array([1548.85, 2303.37, 7080., 5400.])
    ge_arr = np.zeros([len(ra),wv_eff.size])
    for idx,c_av in enumerate(direct_geav):
        #wv_eff[2] = wv_eff.value * (1. + z)
        ge_arr[idx] = observer.gecorrection ( wv_eff, c_av, rv, return_magcorr=False)
    return ge_arr
            
def compute_emissioncorrections(
    mcat,
    use_dustengine=False,
    load_from_pickle=False,
    verbose=1,
    zphot='z500',
    rakey='RA',
    deckey='DEC',
    estimated_av=None,
    u_estimated_av=None,
    logmstar_key='logmass_gaap',
    dirstem=None
):
    """
    Compute various corrections for H-alpha flux measurements.
    
    This function calculates aperture corrections, emission line contamination
    corrections, Galactic extinction corrections, and internal dust corrections
    for H-alpha observations.
    
    Parameters
    ----------
    mcat : pandas.DataFrame
        Catalog containing photometric and derived quantities
    use_dustengine : bool, optional
        Use dust engine for Galactic extinction instead of precomputed map (default: False)
    load_from_pickle : bool, optional
        Load dust engine from pickle file (default: False)
    verbose : int, optional
        Verbosity level (default: 1)
    zphot : str or float, optional
        Photometric redshift column name or constant value (default: 'z500')
    rakey : str, optional
        Right ascension column name (default: 'RA')
    deckey : str, optional
        Declination column name (default: 'DEC')
    estimated_av : array-like, optional
        External estimate of A_V values (default: None, uses mcat['AV'])
    u_estimated_av : array-like, optional
        Uncertainties in estimated A_V values (default: None)
    logmstar_key : str, optional
        Stellar mass column name (default: 'logmass_gaap')
    dirstem : str, optional
        Directory path for input data files (default: DEFAULT_MERIAN_DIR + '/local_data/inputs/')
    
    Returns
    -------
    tuple
        (emission_correction, ge_correction, dust_correction, aperture_correction)
        - emission_correction: Correction for [NII] and [SII] line contamination
        - ge_correction: Galactic extinction correction factors
        - dust_correction: Internal dust extinction corrections
        - aperture_correction: Aperture correction factors
    
    Notes
    -----
    - Emission line corrections use Mintz+24 formalism
    - Galactic extinction can use either dust engine or precomputed A_V map
    - Internal dust corrections assume R_V = 4.05
    """
    
    if dirstem is None:
        dirstem = importlib.resources.files("carpenter").joinpath(f"../../data/galactic_extinction_maps/")
        #dirstem = DEFAULT_MERIAN_DIR + '/local_data/inputs/'
    
    if verbose > 0:
        start = time.time()
    
    # Calculate aperture correction from i-band model to 1" aperture flux ratio
    aperture_correction = mcat['i_cModelFlux_Merian'] / mcat['i_gaap1p0Flux_Merian']
    
    if verbose > 0:
        print(f'Computed aperture correction in {time.time() - start:.1f} seconds.')
        start = time.time()
    
    # Correct for contamination from other emission lines using Mintz+24 formalism
    if isinstance(zphot, float):
        redshift_values = zphot
    else:
        redshift_values = mcat[zphot]
    
    # Apply aperture correction to stellar mass for line contamination estimate
    if 'gaap' in logmstar_key:
        aperture_corrected_logmass = mcat[logmstar_key] + np.log10(aperture_correction)
    else:
        aperture_corrected_logmass = mcat[logmstar_key]
    emission_correction = correct_NIISII(redshift_values, aperture_corrected_logmass)**-1
    
    if verbose > 0:
        print(f'Computed line contamination correction in {time.time() - start:.1f} seconds.')
        start = time.time()

    # Apply Galactic extinction correction to photometry
    #ge_correction = photometry.uvopt_gecorrection(mcat, av=direct_geav)
    ge_correction = compute_galactic_extinction (mcat['RA'], mcat['DEC'])
    
    if verbose > 0:
        print(f'Computed Galactic extinction correction in {time.time() - start:.1f} seconds.')
        start = time.time()
    
    # Internal dust extinction correction
    # Rest wavelengths for different lines/bands (divided by 1.08 for some reason)
    restwl = np.array([1548.85, 2303.37, 7080., 5400.]) / 1.08
    dust_correction = np.zeros((2, len(emission_correction), len(restwl)))
    
    # Use provided A_V estimates or catalog values
    if estimated_av is None:
        avbase = mcat['AV']
        u_avbase = None
    else:
        avbase = estimated_av
        u_avbase = u_estimated_av
    
    # Calculate extinction corrections for each source
    for idx, av in enumerate(avbase):
        if u_avbase is not None:
            u_av = u_avbase[idx]
        else:
            u_av = None
            
        # Handle masked values
        if hasattr(av, 'mask') and av.mask:
            dust_correction[:, idx] = np.NaN
        else:
            # Calculate extinction correction using R_V = 4.05
            dc = observer.extinction_correction(restwl, av, u_av=u_av, RV=4.05)
            dust_correction[0, idx] = dc[0].data
            if dc[1] is not None:
                dust_correction[1, idx] = dc[1]
    
    if verbose > 0:
        print(f'Computed internal extinction corrections in {time.time() - start:.1f} seconds.')
    
    return emission_correction, ge_correction, dust_correction, aperture_correction

def mbestimate_emission_line(
        mb_data, 
        gdata,
        rdata, 
        idata,
        zdata, 
        redshift,        
        u_mb_data,
        u_rdata=0.,
        u_idata=0.,        
        do_aperturecorrection=True, 
        do_extinctioncorrection=True,
        do_gecorrection=True, 
        do_linecorrection=True,
        apercorr=1.,
        ex_correction=1.,
        u_ex_correction=0.,
        ge_correction=1.,
        ns_correction=1.,
        specflux_unit=None,
        filter_curve_file=None,
        zp=31.4,
        band='n708',
        ctype='powerlaw',
        plawbands='griz',
        return_continuum_at_line_center=False

    ):
    """
    Estimate emission line flux from medium-band photometry.
    
    This function estimates emission line fluxes (H-alpha for N708, [OIII]5007 for N540)
    by subtracting continuum estimated from broadband photometry.
    
    Parameters
    ----------
    mb_data : array-like
        Medium-band photometry data (magnitudes)
    gdata, rdata, idata, zdata : array-like
        Broadband photometry data in g, r, i, z bands (magnitudes)
    redshift : array-like
        Redshift of sources
    u_mb_data : array-like
        Uncertainty in medium-band photometry
    u_rdata, u_idata : array-like, optional
        Uncertainties in r and i band photometry
    do_aperturecorrection : bool, optional
        Apply aperture correction (default: True)
    do_extinctioncorrection : bool, optional
        Apply internal extinction correction (default: True)
    do_gecorrection : bool, optional
        Apply Galactic extinction correction (default: True)
    do_linecorrection : bool, optional
        Apply line corrections for contaminating lines (default: True)
    apercorr : float, optional
        Aperture correction factor (default: 1.0)
    ex_correction : float, optional
        Internal extinction correction factor (default: 1.0)
    u_ex_correction : float, optional
        Uncertainty in extinction correction (default: 0.0)
    ge_correction : float, optional
        Galactic extinction correction factor (default: 1.0)
    ns_correction : float, optional
        Correction for [NII] and [SII] contamination (N708 only) (default: 1.0)
    specflux_unit : astropy.units.Unit, optional
        Unit for spectral flux density (default: computed from zp)
    filter_curve_file : str, optional
        Path to filter transmission curve file
    zp : float, optional
        Photometric zero point (default: 31.4)
    band : str, optional
        Medium-band filter name ('n708' or 'n540') (default: 'n708')
    ctype : str, optional
        Continuum estimation method ('powerlaw', 'linear', 'cubic_spline', 'ri_avg')
        (default: 'powerlaw')
    plawbands : str, optional
        Bands to use for powerlaw/linear fitting (default: 'griz')
    
    Returns
    -------
    tuple
        (line_flux, line_luminosity, equivalent_width, continuum_flux)
        Each element is a tuple of (value, uncertainty)
    
    Notes
    -----
    - Line corrections (ns_correction) are only valid for N708 band
    - Continuum estimation via powerlaw method is recommended for most applications
    - Distance uncertainty calculation is hardcoded and needs generalization
    """
    
    # Set up spectral flux unit if not provided
    if specflux_unit is None:
        # Convert from AB magnitude zero point to flux density
        # -2.5 log10(X/3631 Jy) = zp
        # X = 10^(zp/-2.5) * 3631 Jy
        specflux_unit = 10.**(zp/-2.5) * 3631. * u.Jy
    
    # Load filter transmission curve and get band properties
    transmission = load_transmission(filter_curve_file, band=band)
    wv_eff_mb = {'n708': 7080., 'n540': 5400.}[band]
    wv_rest_mb = {'n708': 6563., 'n540': 5007.}[band]
    line_restwl = wv_rest_mb * u.AA   
    
    # Estimate continuum flux at medium-band wavelength
    if ctype == 'ri_avg':
        # Simple average of r and i bands
        bandspecflux_continuum = (rdata + idata) / 2. * specflux_unit
        
    elif ctype == 'cubic_spline':
        # Cubic spline interpolation through r, i, z bands
        wv_eff = np.array([6229., 7703., 8906.])  # r, i, z effective wavelengths
        hscphot = np.array([rdata, idata, zdata]).T
        fy = interpolate.CubicSpline(wv_eff, hscphot, axis=1)
        bandspecflux_continuum = fy(wv_eff_mb) * specflux_unit
        
    elif ctype == 'linear':
        # Linear fit in wavelength space
        wdict = {'g': 4809., 'r': 6229., 'i': 7703., 'z': 8906.}
        fdict = {'g': gdata, 'r': rdata, 'i': idata, 'z': zdata}
        wv_eff = np.array([wdict[band] for band in plawbands])
        lsq_x = wv_eff
        lsq_y = np.array([fdict[band] for band in plawbands])
        lsq_coeffs = fit.closedform_leastsq(lsq_x, lsq_y)
        bandspecflux_continuum = (lsq_coeffs[0] + lsq_coeffs[1] * wv_eff_mb).flatten() * specflux_unit
        
    elif ctype == 'powerlaw':
        # Power law fit in log space (recommended method)
        # Uses vectorized least squares: x = (A^T A)^-1 A^T b
        wdict = {'g': 4809., 'r': 6229., 'i': 7703., 'z': 8906.}
        fdict = {'g': gdata, 'r': rdata, 'i': idata, 'z': zdata}
        wv_eff = np.array([wdict[band] for band in plawbands])
        lsq_x = np.log10(wv_eff)
        lsq_y = np.log10(np.array([fdict[band] for band in plawbands]))
        lsq_coeffs = fit.closedform_leastsq(lsq_x, lsq_y)
        
        # Continuum flux at line center 
        line_center_continuum = 10.**(lsq_coeffs[0] + lsq_coeffs[1] * np.log10(wv_rest_mb * (1+redshift))).flatten() * specflux_unit

        # Calculate total continuum contribution through medium band
        lsq_coeffs = lsq_coeffs.reshape((2, -1)) # reshape for efficiency
        continuum = 10.**(lsq_coeffs[0][:, np.newaxis] + lsq_coeffs[1][:, np.newaxis] * np.log10(transmission['wv']))* specflux_unit
        filt_norm = np.trapz(transmission['transmission_lambda'] / transmission['wv'], transmission['wv']) 
        bandspecflux_continuum = np.trapz(
                                        transmission['transmission_lambda'] * continuum / transmission['wv'], 
                                        transmission['wv'], axis=1) / filt_norm
        
    # Subtract continuum from medium-band flux to get line flux
    bandspecflux_line = mb_data * specflux_unit - bandspecflux_continuum
    
    # Propagate uncertainties (0.25 factors for r,i bands) # XXX: verify these factors
    u_bandspecflux_line = np.sqrt(
        (u_mb_data * specflux_unit)**2 + 
        0.25 * (u_rdata * specflux_unit)**2 + 
        0.25 * (u_idata * specflux_unit)**2
    )
    
    # Convert to spectral flux density (f_lambda)
    bsf_lambda = observer.fnu_to_flambda(wv_eff_mb * u.AA, bandspecflux_line)
    u_bsf_lambda = observer.fnu_to_flambda(wv_eff_mb * u.AA, u_bandspecflux_line)
    
    # Calculate filter transmission properties
    tc_integrated = math.trapz(
        transmission['transmission_lambda'], 
        transmission['wv'].value
    ) * transmission['wv'].unit
    trans_atline = np.interp(
        line_restwl * (1. + redshift), 
        transmission['wv'], 
        transmission['transmission_lambda']
    )
    
    # Calculate photon energy at observed wavelength
    line_energy = (co.h * co.c / (line_restwl * (1. + redshift))).to(u.erg)
    
    # Convert to line flux
    line_flux = (bsf_lambda * tc_integrated / trans_atline).to(u.erg/u.s/u.cm**2)
    u_line_flux = (u_bsf_lambda * tc_integrated / trans_atline).to(u.erg/u.s/u.cm**2)
    
    # Store pre-correction flux for equivalent width calculation
    line_flux_forew = line_flux.copy()
    u_line_flux_forew = u_line_flux.copy()
    
    # Apply aperture correction (1" aperture -> total flux)
    # Approximated from i_cmodel / i_gaap1p0
    if do_aperturecorrection:
        line_flux *= apercorr
        u_line_flux *= apercorr
        fcontinuum_ac = bandspecflux_continuum * apercorr
    else:
        fcontinuum_ac = bandspecflux_continuum.copy()
        
    # Apply internal extinction correction
    if do_extinctioncorrection:
        # Propagate uncertainties: sigma^2(c*F) = c^2*sigma^2(F) + F^2*sigma^2(c)
        u_line_flux = np.sqrt(
            (ex_correction * u_line_flux)**2 + 
            (u_ex_correction * line_flux)**2
        )
        line_flux *= ex_correction
        fcontinuum_ac *= ex_correction
    
    # Apply Galactic extinction correction
    if do_gecorrection:
        line_flux *= ge_correction
        u_line_flux *= ge_correction
        fcontinuum_ac *= ge_correction

    # Apply line corrections for contaminating lines
    if do_linecorrection:
        if band != 'n708':
            print(f"WARNING: Line corrections (ns_correction) are only valid for N708 band, not {band}")
        
        line_flux *= ns_correction
        u_line_flux *= ns_correction
        line_flux_forew *= ns_correction
        u_line_flux_forew *= ns_correction
    
    # Calculate equivalent width using pre-extinction correction fluxes
    wv_eff = wv_eff_mb * u.AA
    bandspecflux_continuum_wl = co.c * bandspecflux_continuum / wv_eff**2
    line_ew = (line_flux_forew / bandspecflux_continuum_wl).to(u.AA)
    u_line_ew = (u_line_flux_forew / bandspecflux_continuum_wl).to(u.AA)
    
    # Calculate luminosity
    dlum = cosmo.luminosity_distance(redshift).to(u.cm)
    # XXX: Need to generalize this distance uncertainty calculation
    u_dlum = ((cosmo.luminosity_distance(0.09) - cosmo.luminosity_distance(0.07)) / 2.).to(u.cm)
    
    distance_factor = 4. * np.pi * dlum**2
    u_distance_factor = 8. * np.pi * dlum * u_dlum
    
    line_luminosity = line_flux * distance_factor
    u_line_luminosity = np.sqrt(
        (u_line_flux * distance_factor)**2 + 
        (line_flux * u_distance_factor)**2
    )
    
    if return_continuum_at_line_center:
            return (
            (line_flux, u_line_flux), 
            (line_luminosity, u_line_luminosity), 
            (line_ew, u_line_ew), 
            fcontinuum_ac.to(u.nJy),
            line_center_continuum)

    return (
        (line_flux, u_line_flux), 
        (line_luminosity, u_line_luminosity), 
        (line_ew, u_line_ew), 
        fcontinuum_ac.to(u.nJy)
    )
    
def compute_galexluminosities ( galex, redshifts, ge_arr=None, dust_corr=None ):
    gdf = pd.DataFrame(index=galex.index)
    
    if not isinstance(galex, table.Table):
        galex = table.Table.from_pandas(galex.reset_index())
        galex.add_index('index')
    
    #uv_color = galex['fuv_mag'] - galex['nuv_mag']
    for idx,band in enumerate(['fuv','nuv']):
        uvflux = galex[f'{band}_flux'] * u.nJy
        u_uvflux = galex[f'u_{band}_flux'] * u.nJy
        uvflux = uvflux.to(u.erg/u.s/u.cm**2/u.Hz)
        if dust_corr is not None:
            uvflux *= dust_corr[0,:,idx] # \\ internal extinction corrections
            u_uvflux = np.sqrt( (dust_corr[0,:,idx]*u_uvflux)**2 + (dust_corr[1,:,idx]*uvflux)**2 )
        if ge_arr is not None:
            uvflux *= ge_arr[:,idx] # \\ galactic extinction correction
            u_uvflux *= ge_arr[:,idx]
        else:
            print('Not doing GE Correction!')
        # \\ ignoring k-correction because it should be 0.01-0.05 mag in this redshift range
        #uvflux *= observer.calc_kcor(band.upper(), redshifts, 'FUV - NUV', uv_color )
        uvlum = (uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz)   
        u_uvlum = (u_uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz) 
        
        gdf[f'{band}_flux_corrected']  = uvflux.value
        gdf[f'u_{band}_flux_corrected']  = u_uvflux.value
        
        gdf[f'L{band.upper()}'] = uvlum.value        
        gdf[f'u_L{band.upper()}'] = u_uvlum.value
    return gdf
    
def make_mb_catalog (ms, av, u_av):
    emission_corrections = compute_emissioncorrections(ms, load_from_pickle=False, estimated_av=av, u_estimated_av=u_av)
    niisii_correction, ge_correction, extinction_correction, aperture_correction = emission_corrections
        
    redshifts = np.where(np.isnan(ms['z_spec']), 0.08, ms['z_spec'])
    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        ms[utils.photcols['N708']].values,
        ms[utils.photcols['g']].values,
        ms[utils.photcols['r']].values,
        ms[utils.photcols['i']].values,
        ms[utils.photcols['z']].values,
        redshifts,
        ms[utils.u_photcols['N708']].values,
        0.,
        0.,
        band='n708',
        apercorr=aperture_correction.values,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[0,:,2],
        u_ex_correction = 0.*extinction_correction[1,:,2], # \\ we're actually going to do dust errors downstream
        ns_correction=niisii_correction,
        do_aperturecorrection=True,
        do_gecorrection=True,
        do_extinctioncorrection=True,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='powerlaw',
        plawbands='riz',
    ) 
      
      
    n540_fluxes, n540_luminosities, n540_eqws, n540_fcont = photometry.mbestimate_halpha(
        ms[utils.photcols['N540']].values,
        ms[utils.photcols['g']].values,
        ms[utils.photcols['r']].values,
        ms[utils.photcols['i']].values,
        ms[utils.photcols['z']].values,
        redshifts,
        ms[utils.u_photcols['N540']].values,
        0.,
        0.,
        band='n540',
        apercorr=aperture_correction.values,
        ge_correction=ge_correction[:,3],
        ex_correction=extinction_correction[0,:,3],
        u_ex_correction = np.zeros(aperture_correction.shape), # \\ we're actually going to do dust errors downstream
        ns_correction=None,
        do_aperturecorrection=True,
        do_gecorrection=True,
        do_extinctioncorrection=True,
        do_linecorrection=False,
        specflux_unit=u.nJy,
        ctype='linear',
        plawbands='gr',
        #continuum_adjust=-0.6
    ) 
          
    exc = extinction_correction.copy()
    exc[1] = 0. # XXX AV is no longer a RV for Galex correction because it is set by AV above
    lumdf = compute_galexluminosities(ms, redshifts, ge_arr=ge_correction, dust_corr=exc)
    lumdf['halpha_flux_corrected'] = n708_fluxes[0].value
    lumdf['u_halpha_flux_corrected'] = n708_fluxes[1].value
    lumdf['LHa'] = n708_luminosities[0].value
    lumdf['u_LHa'] = n708_luminosities[1].value
    lumdf['EWHa'] = n708_eqws[0].value
    lumdf['u_EWHa'] = n708_eqws[1].value
    
    lumdf['oiii_flux_corrected'] = n540_fluxes[0].value
    lumdf['u_oiii_flux_corrected'] = n540_fluxes[1].value
    lumdf['LOIII'] = n540_luminosities[0].value
    lumdf['u_LOIII'] = n540_luminosities[1].value
    lumdf['EWOIII'] = n540_eqws[0].value
    lumdf['u_EWOIII'] = n540_eqws[1].value 
    lumdf['fc_n540']   = n540_fcont.value
    lumdf['fc_n708']   = n708_fcont.value
    
    return lumdf