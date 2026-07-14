import numpy as np
import astropy.units as u
import astropy.constants as co
import extinction


def mask_image(image, sigma=2,):
    """
    Mask out pixels in image with value in mask.
    """
    from kuaizi.detection import vanilla_detection
    from astropy.convolution import convolve, Gaussian2DKernel
    _, segmap = vanilla_detection(image, sigma=sigma, conv_radius=1, convolve=True, show_fig=False, verbose=False)
    cenid = segmap[segmap.shape[0] // 2, segmap.shape[1] // 2] - 1
    segmap[segmap == (cenid + 1)] = 0.0

    smooth_radius = 3
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    mask = (mask_conv >= gaussian_threshold)
    return mask

def SBP_star(model_dict, hsc_data, ind=0):
    from kuaizi.tractor.utils import HiddenPrints
    from photutils.isophote import Ellipse, EllipseGeometry

    gal = model_dict['i'].catalog[model_dict['i'].target_ind]
    geometry = EllipseGeometry(x0=gal.pos.x, 
                               y0=gal.pos.y, 
                               sma=1, eps=0,
                               pa=0)
    channels = list('grizy') + ['N708', 'N540']
    with HiddenPrints():
        model_img = np.asarray([
            model_dict[key].getModelImage(
                0, srcs=model_dict[key].catalog[model_dict[key].target_ind:model_dict[key].target_ind+1]
                                         ) for key in channels])
    
    ellipse = Ellipse(model_img[ind], geometry)
    try:
        isolist_model = ellipse.fit_image(
    #                                  sma0=10, minsma=0.,
                                    nclip=3, integrmode='median',
                                    maxit=10, maxgerr=1.0,
                                    fix_center=True, fix_pa=True, fix_eps=True)
    except:
        pass
    try:
        isolist_model = ellipse.fit_image(fix_center=True, fix_pa=True, fix_eps=True)
    except:
        print('Failed to fit data')
        return 

    _mean_x = np.mean(isolist_model.x0[isolist_model.sma > 10])
    _mean_y = np.mean(isolist_model.y0[isolist_model.sma > 10])
    
    # Data
    mask = mask_image(hsc_data.images[ind])
    
    ellipse = Ellipse(hsc_data.images[ind] * (~mask), geometry)
    try:
        isolist_data = ellipse.fit_image(
    #                                  sma0=10, minsma=0.,
                                    nclip=3, integrmode='median',
                                    maxit=10, maxgerr=1.0,
                                    fix_center=False, fix_pa=True, fix_eps=True)
    except:
        pass
    try:
        isolist_data = ellipse.fit_image(fix_pa=True, fix_eps=True)
    except:
        print('Failed to fit data')
        isolist_data = None
    return (isolist_model, isolist_data)


# from ekfphys.observer
def flambda_to_fnu ( wv, flux ):
    '''
    nuF_nu = lF_l
    F_nu = l/nu F_l
         = l^2/c F_l
    '''
    if not hasattr ( wv, 'unit' ) or not hasattr ( flux, 'unit' ):
        raise TypeError ("Must supply astropy Quantities!")
    eunit = flux.unit * wv.unit
    return (wv**2/co.c * flux).to(eunit/u.Hz)

def fnu_to_flambda ( wvorfreq, flux ):
    '''
    lam F_lam = nu F_nu 
    F_lam = nu / lam F_nu 
          = nu^2 / c F_nu
    '''    
    if not hasattr ( wvorfreq, 'unit' ) or not hasattr ( flux, 'unit' ):
        raise TypeError ("Must supply astropy Quantities!") 
    if wvorfreq.unit.is_equivalent(u.m):
        freq = co.c/wvorfreq
    else:
        freq = wvorfreq
            
    eunit = flux.unit * freq.unit
    return (freq**2 / co.c * flux).to(eunit/u.AA)

def gecorrection(wave, AvorEBV, Rv=3.1, unit='AA', etype='AV', return_magcorr=False):
    """Calculate the Galactic extinction correction for a given wavelength and Av.

    Parameters:
    -----------
    wave : array_like
        Wavelength in units of `unit`. Must be a 1D array.
    Av : float
        V-band extinction in magnitudes.
    Rv : float, optional (default=3.1)
        Total-to-selective extinction ratio.
    unit : {'aa', 'micron', 'nm', 'um'}, optional (default='AA')
        Unit of the wavelength.
    return_magcorr : bool, optional (default=False)
        If True, return the extinction correction in magnitudes instead of flux ratio.

    Returns:
    --------
    corr : array_like
        Extinction correction factor for the given wavelength(s). If `return_magcorr` is True,
        then the result is the extinction correction in magnitudes.
    """
    if etype == 'AV':
        Av = AvorEBV
    elif etype == 'EBV':
        Av = Rv * AvorEBV
        
    if isinstance(wave, float) or isinstance(wave, int):
        wave = np.array([wave], dtype=float)        
    
    Alambda = extinction.ccm89(wave, Av, Rv, unit=unit.lower())
    if return_magcorr:
        return Alambda
    else:
        corr = 10.**(0.4*Alambda)
        return corr

def extinction_correction ( wavelength, av, u_av=None, RV=4.05, curve=None, return_magcorr=False ):

    if curve is None:
        curve = extinction.calzetti00
    if not hasattr(wavelength, '__len__'):
        wavelength = np.array([wavelength])

    alambda = curve(wavelength, av, RV)
    corr = 10.**(alambda/2.5)
    if u_av is not None:
        u_alambda = curve(wavelength, u_av, RV)
        u_corr = abs(0.4*np.log(10.)*corr) * u_alambda
    else:
        #u_alambda = None
        u_corr = None
    if return_magcorr:
        if u_av is None:
            return alambda
        else:
            return alambda, u_alambda
    
    return corr, u_corr

# from ekfstats
def poly2d ( x, y, coeffs, deg ):
    z_pred = np.zeros_like(x)
    tuples_list = [(i, j) for i in range(deg+1) for j in range(deg+1) if i + j <= deg]    
    
    expected_length = (deg+1)*(deg+2)//2
    assert len(tuples_list) == expected_length
    for idx,(i,j) in enumerate(tuples_list):
        #k = i+j        
        z_pred += coeffs[idx] * x**i * y**j
    return z_pred

def closedform_leastsq (x,y):
    flat = np.ones(x.shape[0])
    w = np.hstack([flat.reshape(-1,1),x.reshape(x.shape[0],-1)])
    leastsquares_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T,w)),w.T),y)
    #leastsquares_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T,w)),w.T),y)
    #offset = np.mean(y - (w*leastsquares_soln).flatten(),axis=0)
    return leastsquares_soln

def trapz ( y, x):    
    order = np.argsort(x)
    return np.trapz(y[order], x[order])

