import copy
import numpy as np
from astropy.modeling import models, fitting
#from astropy.io import fits
from photutils.psf.matching import create_matching_kernel, CosineBellWindow

def _pad_psf ( psf, desired_length=65 ):
    '''
    Zero-pad a PSF image in order to achieve uniform sizes
    '''
    padding = np.array([desired_length,desired_length])
    padding -= np.asarray(psf.shape)
    if not (padding % 2 == 0).all():
        raise ValueError ("Padding requires a non-integer pixel count!")
    padding = [ (pad//2,pad//2) for pad in padding]
    return np.pad ( psf, padding )



class BBMBImage ( object ):
    '''
    Broad Band Medium Band Image
    ----------------------------
    A class to handle multiband processing of HSC + Merian
    cutouts. 
    
    example usage (loading cutouts):
    > bbmb = scratch_class.BBMBImage ( )
    > images = <dictionary containing cutouts>
    > bands = <dictionary containing PSF cutouts>
    > for band in ['g','N540','r','N708','i','z','y']:
    >     bbmb.add_band ( band, images[band], psfs[band] )    
    '''    
    def __init__ ( self, galaxy_id=None, distance=None, resolution=0.168 ):
        '''
        # \\ TODO : account for DECam native res. medium-bands and HSC native res. 
        # \\        broadbands ? 
        '''
        self.arcsec_per_pix = resolution     
        self.image = {}
        self.psf = {}
        self.bands = []
        self.distance = distance
        self.galaxy_id = galaxy_id
        
    def add_band ( self, name, image, psf ):
        '''
        Add a band to the image stack. This assumes that the 
        cutouts are all on the same pixel grid.
        
        {args}
        name : (string) name of band to add
        image: (array-like) single-band cutout
        psf  : (array-like) PSF image corresponding to image
        '''
        self.image[name] = image
        self.psf[name] = psf
        self.bands.append(name)
        
    @property
    def pc_per_pix ( self, ):        
        # \\ convert resolution from arcsec/pix to pc/pix
        if self.distance is None:
            raise ValueError ( "No distance has been defined for this galaxy!" )
        return self.distance * 1e6 * (self.arcsec_per_pix / 3600. * np.pi / 180. ) 
        
    def measure_psfsizes ( self, psf=None, save=True ):
        '''
        Fit a 2D Moffat profile to each PSF cutout in order to determine the
        band to which we should use as our reference PSF (worst-seeing band).
        
        {args}
        psf  : (BBMBImage.psf-like) dictionary of PSF cutouts
        save : (bool, default=True) if True, saves the largest FWHM (in pixels)
               and corresponding band name as attributes
        '''
        if psf is None:
            psf = self.psf
        bands = list(psf.keys())            
                
        fit_p = fitting.LevMarLSQFitter()
        fwhm_a = np.zeros(len(bands))
        model_psf = copy.copy(psf)
        for ix,band in enumerate(bands):
            z = _pad_psf(psf[band])

            mid = np.array(z.shape)//2

            y,x = np.mgrid[:z.shape[0],:z.shape[1]]
            p_init = models.Moffat2D(x_0=mid[0], y_0=mid[1], amplitude=z[mid[0],mid[1]])
            p = fit_p(p_init, x, y, z)
            fwhm_a[ix] = p.fwhm
            model_psf[band] = p(x,y)
        if save:
            self.fwhm_to_match = fwhm_a.max()
            self.band_to_match = self.bands[np.argmax(fwhm_a)]
        return fwhm_a, model_psf

    def match_psfs ( self, matchindex=None, refband=None,
                     psf=None, verbose=True, cbell_alpha=1. ):
        '''
        # \\ TODO: make the window function flexible
        
        Using a cosine bell window function, match all PSFs to the 
        reference PSF. The image corresponding to the reference PSF is not
        changed.
        
        {args}
        matchindex : (int, optional) the index of BBMBImage.bands corresponding to the
                     reference band
        refband    : (str, optional) the reference band. Either matchindex or refband
                     must be defined
        psf        : (BBMBImage.psf-like, optional) cutout PSFs, will use self.psf if None
        verbose    : (bool, default=True)
        cbell_alpha: (float, default=1.) fraction of values in window function that are 
                     tapered
        '''
        if verbose:
            print ('[SEDMap] Matching PSFs')
        if psf is None:
            psf = self.psf
            
        w1 = CosineBellWindow ( alpha=cbell_alpha )

        matched_image = copy.copy ( self.image )
        matched_psf = copy.copy ( psf )
        if matchindex is not None:
            badband = self.bands[matchindex]
        elif refband is not None:
            badband = refband
            matchindex = self.bands.index(refband)
        else:
            raise ValueError ('Either matchindex or refband must be defined!')
        for idx in range(len(self.bands)):
            if idx == matchindex:
                continue
            cband = self.bands[idx]
            kernel = create_matching_kernel(_pad_psf(psf[cband]),
                                            _pad_psf(psf[badband]),
                                            window=w1)
            matched_image[cband] = ndimage.convolve(self.image[cband], kernel)
            matched_psf[cband] = ndimage.convolve(psf[cband], kernel)
        self.matched_image = matched_image
        self.matched_psf = matched_psf
        return matched_image, matched_psf