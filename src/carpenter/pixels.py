import copy
import numpy as np
from scipy import ndimage
from astropy.modeling import models, fitting
#from astropy.io import fits
import sep
from photutils.psf.matching import create_matching_kernel, CosineBellWindow

average_bb = {'N708':('r','i'), 'N540':('g','r')}
single_bb = {'N708':'i', 'N540':'r'}

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
        self.var = {}
        self.psf = {}
        self.bands = []
        self.distance = distance
        self.galaxy_id = galaxy_id
                
        
    def add_band ( self, name, image, var=None, psf=None ):
        '''
        Add a band to the image stack. This assumes that the 
        cutouts are all on the same pixel grid.
        
        {args}
        name : (string) name of band to add
        image: (array-like) single-band cutout
        psf  : (array-like) PSF image corresponding to image
        '''
        self.image[name] = image
        self.var[name] = var
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
        matched_var = copy.copy ( self.var )
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
            
            # \\ Propagate error through convolution following Klein+21
            # \\ ignoring initial pixel correlations
            # \\ ( https://iopscience.iop.org/article/10.3847/2515-5172/abe8df )
            matched_var[cband] = ndimage.convolve(self.var[cband], kernel**2)
            
        self.matched_image = matched_image
        self.matched_psf = matched_psf
        self.matched_var = matched_var
        return matched_image, matched_psf
    
    def compute_mbexcess ( self, band, method='average', psf_matched=True):
        '''
        Compute per-pixel medium-band excess over an estimate of the
        continuum
        '''
        if psf_matched:
            img_d = self.matched_image
            var_d = self.matched_var
        else:
            img_d = self.image
            var_d = self.var
        
        # \\ get MB image
        mbimg = img_d[band]
        v_mbimg = var_d[band]
        
        # \\ compute our continuum estimate:
        if method == 'average':
            bb0,bb1 = average_bb[band]
            bb_blue = img_d[bb0]
            bb_red = img_d[bb1]            
            v_bb_blue = var_d[bb0]
            v_bb_red = var_d[bb1]
            
            continuum = 0.5 * ( bb_blue + bb_red )
            v_continuum = 0.25 * (v_bb_blue + v_bb_red)
        elif method == 'single':
            continuum = img_d[single_bb[band]]
            v_continuum = img_d[single_bb[band]] 
        
        excess = mbimg - continuum
        v_excess = v_mbimg + v_continuum
        return excess, v_excess
    
    def define_autoaper ( self, excess, v_excess, clobber=False, ellipsify=True, thresh=5. ):
        if hasattr(self, 'regionauto') and not clobber:
            return self.regionauto        
            
        catalog, segmap = sep.extract ( excess,
                                        var = v_excess,
                                        thresh=thresh,
                                        deblend_cont=0.05,
                                        segmentation_map=True )
        size = segmap.shape[0]//2
        regionauto = (segmap == segmap[size,size])
    
        
        if ellipsify:
            Y,X = np.mgrid[:regionauto.shape[0],:regionauto.shape[1]]
            if segmap[size,size] > 0:
                cid = segmap[size,size] - 1
            else: # \\ if dead center is a non-detection (e.g. masked)
                R = np.sqrt ( (Y - size)**2 + (X-size)**2 )                
                Rseg = np.where ( segmap > 0, R, np.inf )
                Rmin = np.unravel_index(np.argmin (Rseg), segmap.shape )

                cid = segmap[Rmin] - 1
                #cid = segmap[segmap>0][np.argmin(R[segmap > 0])]
                
            yoff = Y-catalog['y'][cid]
            xoff = X-catalog['x'][cid]
            #theta = catalog['theta'][cid]
            ep = catalog['cyy'][cid]*yoff**2 + catalog['cxx'][cid]*xoff**2 + catalog['cxy'][cid]*xoff*yoff            
            regionauto = ep < 9. # this value comes from SExtractor manual :shrug: 
        
        #self.regionauto = regionauto
        #self.autocat = catalog[[cid]]
        #self.segmap = segmap
        return regionauto, catalog[[cid]]