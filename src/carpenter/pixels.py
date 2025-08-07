import copy
import numpy as np
from scipy import ndimage
from astropy import coordinates
from astropy.modeling import models, fitting
from astropy.io import fits
import sep
from photutils.psf.matching import create_matching_kernel, HanningWindow
from astrocut import FITSCutout

average_bb = {'n708':'riz', 'n540':'gr'}
single_bb = {'n708':'i', 'n540':'r'}

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
        self.hdu = {}    
        self.image = {}
        self.var = {}
        self.psf = {}
        self.bands = []
        self.distance = distance
        self.galaxy_id = galaxy_id          
        
    def add_band(self, name, center, size, image, var=None, psf=None, imslice=None, image_ext='IMAGE', var_ext='VARIANCE', psf_ext=0):
        """
        Add a new band to the image stack.

        This method assumes that all input cutouts are on the same pixel grid.
        It extracts a square cutout of shape (size, size) centered at `center`
        from the input `image`, optional `var` (variance), and stores them along
        with the optional `psf` under the given `name`.

        Parameters
        ----------
        name : str
            Name of the band to add.
        center : tuple
            (ra, dec) coordinates for the center of the cutout.
        size : int
            Half-size of the cutout. The output will be (2*size, 2*size) pixels.
        image : array-like
            Image data for the band.
        var : array-like, optional
            Variance image corresponding to the band.
        psf : array-like, optional
            PSF image corresponding to the band.
        imslice : slice or tuple of slices, optional
            Slice object(s) to apply to the cutout. Defaults to full slice.

        Notes
        -----
        The extracted image and variance arrays are byte-swapped and set to native byte order.
        The band name is appended to `self.bands`, and all relevant metadata is stored
        in internal dictionaries: `self.image`, `self.var`, `self.psf`, and `self.hdu`.
        """
        if not isinstance(center, coordinates.SkyCoord):
            center = coordinates.SkyCoord(*center, unit='deg')
        if imslice is None:
            imslice = slice(None)
        # cc = cutouts._hducut ( x[1], center, [half_size, half_size] )
        imhdu = FITSCutout ( image, center, [size,size], extension=image_ext ).fits_cutouts[0][1]
        self.hdu[name] = imhdu.header
        self.image[name] = imhdu.data[imslice].byteswap().newbyteorder()
        self.var[name] = FITSCutout ( var, center, [size,size], extension=var_ext ).fits_cutouts[0][1].data[imslice].byteswap().newbyteorder()
        self.psf[name] = fits.getdata(psf, psf_ext)
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
                     psf=None, verbose=True ):
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
          
        #w1 = CosineBellWindow ( alpha=cbell_alpha )
        w1 = HanningWindow ()
        
        if verbose:
            print('    Copying to matched arrays ... ')
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
        if verbose:
            print('        ... Done.')
        for idx in range(len(self.bands)):
            if idx == matchindex:
                continue
            cband = self.bands[idx]
            if verbose:
                print(f'    Convolving matching kernel for {self.bands[idx]} ...')
            kernel = create_matching_kernel(_pad_psf(psf[cband]),
                                            _pad_psf(psf[badband]),
                                            window=w1)
            matched_image[cband] = ndimage.convolve(self.image[cband], kernel)
            matched_psf[cband] = ndimage.convolve(psf[cband], kernel)
            if verbose:
                print('         ... Done.')
            
            # \\ Propagate error through convolution following Klein+21
            # \\ ignoring initial pixel correlations!!
            # \\ ( https://iopscience.iop.org/article/10.3847/2515-5172/abe8df )
            matched_var[cband] = ndimage.convolve(self.var[cband], kernel**2)
            
        self.matched_image = matched_image
        self.matched_psf = matched_psf
        self.matched_var = matched_var
        return matched_image, matched_psf
    
    def compute_mbexcess ( self,
                          band, 
                          method='single', 
                          scaling_factor=1.,
                          scaling_band='z',
                          psf_matched=True, 
                          extinction_correction=None,
                          ge_correction=None, 
                          line_correction=None,
                          redshift=0.08, 
                          continuum_type='powerlaw',
                          ):
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
            bb0,bb1 = scaling_band
            bb_blue = img_d[bb0]
            bb_red = img_d[bb1]            
            v_bb_blue = var_d[bb0]
            v_bb_red = var_d[bb1]
            
            continuum = 0.5 * ( bb_blue + bb_red )
            v_continuum = 0.25 * (v_bb_blue + v_bb_red)
        elif method == 'single':
            continuum = img_d[scaling_band]*scaling_factor
            v_continuum = var_d[scaling_band]*scaling_factor**2
        elif method == 'abby':
            from . import emission
            
            emission_package = emission.mbestimate_emission_line(
                img_d[band],
                img_d['g'],
                img_d['r'],
                img_d['i'],
                img_d['z'],
                redshift=redshift,
                u_mb_data=var_d[band]**0.5,
                u_rdata = var_d['g']**0.5,
                u_idata = var_d['r']**0.5,
                band=band,
                do_aperturecorrection=False,
                do_extinctioncorrection=extinction_correction is not None,
                do_gecorrection=ge_correction is not None,
                do_linecorrection=line_correction is not None,
                ge_correction=ge_correction,
                ex_correction=extinction_correction,
                ns_correction=line_correction,
                zp=27.,
                ctype=continuum_type,
                plawbands=average_bb[band]
            )
            return emission_package
             
    
        excess = mbimg - continuum
        v_excess = v_mbimg + v_continuum
        return excess, v_excess
    
    def clean_nonexcess_sources (self,):
        from ekfstats import sampling
        
        self.cleaned_image = {}
        
        for band in self.bands:
            # \\ only detect things that have substantial negative emission
            _,segmap = sep.extract(-self.image[band], 5., var=abs(self.var[band]), segmentation_map=True, minarea=5)
            _,segmap_lsb = sep.extract(-self.image[band], 2., var=abs(self.var[band]), segmentation_map=True, minarea=5)

            # Start with a copy of segmap_lsb to modify
            filtered_map = np.zeros_like(segmap_lsb)

            # Get the unique labels in segmap_lsb (excluding 0 which is background)
            labels = np.unique(segmap_lsb)
            labels = labels[labels != 0]

            # Iterate over each label in segmap_lsb
            for label in labels:
                mask = segmap_lsb == label  # Boolean mask of the current blob
                if np.any(segmap[mask] > 0):  # Check if any pixel in segmap overlaps with a detection
                    filtered_map[mask] = label  # Keep this blob   
            
            bkg_std = sampling.sigmaclipped_std ( self.image[band] )
            bkg_mean = np.mean(sampling.sigmaclip(self.image[band])[0])
            self.cleaned_image[band] = np.where(
                filtered_map,
                np.nan,#np.random.normal(bkg_mean, bkg_std, filtered_map.shape),
                self.image[band]
            )
            
            
        
    
    def define_autoaper ( self, band=None, image=None, var=None, clobber=False, ellipsify=True, thresh=5., ellipse_size=9. ):
        if hasattr(self, 'regionauto') and not clobber:
            return self.regionauto  
        
        if image is None:
            assert band is not None
            image = self.matched_image[band]
            var = self.matched_var[band]
            
        catalog, segmap = sep.extract ( image,
                                        var = var,
                                        thresh=thresh,
                                        #deblend_cont=0.05,
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
            regionauto = ep < ellipse_size # this value comes from SExtractor manual :shrug: 
        else:
            print('[pixels.BBMBImage.define_autoaper] Danger! Doing photometry directly from the segmentation map!')
        
        #self.regionauto = regionauto
        #self.autocat = catalog[[cid]]
        #self.segmap = segmap
        return regionauto, catalog[[cid]]
    
    def do_ephotometry ( bbmb, image=None, band=None ):
        objs, segmap = sep.extract(
            data=bbmb.matched_image['g'],
            thresh=5,
            var=bbmb.matched_var['g'],
            segmentation_map=True
        )
        
        cid = eis.get_centerval(segmap) - 1
        
        pix_conversion=1.
        catparams = objs[cid]
        cyy = catparams['cyy'] * pix_conversion**2
        cxx = catparams['cxx'] * pix_conversion**2
        cxy = catparams['cxy'] * pix_conversion**2
        
        y,x = np.mgrid[:bbmb.matched_image['g'].shape[0],:bbmb.matched_image['g'].shape[1]]
        xoff = x - catparams['x']
        yoff = y - catparams['y']
        ellipse = cyy*yoff**2 + cxx*xoff**2 + cxy*xoff*yoff
        ellipse_size = 9.
        emask = ellipse < ellipse_size
        
        integrated_halum = np.nansum(np.where(emask, halum, 0.))
        
def pull_cutouts ( coordinates, savedir, hsc_username=None, hsc_password=None ):
    handler.fetch_merian(cfile, savedir)
    handler.fetch_hsc(cfile, savedir, hsc_username=username, hsc_passwd=password)

def build_bbmb ( gid, **kwargs ):
    bbmb = pixels.BBMBImage ( )
    for band in ['g','N540','r','N708','i','z','y']:
        bbmb.add_band ( band, *load_image(gid, band) )

    fwhm_a, _ = bbmb.measure_psfsizes()
    mim, mpsf = bbmb.match_psfs ( np.argmax(fwhm_a), cbell_alpha=1., **kwargs )
    return bbmb