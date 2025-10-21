import copy
import numpy as np
from scipy import ndimage
from astropy import coordinates, cosmology
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.io import fits
import sep
from photutils.psf.matching import create_matching_kernel, HanningWindow, CosineBellWindow
from astrocut import FITSCutout
import astropy.units as u
from scipy.ndimage import generic_filter
from astropy.wcs import WCS
from reproject import reproject_interp


cosmo = cosmology.FlatLambdaCDM(70.,0.3)

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
        if psf is not None:
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
        alpha_a = np.zeros(len(bands))
        gamma_a = np.zeros(len(bands))
        model_psf = copy.copy(psf)
        for ix,band in enumerate(bands):
            z = _pad_psf(psf[band])

            mid = np.array(z.shape)//2

            y,x = np.mgrid[:z.shape[0],:z.shape[1]]
            p_init = models.Moffat2D(x_0=mid[0], y_0=mid[1], amplitude=z[mid[0],mid[1]])
            p = fit_p(p_init, x, y, z)
            fwhm_a[ix] = p.fwhm
            alpha_a[ix] = p.alpha.value
            gamma_a[ix] = p.gamma.value
            model_psf[band] = p(x,y)
        if save:
            self.fwhm_to_match = fwhm_a.max()
            self.band_to_match = self.bands[np.argmax(fwhm_a)]
            self.psf_matching_params = {'alpha':alpha_a[np.argmax(fwhm_a)], 'gamma':gamma_a[np.argmax(fwhm_a)]}
        return fwhm_a, model_psf

    def match_psfs ( self, matchindex=None, refband=None,
                     psf=None, verbose=True, w_type = 'hanning',
                     reprojected=True, cbell_alpha=0.5 ):
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
        reprojected: (bool, default=True) if True, match PSFs of reprojected images
                     and variances, otherwise match original images and variances
        '''
        if verbose:
            print ('[SEDMap] Matching PSFs')

        if psf is None:
            psf = self.psf

        if w_type == 'hanning':
            w1 = HanningWindow ()
        elif w_type == 'cosine':
            w1 = CosineBellWindow ( alpha=cbell_alpha )
        
        if verbose:
            print('    Copying to matched arrays ... ')

        if reprojected:
            img_d = self.reprojected_images
            var_d = self.reprojected_var    
        else:
            img_d = self.image
            var_d = self.var  

        matched_image = copy.copy (img_d)
        matched_psf = copy.copy ( psf )
        matched_var = copy.copy ( var_d )
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
            matched_image[cband] = ndimage.convolve(img_d[cband], kernel)
            matched_psf[cband] = ndimage.convolve(psf[cband], kernel)
            if verbose:
                print('         ... Done.')
            
            # \\ Propagate error through convolution following Klein+21
            # \\ ignoring initial pixel correlations!!
            # \\ ( https://iopscience.iop.org/article/10.3847/2515-5172/abe8df )
            matched_var[cband] = ndimage.convolve(var_d[cband], kernel**2)

        self.matched_image = matched_image
        self.matched_psf = matched_psf
        self.matched_var = matched_var
        return matched_image, matched_psf

    def reproject(self, psf_matched=False, refband='N708'):
        """
        Reproject all image bands onto the WCS of a reference band.
        Parameters
        ----------
        psf_matched : bool, optional
            If True, use PSF-matched images and variances for reprojection.
            If False (default), use the original images and variances.
        refband : str, optional
            The band to use as the reference for WCS and output shape.
            Default is 'N708'.
        Returns
        -------
        reprojected_images : dict
            Dictionary mapping band names to their reprojected image arrays.
        Notes
        -----
        - The method also stores the reprojected images and variances in
          `self.reprojected_images` and `self.reprojected_var`.
        - NaN values in the reprojected arrays are replaced with zeros.
        - Each band's image and variance are reprojected using interpolation
          onto the reference WCS and shape.
        """


        
        # ideally, you should reproject before psf matching
        if psf_matched:
            img_d = self.matched_image
            var_d = self.matched_var
        else:
            img_d = copy.copy(self.image)
            var_d = copy.copy(self.var)

        # Choose a reference WCS from one of the bands (N708 in this case)
        reference_wcs = WCS(self.hdu[refband])
        reference_shape = img_d[refband].shape

        # Create a dictionary to store the reprojected images
        reprojected_images = {}
        reprojected_var = {}
        # Reproject each band onto the reference WCS
        for band in self.bands:
            input_wcs = WCS(self.hdu[band])
            reprojected_array, footprint = reproject_interp(
                (img_d[band], input_wcs),
                reference_wcs,
                shape_out=reference_shape
            )
            reprojected_images[band] = np.nan_to_num(reprojected_array, 0)

            reprojected_var_array, footprint_var = reproject_interp(
                (var_d[band], input_wcs),
                reference_wcs,
                shape_out=reference_shape
            )
            reprojected_var[band] = np.nan_to_num(reprojected_var_array, 0)

        self.reprojected_images = reprojected_images
        self.reprojected_var = reprojected_var
        return reprojected_images

    def compute_mbexcess ( self,
                          band, 
                          method='single', 
                          scaling_factor=1.,
                          scaling_band='z',
                          psf_matched=True,
                          reprojected=True,
                          pre_smooth=False,
                          post_smooth=False,
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
        # note that ideally, psf_matched images have already been reprojected
        if psf_matched:
            img_d = self.matched_image
            var_d = self.matched_var
        
        elif reprojected:
            img_d = self.reprojected_images
            var_d = self.reprojected_var

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
        elif method == '2dpowerlaw':
            from . import emission

            if pre_smooth:
                img_c = dict([(b,generic_filter(img_d[b], np.nanmedian, size=3, mode = 'constant', cval = np.nan)) for b in img_d.keys()])
            else:
                img_c = img_d

            emission_package = emission.mbestimate_emission_line(
                img_c[band].flatten(),
                img_c['g'].flatten(),
                img_c['r'].flatten(),
                img_c['i'].flatten(),
                img_c['z'].flatten(),
                redshift=redshift,
                u_mb_data=var_d[band].flatten()**0.5,
                u_rdata=var_d['g'].flatten()**0.5,
                u_idata=var_d['r'].flatten()**0.5,
                band=band.lower(),
                do_aperturecorrection=False,
                do_extinctioncorrection=extinction_correction is not None,
                do_gecorrection=ge_correction is not None,
                do_linecorrection=line_correction is not None,
                ge_correction=ge_correction,
                ex_correction=extinction_correction,
                ns_correction=line_correction,
                zp=27.,
                ctype=continuum_type,
                plawbands=average_bb[band.lower()],
                specflux_unit = u.nJy
            )
            # return emission_package
            continuum = emission_package[3].value.reshape(mbimg.shape)
            continuum = continuum/img_c[scaling_band] * img_d[scaling_band] # will have no effect if pre_smooth=False
            if post_smooth:
                continuum = generic_filter(continuum, np.mean, size=3) 
            v_continuum = np.zeros_like(continuum) # \\ ignoring uncertainty in continuum estimate for now
    
        excess = mbimg - continuum
        v_excess = v_mbimg + v_continuum
        return excess, v_excess, continuum
    
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


def calculate_surface_density(image, pixel_scale, radius, redshift=None, distance=None):
    """
    Calculate the average surface density within radius R for each pixel in an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D input image (e.g., flux density map)
    pixel_scale : astropy.units.Quantity
        Physical or angular size of each pixel (e.g., 0.168*u.arcsec or 0.5*u.kpc)
    radius : astropy.units.Quantity
        Radius within which to calculate average surface density
        (e.g., 2*u.arcsec, 5*u.kpc, or 10*u.pixel)
    redshift : float, optional
        Redshift of the galaxy (required if converting between angular and physical units)
    distance : astropy.units.Quantity or astropy.coordinates.Distance, optional
        Distance to the galaxy (alternative to redshift for distance calculation)
        
    Returns:
    --------
    surface_density : numpy.ndarray
        2D array with average surface density within radius R for each pixel
        
    Notes:
    ------
    - For Merian survey targets (0.05 < z < 0.1), physical scale conversions use
      the angular diameter distance
    - Surface density preserves the units of the input image divided by area units
    - If input image is in flux units, output will be in flux per unit area
    - Use u.pixel for dimensionless pixel units
    
    Examples:
    ---------
    >>> import astropy.units as u
    >>> # HSC pixel scale
    >>> pixel_scale = 0.168 * u.arcsec
    >>> radius = 2.0 * u.arcsec
    >>> surf_dens = calculate_surface_density(image, pixel_scale, radius, redshift=0.075)
    
    >>> # Physical scale
    >>> pixel_scale = 0.168 * u.arcsec  
    >>> radius = 5.0 * u.kpc
    >>> surf_dens = calculate_surface_density(image, pixel_scale, radius, redshift=0.075)
    
    >>> # Pixel units
    >>> pixel_scale = 0.168 * u.arcsec
    >>> radius = 10 * u.pixel
    >>> surf_dens = calculate_surface_density(image, pixel_scale, radius)
    """
    
    # Input validation
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D numpy array")
    
    if not isinstance(pixel_scale, u.Quantity):
        raise ValueError("pixel_scale must be an astropy Quantity with units")
        
    if not isinstance(radius, u.Quantity):
        raise ValueError("radius must be an astropy Quantity with units")
    
    if radius.value <= 0:
        raise ValueError("Radius must be positive")
    
    # Convert radius to pixels
    radius_pixels = _convert_to_pixels(radius, pixel_scale, redshift, distance)
    
    # Create a circular kernel for the averaging
    kernel_size = int(np.ceil(2 * radius_pixels)) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size for proper centering
    
    # Create coordinate grids for the kernel
    center = kernel_size // 2
    y, x = np.ogrid[:kernel_size, :kernel_size]
    
    # Create circular mask
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    circular_mask = distance_from_center <= radius_pixels
    
    # Normalize the kernel (for average calculation)
    kernel = circular_mask.astype(float)
    kernel_sum = np.sum(kernel) # npix
    if radius.unit.is_equivalent(u.kpc):
        kernel_area = kernel_sum * ((pixel_scale * cosmo.kpc_comoving_per_arcmin(redshift))**2).to(u.kpc**2)
    elif radius.unit.is_equivalent(u.arcsec):
        kernel_area = kernel_sum * pixel_scale.to(u.arcsec)**2
    elif radius.unit.is_equivalent(u.pix):
        kernel_area = kernel_sum
    
    if kernel_sum == 0:
        warnings.warn("Radius too small - no pixels included in aperture")
        return np.zeros_like(image)
    
    #kernel = kernel / kernel_sum
    
    # Apply convolution to get average surface density
    
    surface_density = ndimage.convolve(image, kernel, mode='constant', cval=0.0) 
    if hasattr(image,'unit'):
        surface_density = surface_density * image.unit 
    surface_density = surface_density / kernel_area
    
    return surface_density


def _convert_to_pixels(radius, pixel_scale, redshift=None, distance=None):
    """
    Convert a radius measurement to pixels using astropy units.
    
    Parameters:
    -----------
    radius : astropy.units.Quantity
        The radius to convert
    pixel_scale : astropy.units.Quantity
        Size of each pixel
    redshift : float, optional
        Redshift for cosmological distance calculations
    distance : astropy.units.Quantity or astropy.coordinates.Distance, optional
        Distance to the galaxy (alternative to redshift)
        
    Returns:
    --------
    radius_pixels : float
        Radius converted to pixels (dimensionless)
    """
    
    # Handle pixel units directly
    if radius.unit == u.pixel:
        return radius.value
    
    # If pixel_scale is also in pixels, we can't do unit conversion
    if pixel_scale.unit == u.pixel:
        if radius.unit != u.pixel:
            raise ValueError("Cannot convert radius to pixels when pixel_scale is dimensionless")
        return radius.value
    
    # Check if both quantities have compatible dimensions
    try:
        # If they're the same type of unit (both angular or both physical), convert directly
        radius_in_pixel_units = radius.to(pixel_scale.unit)
        return (radius_in_pixel_units / pixel_scale).decompose().value
    except u.UnitConversionError:
        # Need to convert between angular and physical units
        pass
    
    # Determine which quantity is angular and which is physical
    angular_units = [u.arcsec, u.arcmin, u.deg, u.mas, u.rad]
    physical_units = [u.kpc, u.pc, u.Mpc, u.m, u.km, u.cm]
    
    radius_is_angular = any(radius.unit.is_equivalent(unit) for unit in angular_units)
    radius_is_physical = any(radius.unit.is_equivalent(unit) for unit in physical_units)
    pixel_is_angular = any(pixel_scale.unit.is_equivalent(unit) for unit in angular_units)
    pixel_is_physical = any(pixel_scale.unit.is_equivalent(unit) for unit in physical_units)
    
    if not ((radius_is_angular or radius_is_physical) and (pixel_is_angular or pixel_is_physical)):
        raise ValueError(f"Cannot handle conversion between {radius.unit} and {pixel_scale.unit}")
    
    # If one is angular and one is physical, we need distance information
    if (radius_is_angular and pixel_is_physical) or (radius_is_physical and pixel_is_angular):
        if distance is None and redshift is None:
            raise ValueError("Need redshift or distance to convert between angular and physical units")
        
        # Get distance
        if distance is not None:
            if isinstance(distance, Distance):
                dist = distance
            else:
                # Assume it's a Quantity with distance units
                dist = Distance(distance)
        else:
            dist = cosmo.angular_diameter_distance(redshift)
        
        # Convert to consistent units
        if radius_is_angular and pixel_is_physical:
            # Convert radius from angular to physical
            radius_physical = (radius * dist).to(pixel_scale.unit)
            return (radius_physical / pixel_scale).decompose().value
        else:
            # Convert radius from physical to angular
            radius_angular = ((radius / dist).decompose()*u.rad).to(pixel_scale.unit)
            return (radius_angular / pixel_scale).decompose().value
    
    # Should not reach here
    raise ValueError(f"Cannot convert from {radius.unit} to pixels with pixel_scale in {pixel_scale.unit}")


def get_pixel_area(pixel_scale, redshift=None, distance=None):
    """
    Calculate the area of a pixel in various units.
    
    Parameters:
    -----------
    pixel_scale : astropy.units.Quantity
        Size of each pixel (linear dimension)
    redshift : float, optional
        Redshift for cosmological calculations
    distance : astropy.units.Quantity or astropy.coordinates.Distance, optional
        Distance to the galaxy
        
    Returns:
    --------
    dict : Dictionary with pixel areas in different units
        Keys: 'arcsec2', 'kpc2', 'pixel' (if applicable)
    """
    
    areas = {}
    
    # Physical area
    if pixel_scale.unit.is_equivalent(u.kpc):
        physical_area = (pixel_scale**2).to(u.kpc**2)
        areas['kpc2'] = physical_area
        
        # Convert to angular if distance available
        if distance is not None or redshift is not None:
            if distance is not None:
                if isinstance(distance, Distance):
                    dist = distance
                else:
                    dist = Distance(distance)
            else:
                dist = cosmo.angular_diameter_distance(redshift)
            
            angular_scale = (pixel_scale / dist).to(u.arcsec)
            areas['arcsec2'] = (angular_scale**2).to(u.arcsec**2)
    
    # Angular area
    elif pixel_scale.unit.is_equivalent(u.arcsec):
        angular_area = (pixel_scale**2).to(u.arcsec**2)
        areas['arcsec2'] = angular_area
        
        # Convert to physical if distance available
        if distance is not None or redshift is not None:
            if distance is not None:
                if isinstance(distance, Distance):
                    dist = distance
                else:
                    dist = Distance(distance)
            else:
                dist = cosmo.angular_diameter_distance(redshift)
            
            physical_scale = (pixel_scale.to(u.rad).value * dist).to(u.kpc)
            areas['kpc2'] = (physical_scale**2).to(u.kpc**2)
    
    # Dimensionless pixels
    elif pixel_scale.unit == u.pixel:
        areas['pixel'] = pixel_scale**2
    
    return areas


        
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