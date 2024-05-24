"""Generate cutout for Merian images using HSC or DECam images.
See https://github.com/MerianSurvey/caterpillar/blob/main/caterpillar/lsstpipe/cutout.py
"""

import os
import shutil
import argparse
import warnings

from datetime import date
from matplotlib import collections

import numpy as np

import astropy.units as u
from astropy.table import Table, QTable

from joblib import Parallel, delayed
from spherical_geometry.polygon import SphericalPolygon

try:
    import lsst.log
    Log = lsst.log.Log()
    Log.setLevel(lsst.log.ERROR)

    import lsst.daf.butler as dafButler
    import lsst.geom as geom
    import lsst.afw.image as afwImage
    import lsst.afw.geom as afwGeom
except ImportError:
    warnings.warn("lsstPipe is not installed. Please install it first.")

MERIAN_REPO = '/projects/MERIAN/repo'
PIXEL_SCALE = 0.168  # arcsec / pixel


def _afw_coords(coord_list):
    """
    Convert list of ra and dec to lsst.afw.coord.IcrsCoord.

    Parameters
    ----------
    coord_list : list of tuples or tuple
        ra and dec in degrees.

    Returns
    -------
    afw_coords : list of lsst.afw.coord.IcrsCoord
    """
    if type(coord_list[0]) in (float, int, np.float64):
        ra, dec = coord_list
        afw_coords = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)
    else:
        afw_coords = [
            geom.SpherePoint(ra * geom.degrees, dec * geom.degrees) for ra, dec in coord_list]

    return afw_coords


def sky_cone(ra_c, dec_c, theta, steps=50, include_center=True):
    """
    Get ra and dec coordinates of a cone on the sky.

    Parameters
    ----------
    ra_c, dec_c: float
        Center of cone in degrees.
    theta: astropy Quantity, float, or int
        Angular radius of cone. Must be in arcsec
        if not a Quantity object.
    steps: int, optional
        Number of steps in the cone.
    include_center: bool, optional
        If True, include center point in cone.

    Returns
    -------
    ra, dec: ndarry
        Coordinates of cone.
    """
    if isinstance(theta, float) or isinstance(theta, int):
        theta = theta * u.Unit('arcsec')

    cone = SphericalPolygon.from_cone(
        ra_c, dec_c, theta.to('deg').value, steps=steps)
    ra, dec = list(cone.to_lonlat())[0]
    ra = np.mod(ra - 360., 360.0)
    if include_center:
        ra = np.concatenate([ra, [ra_c]])
        dec = np.concatenate([dec, [dec_c]])
    return ra, dec


def get_tract_patch_list(coord_list, skymap):
    """
    Find the tracts and patches that overlap with the
    coordinates in coord_list. Pass the four corners of
    a rectangle to get all tracts and patches that overlap
    with this region.

    Parameters
    ----------
    coord_list : list (tuples or lsst.afw.coord.IcrsCoord)
        ra and dec of region
    skymap : lsst.skymap.ringsSkyMap.RingsSkyMap, optional
        The lsst/hsc skymap.

    Returns
    -------
    region_ids : structured ndarray
        Tracts and patches that overlap coord_list.
    tract_patch_dict : dict
        Dictionary of dictionaries, which takes a tract
        and patch and returns a patch info object.
    """
    if isinstance(coord_list[0], float) or isinstance(coord_list[0], int):
        coord_list = [_afw_coords(coord_list)]
    elif not isinstance(coord_list[0], geom.SpherePoint):
        coord_list = _afw_coords(coord_list)

    tract_patch_list = skymap.findTractPatchList(coord_list)

    ids = []
    for tract_info, patch_info_list in tract_patch_list:
        for patch_info in patch_info_list:
            ids.append((tract_info.getId(), patch_info.getSequentialIndex()))

    return np.array(ids, dtype=[('tract', int), ('patch', int)])


def _get_patches(butler, skymap, skymap_name, coord_list, band, data_type='deepCoadd'):
    """
    Retrieve the data products for all the patches that overlap with the coordinate.
    """
    # Retrieve the Tracts and Patches that cover the cutout region
    patches = get_tract_patch_list(coord_list, skymap)
    # Collect the images
    images = []
    for t, p in patches:
        data_id = {'tract': t, 'patch': p,
                   'band': band, 'skymap': skymap_name}
        try:
            if butler.datasetExists(data_type, data_id):
                img = butler.get(data_type, data_id)
                images.append(img)
        except LookupError as e:
            print(e)
            # Some times a Tract or Patch is not available in the data repo
            pass

    if len(images) == 0:
        return None
    return images


def _get_single_cutout(img, coord, half_size_pix):
    """Cutout from a single patch image.

    half_size_pix needs to be in pixels.
    """
    # Get the WCS and the pixel coordinate of the central pixel
    wcs = img.getWcs()
    pix = geom.Point2I(wcs.skyToPixel(coord))

    # Define a bounding box for the cutout region
    bbox = geom.Box2I(pix, pix)
    bbox.grow(half_size_pix)

    # Original pixel coordinate of the bounding box
    x0, y0 = bbox.getBegin()

    # Clip the cutout region from the original image
    bbox.clip(img.getBBox(afwImage.PARENT))

    # Make an afwImage object
    cut = img.Factory(img, bbox, afwImage.PARENT)

    return cut, x0, y0


def _build_cutout_wcs(coord, cutouts, index, origins):
    """Build new WCS header for the cutout."""
    # Get the WCS information from the largest cutout
    largest_cutout = cutouts[index]
    subwcs = largest_cutout.getWcs()

    # Information for the WCS header
    crpix_1, crpix_2 = subwcs.skyToPixel(coord)
    crpix_1 -= origins[index][0]
    crpix_2 -= origins[index][1]
    cdmat = subwcs.getCdMatrix()

    wcs_header = lsst.daf.base.PropertyList()
    wcs_header.add('CRVAL1', coord.getRa().asDegrees())
    wcs_header.add('CRVAL2', coord.getDec().asDegrees())
    wcs_header.add('CRPIX1', crpix_1 + 1)
    wcs_header.add('CRPIX2', crpix_2 + 1)
    wcs_header.add('CTYPE1', 'RA---TAN')
    wcs_header.add('CTYPE2', 'DEC--TAN')
    wcs_header.add('CD1_1', cdmat[0, 0])
    wcs_header.add('CD2_1', cdmat[1, 0])
    wcs_header.add('CD1_2', cdmat[0, 1])
    wcs_header.add('CD2_2', cdmat[1, 1])
    wcs_header.add('RADESYS', 'ICRS')

    return afwGeom.makeSkyWcs(wcs_header)


def _get_psf(exp, coord):
    """Get the coadd PSF image.

    Parameters
    ----------
    exp: lsst.afw.image.exposure.exposure.ExposureF
        Exposure
    coord: lsst.geom.SpherePoint
        Coordinate for extracting PSF

    Returns
    -------
    psf_img: lsst.afw.image.image.image.ImageD
        2-D PSF image
    """
    wcs = exp.getWcs()
    if not isinstance(coord, geom.SpherePoint):
        coord = _afw_coords(coord)
    coord = wcs.skyToPixel(coord)
    psf = exp.getPsf()

    try:
        psf_img = psf.computeKernelImage(coord)
        return psf_img
    except Exception:
        print('**** Cannot compute PSF Image *****')
        return None


def generate_cutout(butler, skymap_name, ra, dec, band='N708', data_type='deepCoadd_calexp',
                    half_size=10.0 * u.arcsec, psf=True, verbose=False):
    """
    Generate a single cutout image.
    """
    skymap = butler.get('skyMap', skymap=skymap_name)

    if not isinstance(half_size, u.Quantity):
        # Assume that this is in pixel
        size_pix = int(2 * half_size) + 1
    else:
        size_pix = int(2 * half_size.to('arcsec').value / PIXEL_SCALE)

    half_size_pix = int((size_pix - 1) / 2)
    
    # Width and height of the post-stamps
    stamp_shape = (size_pix + 1, size_pix + 1)

    # Make a list of (RA, Dec) that covers the cutout region
    radec_list = np.array(
        sky_cone(ra, dec, half_size_pix * PIXEL_SCALE * u.Unit('arcsec'), steps=50)).T

    # Retrieve the Patches that cover the cutout region
    img_patches = _get_patches(
        butler, skymap, skymap_name, radec_list, band, data_type=data_type)

    if img_patches is None:
        if verbose:
            print('***** No data at {:.5f} {:.5f} *****'.format(ra, dec))
        return None

    # Coordinate of the image center
    coord = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)

    # Making the stacked cutout
    cutouts = []
    idx, bbox_sizes, bbox_origins = [], [], []

    for img_p in img_patches:
        # Generate cutout
        cut, x0, y0 = _get_single_cutout(img_p, coord, half_size_pix)
        cutouts.append(cut)
        # Original lower corner pixel coordinate
        bbox_origins.append([x0, y0])
        # New lower corner pixel coordinate
        xnew, ynew = cut.getBBox().getBeginX() - x0, cut.getBBox().getBeginY() - y0
        idx.append([xnew, xnew + cut.getBBox().getWidth(),
                    ynew, ynew + cut.getBBox().getHeight()])
        # Area of the cutout region on this patch in unit of pixels
        # Will reverse rank all the overlapped images by this
        bbox_sizes.append(cut.getBBox().getWidth() * cut.getBBox().getHeight())

    # Stitch cutouts together with the largest bboxes inserted last
    stamp = afwImage.MaskedImageF(
        geom.BoxI(geom.Point2I(0, 0), geom.Extent2I(*stamp_shape)))
    bbox_sorted_ind = np.argsort(bbox_sizes)

    for i in bbox_sorted_ind:
        masked_img = cutouts[i].getMaskedImage()
        stamp[idx[i][0]: idx[i][1], idx[i][2]: idx[i][3]] = masked_img

    # Build the new WCS of the cutout
    stamp_wcs = _build_cutout_wcs(
        coord, cutouts, bbox_sorted_ind[-1], bbox_origins)

    cutout = afwImage.ExposureF(stamp, stamp_wcs)

    if bbox_sizes[bbox_sorted_ind[-1]] < (half_size_pix * 2 + 1) ** 2:
        flag = 1
    else:
        flag = 2

    # The final product of the cutout
    if psf:
        psf = _get_psf(cutouts[bbox_sorted_ind[-1]], coord)
        return cutout, psf, flag
    return cutout, flag


def padding_img(img, output_size=(100, 100)):
    '''
    If the sizes of imgs in all bands are not the same, 
    this function pads the smaller PSFs to the output size.
    '''
    # Padding PSF cutouts from HSC
    max_len = max(img.shape)

    y_len, x_len = img.shape
    dy = ((max_len - y_len) // 2, (max_len - y_len) // 2)
    dx = ((max_len - x_len) // 2, (max_len - x_len) // 2)

    if (max_len - y_len) == 1:
        dy = (1, 0)
    if (max_len - x_len) == 1:
        dx = (1, 0)

    temp = np.pad(img.astype('float'),
                  (dy, dx), 'constant', constant_values=0)

    # Then padding the image to the output size
    if output_size is not None:
        temp = np.pad(
            temp, ((output_size[0] - temp.shape[0]) // 2, (output_size[1] - temp.shape[1]) // 2))
    if temp.shape[0] == temp.shape[1]:
        if output_size is not None and abs(temp.shape[0] - output_size[0]) <= 1 and abs(temp.shape[1] - output_size[1]) <= 1:
            return temp
        else:
            raise ValueError(
                f'The padded image has a size of {temp.shape}, which is not close to the output size.')
    else:
        raise ValueError(
            'The padded image has a size of {temp.shape}, which is not a square.')