"""
Define carpenter naming conventions
"""
import numpy as np
from astropy import coordinates
import astropy.units as u

def produce_merianobjectname(ra=None, dec=None,skycoordobj=None, unit='deg'):
    """
    Given a `skycoordobj` object, returns a string representing the object name in the standard form:
    J{RAhms}{+/-}{DECdms}

    Parameters:
    -----------
    skycoordobj: astropy.coordinates.SkyCoord
        A `SkyCoord` object representing the coordinates of the object.

    Returns:
    --------
    cname: str
        A string representing the name of the object in the standard form.
    """
    if skycoordobj is None:
        skycoordobj = coordinates.SkyCoord(ra, dec, unit=unit)
    rastring = skycoordobj.ra.to_string(unit=u.hourangle, sep="", precision=2, pad=True)
    decstring = skycoordobj.dec.to_string(unit=u.deg, sep="", precision=2, pad=True)
    sign = '+' if skycoordobj.dec>0 else ''
    cname = f'J{rastring}{sign}{decstring}'
    return cname

def produce_merianfilename ( skycoordobj, filt, objtype=None, savedir='./' ):
    '''
    Make a filename based off default naming convention
    '''
    if objtype is None:
        objtype = ''
    else:
        objtype = f'_{objtype}'
    coordinate_name = produce_merianobjectname(skycoordobj=skycoordobj)
    return f'{savedir}/{coordinate_name}_{filt}{objtype}.fits'

def merianobjectname_to_catalogname ( objname, catalog, rakey='RA', deckey='DEC' ):
    catalog_coords = coordinates.SkyCoord( catalog[rakey], catalog[deckey], unit='deg')
    target = coordinates.SkyCoord(objname, unit=(u.hourangle, u.deg))
    target_separation = target.separation ( catalog_coords )
    assert np.min(target_separation) < (1.*u.arcsec), f'Best match found at {np.min(target_separation).to(u.arcsec).value}".'
    match = np.argmin(target_separation)
    return catalog.index[match]