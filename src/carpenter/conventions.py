"""
Define carpenter naming conventions
"""
import astropy.units as u

def produce_merianobjectname(skycoordobj):
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
    coordinate_name = produce_merianobjectname(skycoordobj)
    return f'{savedir}/{coordinate_name}_{filt}{objtype}.fits'