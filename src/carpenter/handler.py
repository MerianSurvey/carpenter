"""
Top level handling of cutout creation
"""
import argparse
import os
import subprocess
import warnings
import numpy as np
from astropy import coordinates
from astropy.io import fits
#from astropy.table import Table, hstack
import astropy.units as u
from lsst.daf import butler as dafButler
from .cutout import generate_cutout
from . import conventions

_DEFAULT_REPO = '/projects/MERIAN/repo/'
_DEFAULT_COLLECTIONS = 'DECam/runs/merian/dr1_wide'
def instantiate_butler ( repo=None, collections=None ):
    """
    Instantiate a Butler object for accessing data from a Data Butler repository.

    Parameters
    ----------
    repo : str or pathlib.Path, optional
        Path to the repository root directory. If not specified, the default
        repository path is used.
    collections : str or list of str, optional
        Name of the data collection or list of names to use within the repository.
        If not specified, the default collection is used.

    Returns
    -------
    dafButler.Butler
        A Butler object that can be used to access data from the specified repository.

    Notes
    -----
    The Data Butler is a data access framework used in the LSST software stack. This
    function provides a simple way to instantiate a Butler object for accessing data
    from a repository. If the `repo` or `collections` parameters are not specified,
    default values are used. These default values are specific to the Merian project,
    so you may need to modify them for use with a different repository.

    Example
    -------
    To instantiate a Butler object for accessing data from the default Merian repository:

    >>> butler = instantiate_butler()

    To instantiate a Butler object for accessing data from a different repository:

    >>> butler = instantiate_butler(repo='/path/to/repository', collections=['my_collection'])

    """
    
    if repo is None:
        repo = _DEFAULT_REPO
    if collections is None:
        collections = _DEFAULT_COLLECTIONS
    butler = dafButler.Butler(repo, collections=collections )
    return butler

def pull_merian_cutouts(coordlist, butler, save=True, savedir='./', half_size=None):
    """
    Pull cutouts of Merian objects from the specified collection.

    Parameters
    ----------
    coordlist : list of tuples
        List of (RA, Dec) tuples for which to retrieve cutouts.
    butler : lsst.daf.persistence.Butler
        Butler containing the collection from which to retrieve cutouts.
    save : bool, optional
        If True, save the retrieved cutouts to disk. Otherwise, keep them in memory.
    savedir : str, optional
        The directory to save cutouts to if `save=True`.
    half_size : astropy.units.Quantity, optional
        The size of each cutout. If not specified, defaults to 30 arcseconds.

    Returns
    -------
    list of lists of str or `astropy.io.fits.HDUList`
        The retrieved cutouts for each object. The outer list has one element per object,
        and each element is a list containing the cutouts for each filter (N708, N540).
        If `save=True`, each element of the inner list is a string containing the
        filename of the saved FITS file. Otherwise, each element is an `HDUList`.
    """
    if half_size is None:
        half_size = 30.*u.arcsec
    if not save:
        warnings.warn ( 'You have chosen to keep all requested cutouts in memory!' )
        
    merian_cutouts = []
    for ccoord in coordlist:
        ra, dec = ccoord
        mer_imgs = []
        for filt in ['N708', 'N540']:
            img, psf, _ = generate_cutout(butler, 'hsc_rings_v1', 
                                        ra, dec, 
                                        half_size=half_size,
                                        band=filt, 
                                        data_type='deepCoadd_calexp',
                                        psf=True)
            if save:            
                sc = coordinates.SkyCoord ( *ccoord, unit='deg' )
                filename = conventions.produce_merianfilename ( sc, filt, objtype='merim', savedir=savedir )  
                psfname = conventions.produce_merianfilename ( sc, filt, objtype='merpsf', savedir=savedir )
                img.writeFits ( filename )
                psf.writeFits ( psfname )
                print(f'Saved {filename} ({psfname})')    
                mer_imgs.append(filename)            
            else:
                mer_imgs.append(img)
        merian_cutouts.append(mer_imgs)        
            
    return merian_cutouts

def setup_hsc_request ( coordlist, 
                       half_size=None, 
                       rerun='pdr3_wide', 
                       image='true', 
                       mask='true', 
                       variance='true', 
                       filetype='coadd', 
                       savedir='./',
                       listname='hsc_download.txt'
                       ):
    if half_size is None:
        half_size = 30.*u.arcsec
    sw = sh = f'{half_size.to(u.arcsec).value:.0f}asec'
    
    header = "#? rerun        ra       dec       sw    sh   filter  image  mask variance type name"
    downloadfile =  f'{savedir}/{listname}'
    with open ( downloadfile,'w') as f:
        print(header, file=f)
        for ccoord in coordlist:
            sc = coordinates.SkyCoord ( *ccoord, unit='deg' )
            ra_s = sc.ra.to_string(unit='hourangle', sep=":", precision=2 )
            dec_s = sc.dec.to_string(unit='deg', sep=":", precision=2 )
            for band in 'grizy':
                objname = f'{conventions.produce_merianobjectname(sc)}_HSC-{band}'
                row = f'{rerun} {ra_s} {dec_s} {sw} {sh} {band} {image} {mask} {variance} {filetype} {objname}'
                print(row, file=f)        
    return downloadfile

def download_hsccutouts ( filename, savedir='./', username=None, ):
    if username is None:
        username = f"{os.environ['USER']}@local"
    cutout_url = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout"
    psf_url = "https://hsc-release.mtk.nao.ac.jp/psf/pdr3/cgi/getpsf?bulk=on"
    cutout_command = f'cd {savedir}; curl {cutout_url} --form list=@{filename} --user "{username}" | tar xvf -'
    psf_command = f'cd {savedir}; curl {psf_url} --form list=@{filename} --user "{username}" | tar xvf -'
    #subprocess.run ( command )
    #print(command)
    #output, error = process.communicate ()
    
def fetch ( coordlist, savedir, butler=None, hsc_username=None, *args, **kwargs ):
    if not os.path.exists(f'{savedir}/merian/'):
        os.makedirs ( f'{savedir}/merian/' )
    if not os.path.exists(f'{savedir}/hsc'):
        os.makedirs ( f'{savedir}/hsc/' )
            
    coordlist = np.genfromtxt (coordlist)
    if butler is None:
        butler = instantiate_butler ()
    merian_cutouts = pull_merian_cutouts ( coordlist, butler, savedir=f'{savedir}/merian/', **kwargs)
    downloadfile = setup_hsc_request ( coordlist, savedir=f'{savedir}/hsc', **kwargs)
    download_hsccutouts ( downloadfile, f'{savedir}/hsc', username=hsc_username )
