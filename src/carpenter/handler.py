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
from astropy.nddata import Cutout2D
from astropy import wcs
import astropy.units as u
#try:
#    from lsst.daf import butler as dafButler
#except ModuleNotFoundError:
#    print('[LSSTExistenceWarning]: no lsst module found.')
from lsst.daf import butler as dafButler
from .cutout import generate_cutout
from . import conventions
import glob
from tqdm import tqdm
import re

#/scratch/gpfs/MERIAN/repo/main/DECam/runs/merian/dr1_wide
#/scratch/gpfs/MERIAN/repo/repos/main_2022_12_19
_DEFAULT_REPO = '/scratch/gpfs/MERIAN/repo/repos/main_2022_12_19' #'/projects/MERIAN/repo/'
_DEFAULT_COLLECTIONS = 'DECam/runs/merian/dr1_wide'#dr1_wide'


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
                if psf is not None:
                    psf.writeFits ( psfname )
                    print(f'Saved {filename} ({psfname})')    
                else:
                    print(f'Saved {filename}')
                mer_imgs.append(filename)            
            else:
                mer_imgs.append(img)
        merian_cutouts.append(mer_imgs)        
            
    return merian_cutouts

def setup_hsc_request ( coordlist, 
                       half_size, 
                       rerun='pdr3_wide', 
                       image='true', 
                       mask='true', 
                       variance='true', 
                       filetype='coadd', 
                       savedir='./',
                       listname='hsc_download.txt',
                       psf_centered="true", psf_file=False,
                       bands = 'grizy'):
    sw = sh = f'{half_size.to(u.arcsec).value:.0f}asec'
    
    if not psf_file:
        header = "#? rerun        ra       dec       sw    sh   filter  image  mask variance type name"
    else:
        header = "#? rerun        ra       dec       sw    sh   filter  image  mask variance type name centered"
        listname = f"{listname.split('.')[0]}_psf.txt"
    downloadfile =  f'{savedir}/{listname}'
    with open ( downloadfile,'w') as f:
        print(header, file=f)
        for ccoord in coordlist:
            sc = coordinates.SkyCoord ( *ccoord, unit='deg' )
            ra_s = sc.ra.to_string(unit='hourangle', sep=":", precision=4 )
            dec_s = sc.dec.to_string(unit='deg', sep=":", precision=4 )
            for band in bands:
                objname = f'{conventions.produce_merianobjectname(skycoordobj=sc)}_HSC-{band}'
                if not psf_file:
                    row = f'{rerun} {ra_s} {dec_s} {sw} {sh} {band} {image} {mask} {variance} {filetype} {objname}'
                else:
                    row = f'{rerun} {ra_s} {dec_s} {sw} {sh} {band} {image} {mask} {variance} {filetype} {objname} {psf_centered}'
                print(row, file=f)        
    return listname

def download_hsccutouts ( filename, filename_psf=None, savedir='./', username=None, passwd=None):
    if username is None:
        username = f"{os.environ['USER']}@local"
    cutout_url = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout"
    psf_url = "https://hsc-release.mtk.nao.ac.jp/psf/pdr3/cgi/getpsf?bulk=on"

    if passwd is not None:
        username = f"{username}:{passwd}"

    if filename_psf is None:
        filename_psf=filename
    cutout_command = f'cd {savedir}; curl {cutout_url} --form list=@{filename} --user "{username}" | tar xvf -'
    psf_command = f'cd {savedir}; curl {psf_url} --form list=@{filename_psf} --user "{username}" | tar xvf -'
    for command in [cutout_command, psf_command]:
        os.system ( command )
        print(command)
    #output, error = process.communicate ()
    
def fetch ( coordlist, savedir, butler=None, hsc_username=None, hsc_passwd=None, *args, **kwargs ):
    if not os.path.exists(f'{savedir}/merian/'):
        os.makedirs ( f'{savedir}/merian/' )
    if not os.path.exists(f'{savedir}/hsc'):
        os.makedirs ( f'{savedir}/hsc/' )
            
    coordlist = np.genfromtxt (coordlist)
    if butler is None:
        butler = instantiate_butler ()
    merian_cutouts = pull_merian_cutouts ( coordlist, butler, savedir=f'{savedir}/merian/', **kwargs)
    downloadfile = setup_hsc_request ( coordlist, savedir=f'{savedir}/hsc', psf_file=False, **kwargs)
    download_hsccutouts (downloadfile, savedir = f'{savedir}/hsc', username=hsc_username, passwd=hsc_passwd)

def fetch_hsc( coordlist, savedir, hsc_username=None, hsc_passwd=None, overwrite=False,
               mvfromsubdir = True, psf_centered="true", rename_psf = True, half_size=None, 
               filetype='coadd/bg', retrim=True, bands = 'grizy', *args, **kwargs):
    if not os.path.exists(f'{savedir}/hsc'):
        os.makedirs ( f'{savedir}/hsc/' )

    # for band in 'grizy':
    #     for tp in ['image','psf']:
    #         if not os.path.exists(f'{savedir}/hsc/hsc_{band}/{tp}'):
    #             os.makedirs(f'{savedir}/hsc/hsc_{band}/{tp}')
    
    if half_size is None:
        half_size = 30. * u.arcsec
    
    if isinstance(coordlist, str):
        coordlist = np.genfromtxt (coordlist)
        if len(coordlist.shape) == 1: 
            coordlist = coordlist.reshape(1,-1)        

    if not overwrite:
        print ("Finding which coordinates' cutouts have already been downloaded...")
        keepcoord = [not hsc_images_already_downloaded(c, savedir, bands) for c in tqdm(coordlist)]
        if sum(keepcoord)==0:
            print("All of these coordinates' cutouts have already been saved!")
            return()
        elif sum(keepcoord)< len(coordlist):
            print(f"Some of these coordinates' cutouts have already been saved! Downloading cutouts for {sum(keepcoord)} sources.")
        coordlist = coordlist[keepcoord]
    #if butler is None:
    #    butler = instantiate_butler ()
    downloadfile = setup_hsc_request ( coordlist, savedir=f'{savedir}/hsc', psf_file=False, half_size=half_size, filetype=filetype, bands=bands, **kwargs)
    downloadfile_psf = setup_hsc_request ( coordlist, savedir=f'{savedir}/hsc', psf_centered=psf_centered, psf_file=True, half_size=half_size,
                                          filetype='coadd', bands=bands, **kwargs)
    download_hsccutouts ( downloadfile, downloadfile_psf, f'{savedir}/hsc', username=hsc_username, passwd=hsc_passwd)

    if mvfromsubdir:
        clean_hsc_subdirs (savedir, half_size, retrim)
    if rename_psf:
        do_rename_psf(coordlist, savedir, bands)
        
def clean_hsc_subdirs (savedir, half_size, retrim):
    archdir = [i for i in glob.iglob(os.path.join(savedir, "hsc/arch*")) if os.path.isdir(i)]
    for ad in archdir:
        files = glob.glob(os.path.join(ad, "*"))
        for idx, current_file in enumerate(files):
            new_file = os.path.join(savedir, "hsc", current_file.split("/")[-1])
            
            os.rename ( current_file, new_file )
                    
            if retrim:
                exitcode, exitmessage = retrim_hsc ( new_file, half_size )
                if exitcode > 0:
                    print(f'{new_file} has no WCS!')
                    os.remove(new_file)
                    
        os.rmdir(ad)    

def retrim_hsc ( new_file, half_size, ):
    hsc = fits.open(new_file)
    name = new_file.split('/')[-1].split("_")[0]
    cname = f'{name[1:3]}h{name[3:5]}m{name[5:10]}s {name[10:13]}d{name[13:15]}m{name[15:]}s'
    center = coordinates.SkyCoord ( cname )
    hsc_wcs = wcs.WCS ( hsc[1].header )
    if not hsc_wcs.has_celestial:
        return 1, 'WCS has no celestial coordinates'
    
    newhdulist = fits.HDUList ()
    newhdulist.append ( hsc[0] )
    labels = ['','IMAGE','MASK','VARIANCE']
    for idx in range(1, len(hsc)):
        try:
            cutout = Cutout2D(hsc[idx].data, center, 2.*half_size, wcs=hsc_wcs )
            imhdu = fits.ImageHDU ( data=cutout.data, header=cutout.wcs.to_header(), name=labels[idx] )
        except (TypeError, ValueError, MemoryError):
            imhdu = fits.ImageHDU ( data=None, header=hsc_wcs.to_header(), name=labels[idx] )
        newhdulist.append(imhdu)    
    
    newhdulist.writeto(new_file, overwrite=True)
    return 0, 'Completed'
    
def do_rename_psf(coordlist, savedir, bands):
    # do this instead of glob - much faster
    all_files = os.listdir(os.path.join(savedir, 'hsc'))
    filename_psf_new = lambda band, cname: os.path.join(savedir, "hsc", f"{cname}_HSC-{band.lower()}_psf.fits")
    for c in coordlist:
        ra, dec = c
        sc = coordinates.SkyCoord (ra, dec, unit='deg' )
        cname = conventions.produce_merianobjectname(skycoordobj=sc)
        fnames = [f.split('/')[-1].replace(".", r'\.').replace("*", ".*") for f in hscpsf_filename_original ('X', ra, dec, savedir)]
        for band in bands:
            old_file = np.concatenate([[f for f in all_files if re.compile(fname.replace('X', band.upper())).match(f)] for fname in fnames])
            if len(old_file) > 0:
                old_file = old_file[0]
                if not os.path.exists(os.path.join(savedir, 'hsc', old_file)):
                    print(f"No old file found for {cname} in {band} band")
                    continue
                new_file = filename_psf_new(band, cname)
                os.rename(os.path.join(savedir, 'hsc', old_file), new_file)
                print(f"Renamed {old_file.split('/')[-1]} to {new_file.split('/')[-1]}")
            else:
                print(f"No old file found for {cname} in {band} band")

def hscpsf_filename_original (band, ra, dec, savedir):
        ra_trunc = str(ra).split(".")
        ra_trunc = ra_trunc[0] + "." + ra_trunc[1][:2]

        dec_trunc = str(dec).split(".")
        dec_trunc = dec_trunc[0] + "." + dec_trunc[1][:2]

        fname_trunc = os.path.join(savedir, "hsc", f"*{band.upper()}*{ra_trunc}*{dec_trunc}*")
        fname_round = os.path.join(savedir, "hsc", f"*{band.upper()}*{ra:.2f}*{dec:.2f}*")
        if fname_trunc != fname_round:
            return(fname_trunc, fname_round)
        else:
            return([fname_trunc])
    
def hsc_images_already_downloaded(coord, savedir, bands):
    ra, dec = coord
    sc = coordinates.SkyCoord (ra, dec, unit='deg' )
    cname = conventions.produce_merianobjectname(skycoordobj=sc)

    # filename_cutout = lambda band: os.path.join(savedir, f"hsc/hsc_{band.lower()}/image", f"{cname}_HSC-{band}.fits")
    filename_cutout = lambda band: os.path.join(savedir, f"hsc", f"{cname}_HSC-{band}.fits")
    cutout_exists = np.array([os.path.isfile(filename_cutout(band)) for band in bands])

    if not np.all(cutout_exists):
        return False

    # filename_psf_new = lambda band, cname: os.path.join(savedir, f"hsc/hsc_{band.lower()}/psf/", f"{cname}_HSC-{band.lower()}_psf.fits")
    filename_psf_new = lambda band, cname: os.path.join(savedir, f"hsc/", f"{cname}_HSC-{band.lower()}_psf.fits")
    psf_new_exists = np.array([os.path.isfile(filename_psf_new(band, cname)) for band in bands])
    return(np.all(psf_new_exists))

    # much faster to only look for new psf file names - might want to add this back as an option?
    # psf_exists = np.array([np.any([len(glob.glob(i))>0 for i in hscpsf_filename_original (band, ra, dec, savedir)]) for band in bands])
    # return(np.all(psf_exists) | np.all(psf_new_exists))

def fetch_merian(coordlist, savedir, butler=None, overwrite=False,
                 *args, **kwargs ):
    if not os.path.exists(f'{savedir}/merian'):
        os.makedirs ( f'{savedir}/merian/' )
    
    if isinstance(coordlist, str): # \\ if we provide a file, read it in. Else assume that the array is the coordinates
        coordlist = np.genfromtxt (coordlist)
        if len(coordlist.shape) == 1: 
            coordlist = coordlist.reshape(1,-1)
            
        
    if not overwrite:
        keepcoord = [not merian_images_already_downloaded(c, savedir) for c in coordlist]
        if sum(keepcoord)==0:
            print("All of these coordinates' cutouts have already been saved!")
            return()
        elif sum(keepcoord)< len(coordlist):
            print(f"Some of these coordinates' cutouts have already been saved! Downloading cutouts for {sum(keepcoord)} sources.")
        coordlist = coordlist[keepcoord]
    if butler is None:
        butler = instantiate_butler ()

    pull_merian_cutouts ( coordlist, butler, savedir=f'{savedir}/merian/', **kwargs)

def merian_images_already_downloaded(coord, savedir):
    ra, dec = coord
    #if isinstance(ra, float):
    #    ra = [ra]
    #    dec = [dec]
    sc = coordinates.SkyCoord (ra, dec, unit='deg' )
    cname = conventions.produce_merianobjectname(skycoordobj=sc)

    filename_cutout = lambda band: os.path.join(savedir, "merian", f"{cname}_{band}_merim.fits")
    cutout_exists = np.array([os.path.isfile(filename_cutout(band)) for band in ["N540", "N708"]])

    filename_psf = lambda band: os.path.join(savedir, "merian", f"{cname}_{band}_merpsf.fits")
    psf_exists = np.array([os.path.isfile(filename_psf(band)) for band in ["N540", "N708"]])

    return(np.all(cutout_exists) & np.all(psf_exists))
