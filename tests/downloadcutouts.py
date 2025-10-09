"""
Script to download Merian and/or HSC cutouts for a list of coordinates. 
Cannot be run on compute nodes because of butler - needs to be run on login node.
Recommended to use screen to avoid disconnection. 

Need to do this first:
    source /scratch/gpfs/LSST/stack/loadLSST.sh
    setup lsst_distrib

Then for example:
python ~/Merian/Metallicity/downloadcutouts.py --coordfile=/scratch/gpfs/JENNYG/abbymintz/Merian/Metallicity/desi_match_coords.txt --savedir=/scratch/gpfs/JENNYG/abbymintz/Merian/Metallicity/cutouts/ --imagetype=hsc --hscfilts=griz --halfsize=15"""

import os, sys
sys.path.append('/home/am2907/Merian/carpenter/src')
from carpenter import handler
from carpenter.conventions import produce_merianobjectname
import numpy as np
from astropy import coordinates
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import make_lupton_rgb
import argparse

cfile ="carpenter/tests/coordlist_example.txt"
savedir = 'savedir_example'
imagetype = "merian"

password = "your_hsc_password"
username = "your_hsc_username"

butler = handler.instantiate_butler (repo='/scratch/gpfs/MERIAN/repo/repos/main_2022_12_19', collections='DECam/runs/merian/dr1_wide')

parser = argparse.ArgumentParser(description="Process a coordinate file and save directory.")
parser.add_argument("--coordfile", type=str, default=cfile, help="Path to the coordinate file")
parser.add_argument("--savedir", type=str, default=savedir, help="Directory to save results")
parser.add_argument("--imagetype", type=str, default=imagetype, help="hsc or merian images")
parser.add_argument("--halfsize", type=str, default=30, help="Half size of cutout in arcsec (default: 30)")
parser.add_argument("--hscfilts", type=str, default='grizy', help="HSC filters to use (default: grizy)")
args = parser.parse_args()

cfile = args.coordfile
savedir = args.savedir
imagetype = args.imagetype
halfsize = args.halfsize
hscfilts = args.hscfilts

halfsize = float(halfsize) * u.arcsec

print(f"\nDownloading {imagetype} cutouts from {cfile} to {savedir}.\n")

if imagetype == "merian":
    handler.fetch_merian(cfile, savedir, butler=butler, half_size=halfsize)

elif imagetype == "hsc":
    handler.fetch_hsc(cfile, savedir, hsc_username=username, hsc_passwd=password, half_size=halfsize, bands=hscfilts)
elif imagetype == "both":
    handler.fetch_merian(cfile, savedir, butler=butler, half_size=halfsize)
    handler.fetch_hsc(cfile, savedir, hsc_username=username, hsc_passwd=password, half_size=halfsize, bands=hscfilts)