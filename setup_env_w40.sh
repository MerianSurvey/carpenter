#!/bin/sh

# Setup hscPipe enviroment
LSST_CONDA_ENV_NAME=lsst-scipipe-4.1.0
source /projects/HSC/LSST/stack/loadLSST.bash

setup lsst_apps
setup lsst_distrib -t w_2022_40
setup obs_subaru

echo "LOAD ENVIRONMENT LSSTPIPE-4.1.0"