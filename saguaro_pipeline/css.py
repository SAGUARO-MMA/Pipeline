#!/usr/bin/env python

"""
Telescope setting file for CSS 1.5m Mt Lemmon telescope.
"""

<<<<<<< Updated upstream
__version__ = "2.1.7"  # last updated 2024-04-23
=======
__version__ = "2.2.1"  # last updated 2024-11-04
>>>>>>> Stashed changes

import datetime
import gc
import glob
import os
import subprocess
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from . import util
from importlib_resources import files

warnings.simplefilter('ignore', category=AstropyWarning)
gc.enable()


def incoming_path(date):
    """
    Returns the absolute path containing the incoming raw data from this telescope.
    """
    data_date = datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%y%b%d')
    return f'{write_path()}/inc/{data_date}'


def read_path(date):
    """
    Returns the absolute path where raw files are stored.
    """
    read_date = datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%Y/%y%b%d')
    return f'{write_path()}/raw/{read_date}'


def write_path():
    """
    Returns the absolute path containing all the data products for this telescope.
    """
    return f'{os.environ["SAGUARO_ROOT"]}/data/css'


def work_path(date):
    """
    Returns the absolute path to the working directory for the pipeline.
    """
    return f'{write_path()}/tmp/{date}'


def log_path():
    """
    Returns the absolute path where log files are written to.
    """
    return f'{write_path()}/log'


def red_path(date):
    """
    Returns the absolute path where reduced science images are written to.
    """
    return f'{write_path()}/red/{date}'


def ref_path(field_id):
    """
    Returns the absolute path where reference images are stored.
    """
    return f'{write_path()}/ref/{field_id}'


def file_name():
    """
    Returns name pattern of input files.
    """
    return 'G96_*_med.fits.fz'


def create_mask():
    """
    Returns if a mask should be created for each image.
    """
    return True


def bad_pixel_mask():
    """
    Returns the full path of the bad pixel mask.
    """
    return files('saguaro_pipeline').joinpath('css_bpm.fits')


def fieldID(header):
    """
    Returns field ID of image.
    """
    fieldID = header['OBJECT']
    return fieldID


def saturation():
    """
    Returns the saturation level in electrons.
    """
    return 57000


def binning():
    """
    Returns the image binning used during the determination of the satellite trail mask.
    """
    return 2


def mask_edge_pixels(data, mask):
    """
    Returns the image with edge pixels masked.
    """
    return data


def mask_bp():
    """
    Returns if bad pixels were masked.
    """
    return 'F'


def mask_cr():
    """
    Returns if cosmic rays were masked.
    """
    return 'T'


def mask_sp():
    """
    Returns if saturated pixels were masked.
    """
    return 'T'


def mask_scp():
    """
    Returns if pixels connected to saturated pixles were masked.
    """
    return 'T'


def mask_sat():
    """
    Returns if satellite trails were masked.
    """
    return 'F'


def mask_ep():
    """
    Returns if edge pixels were masked.
    """
    return 'T'


def cosmic():
    """
    Returns if cosmic ray correction should be done.
    """
    return 'F'


def sat():
    """
    Returns if satellite trail correction should be done."""
    return 'T'


def sat_readnoise(header):
    """
    Returns the readnoise value from the header.
    """
    return header['RDNOISE']


def sat_gain(header):
    """
    Returns the gain value from the header.
    """
    return header['GAIN']


def sat_objlim():
    """
    Returns the value used in the mask creation for determining saturated stars.
    """
    return 20


def sat_sigclip():
    """
    Returns the vaule used for sigma clipping in the satellite trail fitting.
    """
    return 6


def sat_repeat():
    """
    Return how many times the satellite fitting should be done.
    """
    return 0


def time_zone():
    """
    Returns the time zone of the pipeline.
    """
    return -7


def tel_zone():
    """
    Returns the adjustment to the time zone for the telescope.
    """
    return 0


def tel_delta():
    """
    Returns the adjustment to the time zone for the telescope.
    """
    return 7


def stop_hour():
    """
    Returns the UTC hour at which the pipeline will stop for this telescope.
    """
    return 15


def output(f, date):
    """
    Return the full path of the output used to determine if a file has been processed.
    """
    return os.path.exists(red_path(date) + os.path.basename(f.replace('.fits', '_trans.fits')))


def find_ref(reduced):
    """
    Function to find a reference for image subtraction.
    """
    with fits.open(reduced) as hdr:
        header = hdr[0].header
    field = fieldID(header)
    ref_files = glob.glob(ref_path(field) + '/*')
    for f in ref_files:
        subprocess.call(['cp', f, '.'])
        if '.fz' in f:
            util.funpack_file(os.path.basename(f))
    ref_file = field + '_wcs.fits'
    if os.path.exists(ref_file):
        return ref_file
    else:
        return ''
