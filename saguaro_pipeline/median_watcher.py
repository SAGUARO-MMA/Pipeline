#!/usr/bin/env python

"""
Script to create median images from the 4 CSS images per field.
"""

__version__ = "2.1.2"  # last updated 2023-11-01

import argparse
import numpy as np
import os
import datetime
import time
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import warnings
import glob
import subprocess
import gc
import uuid
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fnmatch as fn
from . import css, saguaro_logging, util
import shutil
from importlib_resources import files

warnings.simplefilter('ignore', category=AstropyWarning)
gc.enable()


def check_field(event):
    """
    Check if this field has already been processed.
    """
    try:
        try:
            file = str(event.dest_path)
        except AttributeError:
            file = str(event.src_path)  # get name of new file
        if fn.fnmatch(os.path.basename(file), 'G96_*.calb.fz'):  # only continue if event is a fits file
            util.copying(file)  # check to see if write is finished writing
    except AttributeError:  # if event is a file
        file = event
    field = file.split('_2B_')[1].split('_00')[0]
    if field not in field_list:
        field_list.append(field)
        if 'N' in field or 'S' in field:
            return field


def action(event, date, read_path, write_path, field):
    """
    Waits for all 4 images (max time 5 mins) to create a median image.
    """
    t1 = datetime.datetime.utcnow()
    while True:
        images = sorted(glob.glob(read_path + '/G96_*' + field + '*.calb.fz'))
        if len(images) == 4:
            break
        else:
            t2 = datetime.datetime.utcnow()
            if (t2 - t1).total_seconds() > 1800:
                break
            else:
                time.sleep(1)
    t1 = datetime.datetime.utcnow()
    while True:
        headers = sorted(glob.glob(read_path + '/G96_*' + field + '*.arch_h'))
        if len(headers) == len(images):
            break
        else:
            t2 = datetime.datetime.utcnow()
            if (t2 - t1).total_seconds() > 300:
                break
            else:
                time.sleep(1)
    logger.info('Number of files for field = ' + str(len(images)))
    combine = []
    mjd = []
    back = []
    global bad_images
    out_file = os.path.basename(images[0]).split('_00')[0] + '_med.fits'
    unique_dir = f'{css.work_path(date)}/{uuid.uuid1().hex}/'
    os.makedirs(unique_dir)
    for i, f in enumerate(images):
        subprocess.call(['cp', f, unique_dir])
        header_filename = f.replace('calb.fz', 'arch_h')
        if not os.path.exists(header_filename):
            logger.error('No header available for ' + f)
            continue
        header = fits.open(header_filename)[0].header
        header['CTYPE1'] = 'RA---TPV'
        header['CTYPE2'] = 'DEC--TPV'
        c = unique_dir + os.path.basename(f)
        try:
            with fits.open(c) as hdr:
                hdr.verify('fix+ignore')
                data = hdr[1].data
        except:
            logger.critical('Error opening file ' + f)
        hdul = fits.HDUList([fits.PrimaryHDU(data, header)])
        hdul.writeto(c.replace('.calb.fz', '.fits'), output_verify='fix+ignore')
        stars = header.get('WCSMATCH', 0)  # check image type
        t = header['MJD']
        if stars > 100:  # only continue if science image
            combine.append(c.replace('.calb.fz', '.fits'))
            mjd.append(t)
            back.append(np.median(data))
        else:
            bad_images += 1
        with fits.open(css.bad_pixel_mask()) as bpm_hdr:
            mask_header = bpm_hdr[0].header
            data = bpm_hdr[0].data
        fits.writeto(c.replace('.calb.fz', '_mask.fits'), data, mask_header + header,
                     output_verify='fix+ignore')
        with open(unique_dir + out_file.replace('.fits', '.head'), 'w') as swarp_head:
            for card in header.cards:
                swarp_head.write(str(card) + '\n')
        shutil.copy(unique_dir + out_file.replace('.fits', '.head'),
                    unique_dir + out_file.replace('.fits', '_mask.head'))
    if len(combine) > 1:
        masks = [x.replace('.fits', '_mask.fits') for x in combine]
        swarp_config_file = str(files('zogy').joinpath('Config/swarp_css.config'))
        subprocess.call(['swarp'] + combine + ['-c', swarp_config_file, '-IMAGE_SIZE', '5280,5280',
                                               '-IMAGEOUT_NAME', unique_dir + out_file, '-SUBTRACT_BACK', 'YES',
                                               '-GAIN_KEYWORD', 'GAIN', '-BACK_SIZE', '256', '-BACK_FILTERSIZE', '3',
                                               '-FSCALASTRO_TYPE', 'VARIABLE', '-FSCALE_KEYWORD', 'FLXSCALE',
                                               '-VERBOSE_TYPE', 'LOG'])
        subprocess.call(['swarp'] + masks + ['-c', swarp_config_file, '-IMAGE_SIZE', '5280,5280',
                                             '-IMAGEOUT_NAME', unique_dir + out_file.replace('.fits', '_mask.fits'),
                                             '-SUBTRACT_BACK', 'NO', '-GAIN_DEFAULT', '1', '-COMBINE_TYPE', 'SUM',
                                             '-VERBOSE_TYPE', 'LOG'])
        with fits.open(unique_dir + out_file, mode='update') as hdr:
            header_swarp = hdr[0].header
            data = hdr[0].data
            data /= header_swarp['FLXSCALE']
            data += np.median(back)
            header_swarp['EXPTIME'] = 30
            header_swarp['RDNOISE'] = 11.6 / np.sqrt(len(combine))
            header_swarp['GAIN'] = 3.1
            try:
                del header_swarp['CROTA1']
                del header_swarp['CROTA2']
            except KeyError:
                pass
            header_swarp['MJD'] = mjd[0] + ((mjd[-1] - mjd[0]) / 2)
            header_swarp['NCOMBINE'] = (len(combine), 'Number of files used to create median.')
            for i, c in enumerate(combine):
                header_swarp['FILE' + str(i + 1)] = c
        with fits.open(unique_dir + out_file.replace('.fits', '_mask.fits')) as mask_hdr:
            mask_header = mask_hdr[0].header
            mask_data = mask_hdr[0].data
        mask_data = (mask_data + 0.5).astype(np.uint8)
        mask_data[mask_data != 0] = 1
        fits.writeto(unique_dir + out_file.replace('.fits', '_mask.fits'), mask_data, mask_header, overwrite=True)
        subprocess.call(['fpack', '-D', '-Y', '-g', '-q0', '16', unique_dir + out_file])
        subprocess.call(['fpack', '-D', '-Y', '-g', unique_dir + out_file.replace('.fits', '_mask.fits')])
        subprocess.call(['mv', unique_dir + out_file + '.fz', write_path])
        subprocess.call(['mv', unique_dir + out_file.replace('.fits', '_mask.fits') + '.fz', write_path])
        os.chdir(os.environ['SAGUARO_ROOT'])
        subprocess.call(['rm', '-r', unique_dir])
        print('Created ' + out_file)
    ncombine.append(len(combine))
    logger.info('Number of files used in median = ' + str(len(combine)))


class FileWatcher(FileSystemEventHandler, object):
    """
    Monitors directory for new files.
    """

    def __init__(self, date, read_path, write_path):  # parameters needed for action
        self._date = date
        self._read_path = read_path
        self._write_path = write_path
        self._field_list = field_list

    def on_created(self, event):
        """
        Action to take for new files.
        """
        field = check_field(event)
        if field:
            logger.info('Found field ' + field)
            action(event, self._date, self._read_path, self._write_path, field)

    def on_moved(self, event):
        """
        Action to take for renamed files.
        """
        field = check_field(event)
        if field:
            action(event, self._date, self._read_path, self._write_path, field)


def cli():
    global field_list, ncombine, logger, bad_images, missing_head

    t0 = datetime.datetime.utcnow()

    params = argparse.ArgumentParser(description='User parameters.')
    params.add_argument('--date', default=None, help='Date of files to process.')  # optional date argument
    args = params.parse_args()

    if args.date:
        date = args.date
        submit_all = True
    else:
        date = datetime.datetime.utcnow().strftime('%Y/%m/%d')
        submit_all = False

    log_file_name = f'{css.log_path()}/median_watcher_' + datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    logger = saguaro_logging.initialize_logger(log_file_name)

    logger.critical('Median watcher started.')

    read_path = css.incoming_path(date)
    read_dir = False
    while read_dir is False:
        if not os.path.exists(read_path):
            print(f'waiting for directory {read_path} to be created...')
            done = util.scheduled_exit(t0, css)
            if done:
                logger.critical('Scheduled time reached. No data ingested.')
                logger.shutdown()
                return
            else:
                time.sleep(1)
        else:
            read_dir = True
    write_path = css.read_path(date)  # median watcher writes the stacked images to the pipeline's read_path
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    field_list = []
    ncombine = []
    bad_images = 0
    missing_head = 0

    if submit_all:  # to rerun all data
        images = glob.glob(read_path + '/G96*.calb.fz')
        for f in images:
            field = check_field(f)
            if field:
                action(f, date, read_path, write_path, field)
    else:  # normal real-time reduction
        observer = Observer()
        observer.schedule(FileWatcher(date, read_path, write_path), read_path, recursive=False)
        observer.start()
        while True:
            done = util.scheduled_exit(t0, css)
            if done:
                observer.stop()
                observer.join()
                break
            else:
                time.sleep(1)

    ncombine = np.array(ncombine)
    files_raw = glob.glob(read_path + '/G96*_[NS]*.calb.fz')
    files_head = glob.glob(read_path + '/G96*_[NS]*.arch_h')
    logger.critical(f'''Scheduled time reached. Median watcher summary:
    Received {len(files_raw):d} calibrated images and {len(files_head):d} header files.
    {len(ncombine):d} fields observed,
    {np.sum(ncombine == 4):d} medians made with 4 images,
    {np.sum(ncombine == 3):d} medians made with 3 images,
    {np.sum(ncombine == 2):d} medians made with 2 images,
    {np.sum(ncombine == 1):d} medians made with 1 image,
    {np.sum(ncombine == 0):d} medians not made,
    {bad_images:d} images not used due to bad weather.
    ''')
    logger.shutdown()
