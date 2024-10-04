#!/usr/bin/env python

"""
Pipeline for real-time data reduction and image subtraction.
"""

__version__ = "2.2.0"  # last updated 2024-10-04

import argparse
import datetime
import glob
import importlib
import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool, Manager
import astroscrappy
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from acstools.satdet import detsat, make_mask
import gc
import uuid
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fnmatch as fn
from zogy import zogy
from . import ingestion, saguaro_logging, util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', category=AstropyWarning)
gc.enable()


def cleanup(file, ref, unique_dir):
    """
    Moves relevant files from the tmp directory to the output path.
    """
    os.chdir(unique_dir)
    with fits.open(file) as hdr:
        header = hdr[0].header
    fieldID = tel.fieldID(header)
    print('fieldID: ', fieldID)
    print(red_path)
    if ref:
        cp_dir = red_path
        cp_files = ['_wcs.fits', '_mask.fits', '_trans.fits', '_Scorr.fits']
    else:
        file = fieldID
        cp_dir = tel.ref_path(fieldID)
        cp_files = ['_wcs.fits', '_bkg.fits', '_bkg_std.fits', '_cat.fits', '_ldac.fits', '_psf.fits', '_psfex.cat',
                    '.log']

    for ext in cp_files:
        f = file.replace('.fits', '') + ext
        if '.fits' in ext:
            print(f)
            f = fpack_file(f)
            print(f)
        #        q.put(logger.info('mv ',f,' ',cp_dir))  ##didn't pass logger?
        print(['mv', f, cp_dir])
        subprocess.call(['mv', f, cp_dir])
    extra = glob.glob(file.replace('.fits', '') + '*.pdf')
    extra += glob.glob(file.replace('.fits', '') + '*.reg')
    for x in extra:
        subprocess.call(['mv', x, cp_dir])
    return


def fpack_file(file):
    """
    Fpack file and return new name.
    """
    print(['fpack', '-D', '-Y', '-g', '-q', '0', file])
    try:
        if os.path.exists(file):
            subprocess.call(['fpack', '-D', '-Y', '-g', '-q', '0', file])
        else:
            print(file, " doesn't exist")
    except:
        print('fpack failed')
    return file.replace('.fits', '.fits.fz')


def mask_create(science_file, tel, unique_dir, Red, mask_bp, header, comment, log_file_name):
    """
    Creates a mask file for an image.
    """
    mask_infnan = ~np.isfinite(Red)
    mask_bp[mask_infnan & (mask_bp != 32)] = 1
    Red[mask_infnan] = 0
    Red = tel.mask_edge_pixels(Red, mask_bp)
    mask_sat = np.zeros((np.shape(Red)[0], np.shape(Red)[1])).astype(np.bool)  # create empty mask
    astroscrappy.update_mask(Red, mask_sat, tel.saturation(), True)  # add saturated stars to mask
    mask_sat = mask_sat.astype(np.uint8)  # set saturated star mask type
    mask_sat[Red >= tel.saturation()] = 4  # set saturated pixel flag
    mask_sat[mask_sat == 1] = 8  # set connected to saturated pixel flag
    header['COSMIC-P'] = (tel.cosmic(), 'Corrected for cosmic rays?')
    if tel.cosmic() == 'T':
        # clean science image of cosmic rays and create cosmic ray mask
        mask_cr, clean = astroscrappy.detect_cosmics(Red, inmask=(masks[i] + mask_sat).astype(np.bool),
                                                     sigclip=tel.sat_sigclip(), readnoise=tel.sat_readnoise(header),
                                                     gain=tel.sat_gain(header), satlevel=tel.saturation(),
                                                     objlim=tel.sat_objlim())
        mask_cr = mask_cr.astype(np.uint8)  # set cosmic ray mask type
        mask_cr[mask_cr == 1] = 2  # set cosmic ray flag
    else:
        mask_cr = np.zeros((np.shape(Red)[0], np.shape(Red)[1])).astype(np.uint8)
        clean = Red
    binning = tel.binning()
    binned_data = clean.reshape(np.shape(clean)[0] // binning, binning,
                                np.shape(clean)[1] // binning, binning).sum(3).sum(1)  # bin data
    header['SAT-P'] = (tel.sat(), 'Corrected for satellite trails?')
    if tel.sat() == 'T':
        satellite_fitting = False
        for j in range(tel.sat_repeat()):
            fits.PrimaryHDU(binned_data).writeto(unique_dir + '/binned_mask.fits', overwrite=True)  # write to tmp file
            results, errors = detsat(unique_dir + '/binned_mask.fits', chips=[0], n_processes=1, buf=40, sigma=3,
                                     h_thresh=0.2)  # detect sateliite trails
            trail_coords = results[(unique_dir + '/binned_mask.fits', 0)]  # create satellite trail if found
            if len(trail_coords) > 0:  # continue if satellite trail found
                trail_segment = trail_coords[0]
                try:
                    mask_binned = make_mask(unique_dir + '/binned_mask.fits', 0, trail_segment, sublen=5, pad=0,
                                            sigma=5, subwidth=5000).astype(np.uint8)  # create satellite trail mask
                except ValueError:  # if error occurs, add comment
                    comment += f'Satellite trail could not be fitted for file {science_file} ' \
                               f'and is not included in the mask.'
                satellite_fitting = True
                binned_data[mask_binned == 1] = np.median(binned_data)
                try:
                    open_old_mask = fits.open(unique_dir + '/old_mask.fits')
                    old_mask = open_old_mask[0].data
                    open_old_mask.close()
                    mask_binned = old_mask + mask_binned
                except IOError:
                    pass
                fits.writeto(unique_dir + '/old_mask.fits', mask_binned, overwrite=True)
            else:
                break
        if satellite_fitting:
            mask_sate = np.kron(mask_binned, np.ones((binning, binning))).astype(np.uint8)  # unbin mask
            mask_sate[mask_sate == 1] = 16  # set satellite trail flag
        else:  # if no satellite trails are found, create empty mask
            mask_sate = (np.zeros([np.shape(clean)[0], np.shape(clean)[1]])).astype(np.uint8)
    else:
        mask_sate = (np.zeros([np.shape(clean)[0], np.shape(clean)[1]])).astype(np.uint8)
    mask = mask_bp + mask_cr + mask_sat + mask_sate  # combine bad pixel, cosmic ray, saturated star, satellite masks
    mask_name = science_file.replace('.fits', '_mask.fits').replace('.arch', '_mask.fits').replace('.fz', '')
    mask_hdu = fits.PrimaryHDU(mask)  # create mask Primary HDU
    mask_hdu.header['LOG'] = (log_file_name + '.log', 'Name of log file.')  # name of log file
    mask_hdu.header['USE'] = ('Complex mask using additive flags.', 'e.g. 6 = 2 + 4')  # header comment
    mask_hdu.header['M-BP'] = (tel.mask_bp(), 'Bad pixels included in mask?')
    mask_hdu.header['M-BPVAL'] = (1, 'Value of masked bad pixels.')
    mask_hdu.header['M-BPNUM'] = (np.sum(mask & 1 == 1), 'Number of bad pixels.')
    mask_hdu.header['M-CR'] = (tel.mask_cr(), 'Cosmic ray pixels included in mask?')
    mask_hdu.header['M-CRVAL'] = (2, 'Value of masked cosmic ray pixels.')
    mask_hdu.header['M-CRNUM'] = (np.sum(mask & 2 == 2), 'Number of cosmic ray pixels.')
    mask_hdu.header['SATURATE'] = (tel.saturation(), 'Level of saturation.')
    mask_hdu.header['M-SP'] = (tel.mask_sp(), 'Saturated pixels included in mask?')
    mask_hdu.header['M-SPVAL'] = (4, 'Value of masked saturated pixels.')
    mask_hdu.header['M-SPNUM'] = (np.sum(mask & 4 == 4), 'Number of saturated pixels.')
    mask_hdu.header['M-CSP'] = (tel.mask_scp(), 'Saturated-connected pixels included in mask?')
    mask_hdu.header['M-CSPVAL'] = (8, 'Value of masked saturated-connected pixels.')
    mask_hdu.header['M-CSPNUM'] = (np.sum(mask & 8 == 8), 'Number of saturated-connected pixels.')
    mask_hdu.header['M-SAT'] = (tel.mask_sat(), 'Satellite trail pixels included in mask?')
    mask_hdu.header['M-SATVAL'] = (16, 'Value of masked satellite trail pixels.')
    mask_hdu.header['M-SATNUM'] = (np.sum(mask & 16 == 16), 'Number of satellite trail pixels.')
    mask_hdu.header['M-EP'] = (tel.mask_ep(), 'Edge pixels included in mask?')
    mask_hdu.header['M-EPVAL'] = (32, 'Value of masked Edge pixels.')
    mask_hdu.header['M-EPNUM'] = (np.sum(mask & 32 == 32), 'Number of Edge pixels.')
    mask_hdu.writeto(mask_name, overwrite=True)  # write mask to file
    return Red, header, comment


def science_process(science_file, unique_dir, log_file_name):
    """
    Function to process science images. As CSS data has already been processed, so only a mask is created.
    """
    subprocess.call(['cp', science_file, science_file.replace('.fits', '_mask.fits'), '.'])
    science_file = util.funpack_file(os.path.basename(science_file))
    mask_file = util.funpack_file(os.path.basename(science_file).replace('.fits', '_mask.fits'))
    with fits.open(science_file) as hdr:
        Red = hdr[0].data
        header = hdr[0].header
    comment = 'No reduction needed. Creating mask. '
    with fits.open(mask_file) as hdr:
        mask_bp = hdr[0].data
    Red, header, comment = mask_create(science_file, tel, unique_dir, Red, mask_bp, header, comment, log_file_name)
    fits.writeto(science_file, Red, header, overwrite=True)
    return science_file, comment


def action(item_list):
    """
    Action to take for each file. Defines the main structure of the pipeline.
    """
    try:
        item = item_list.get(True)  # get parameters for list
    except AttributeError:
        item = item_list
    event = item[0]
    telescope = item[1]
    try:
        try:
            file = str(event.dest_path)
        except AttributeError:
            file = str(event.src_path)  # get name of new file
        q.put(logger.info('Found new file ' + file))
    except AttributeError:  # if event is a file
        file = event
        q.put(logger.info('Found old file ' + file))
    if fn.fnmatch(os.path.basename(file), file_name):  # only continue if the file matches the expected file name
        util.copying(file)  # check to see if write is finished writing
    else:
        return
    unique_dir = work_path + '/' + uuid.uuid1().hex  # create a unique tmp directoty to work in
    os.mkdir(unique_dir)
    os.chdir(unique_dir)
    q.put(logger.info('Starting reduction for '+file))
    reduced, comment = science_process(file, unique_dir, log_file_name)  # submit image for reduction
    q.put(logger.info('Ending reduction for '+file+' '+comment))
    ref = tel.find_ref(reduced)  # find reference image
    try:
        if ref:  # submit as subtraction job
            q.put(logger.info("Reference found. Starting zogy subtraction for "+reduced))
            status, comment = zogy.optimal_subtraction(new_fits=reduced,
                                                       ref_fits=ref,
                                                       new_fits_mask=reduced.replace('.fits', '_mask.fits'),
                                                       ref_fits_mask=ref.replace('_wcs.fits', '_mask.fits'),
                                                       telescope=telescope, log=logger, nthread=2)
        else:  # submit image to create reference
            q.put(logger.info("No reference found. Starting zogy reference creation for "+reduced))
            status, comment = zogy.optimal_subtraction(ref_fits=reduced,
                                                       ref_fits_mask=reduced.replace('.fits', '_mask.fits'),
                                                       telescope=telescope, log=logger, nthread=2)
        if status == 'info':
            q.put(logger.info('ZOGY comment: '+comment))
    except BaseException as e:
        q.put(logger.critical('Uncaught error occurred in ZOGY: '+reduced+' - ' + str(e)))
        q.put(logger.error(''.join(traceback.format_exception(type(e), e, e.__traceback__))))
    transient_catalog = reduced.replace('.fits', '_trans.fits')
    q.put(logger.info(unique_dir + '/' + transient_catalog))
    if os.path.exists(unique_dir + '/' + transient_catalog):
        q.put(logger.info('Starting ingestion for '+reduced))
        try:
            ingestion.ingestion(unique_dir + '/' + transient_catalog, logger)
        except BaseException as e:
            q.put(logger.error('Ingestion failed  for '+reduced+' - '+str(e)))
            #q.put(logger.error(e))
        q.put(logger.info('End ingestion for '+reduced))
    else:
        q.put(logger.error('Failed to find transient catalog for '+reduced))
    q.put(logger.info('Starting cleanup for '+reduced))
    q.put(logger.info(reduced + ' ' + ref + ' ' + unique_dir))
    #print(reduced, ref, unique_dir)
    cleanup(reduced, ref, unique_dir)
    q.put(logger.info('End cleanup for '+reduced))
    os.chdir(work_path)
    subprocess.call(['rm', '-r', unique_dir])


class FileWatcher(FileSystemEventHandler, object):
    """
    Monitors directory for new files.
    """

    def __init__(self, queue, telescope):  # parameters needed for action
        self._queue = queue
        self._telescope = telescope

    def on_created(self, event):
        """
        Add new file to queue.
        """
        self._queue.apply_async(action, [[event, self._telescope]])


def main(telescope=None, date=None, cpu=None):
    """
    Main pipeline function.
    """

    t0 = time.time()

    if telescope is None:  # if no telescope is given, exit with error
        print('No telescope given, please give telescope and re-run.')
        sys.exit(-1)
    else:
        global tel
    try:
        tel = importlib.import_module(f'{__package__}.{telescope}')  # import telescope setting file
    except ImportError:
        print('No such telescope file, please check that the file is in the same directory as the pipeline.')
        sys.exit(-1)

    tel_zone = tel.tel_zone()
    tel_delta = tel.tel_delta()
    if date is None:  # if no date is given, run in real-time
        submit_all = False
        date = (datetime.datetime.utcnow() + datetime.timedelta(hours=tel_zone) + datetime.timedelta(hours=tel_delta)
                ).strftime('%Y/%m/%d')
    else:  # if date is given, replace - / .
        submit_all = True
        if '-' in date:
            date = date.replace('-', '/')
        elif '.' in date:
            date = date.replace('.', '/')
        elif '/' in date:
            date = date
        else:
            date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y/%m/%d')

    if cpu is None:  # if no number of CPUs is given, no parallel running of pipeline
        cpu = 1
    else:
        cpu = int(cpu)
        if cpu > multiprocessing.cpu_count():  # if number of CPUs given is greater than system CPUs, set to max CPUs
            cpu = multiprocessing.cpu_count()

    global file_name
    file_name = tel.file_name()
    read_path = tel.read_path(date)  # set read path: where raw data is stored
    read_dir = False
    while read_dir is False:
        if not os.path.exists(read_path):
            try:
                os.makedirs(read_path)
                read_dir = True
            except OSError:
                done = util.scheduled_exit(datetime.datetime.fromtimestamp(t0), tel)  # check if time to exit
                if done:
                    sys.exit()
                else:
                    print(f'waiting for directory {read_path} to be created...')
                    time.sleep(1)
        else:
            read_dir = True

    global work_path, log_path, red_path
    work_path = tel.work_path(date)  # set tmp work path: where data is reduced
    os.makedirs(work_path, exist_ok=True)  # if tmp work path does not exist, make path

    log_path = tel.log_path()  # set logpath: where log is written
    os.makedirs(log_path, exist_ok=True)  # if log path does not exist, make path

    red_path = tel.red_path(date)  # set path where reduced science images are written to
    os.makedirs(red_path, exist_ok=True)  # if path does not exist, make path

    global q, logger, log_file_name
    q = Manager().Queue()  # create queue for logging
    os.chdir(work_path)  # change to workong directory
    log_file_name = f'{log_path}/pipeline_run_{datetime.datetime.utcnow().strftime("%Y%m%d_T%H%M%S")}'
    logger = saguaro_logging.initialize_logger(log_file_name)

    try:
        q.put(logger.info(f'Running pipeline version {__version__}, with setting file version {tel.__version__}.'))
        q.put(logger.info('Pipeline running on ' + str(cpu) + ' CPUs.'))
        if submit_all:  # redo all files for the given date
            pool = Pool(cpu, maxtasksperchild=1)
            files = sorted(glob.glob(read_path + '/' + file_name))  # grab all files
            jobs = []
            for f in files:
                jobs.append(pool.apply_async(action, [[f, telescope]]))  # add waiting files to pool
            pool.close()  # close pool
            pool.join()  # join pool
            for job in jobs:  # wait unitl all science images are finished before exiting
                try:
                    job.get()
                except IOError as e:
                    q.put(logger.error('Job failed due to: ' + str(e)))
            q.put(logger.info(f'Total wall-time spent: {time.time() - t0} s'))
        else:  # reduce data in real time, don't redo files alread reduced
            pool = Pool(cpu, maxtasksperchild=1)  # create pool with given CPUs and queue feeding into action function
            observer = Observer()  # create observer
            observer.schedule(FileWatcher(pool, telescope), read_path, recursive=False)  # setup observer
            files = sorted(glob.glob(read_path + '/' + file_name))  # glob any files already there
            for f in files:  # loop through waiting files
                if tel.output(f, date):
                    q.put(logger.info('Output already exists, skipping image'))
                else:
                    pool.apply_async(action, [[f, telescope]])  # add waiting files to pool
            observer.start()  # start observer
            while True:  # continue to monitor
                done = util.scheduled_exit(datetime.datetime.fromtimestamp(t0), tel)  # check if time to exit
                q.put(logger.info(f'Current time: ' + datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))
                if done:  # if scheduled exit time has been reached, exit pipeline
                    q.put(logger.info('Exiting now'))
                    if pool._cache:
                        q.put(logger.critical(f'Terminating idle processes: {pool._cache}'))

                    observer.stop()  # stop observer
                    observer.join()  # join observer
                    pool.terminate()  # terminate worker processes and close pool
                    pool.join()  # join pool
                    break

                else:  # if scheduled exit time has not reached, continue
                    q.put(logger.info('Continuing to monitor'))
                    time.sleep(60)

        # final summary stats and plot
        input_images = glob.glob(read_path + "/" + file_name)
        output_files = glob.glob(red_path + '/*_trans.fits*')
        candidates = np.array([fits.getval(f, 'T-NTRANS', ext=1) for f in output_files], dtype=int)
        q.put(logger.critical(f'Processed all data taken on the night of {date}:\n'
                              f'    {len(input_images):d} input images found.\n'
                              f'    {len(output_files):d} successfully processed.\n'
                              f'    {candidates.sum():d} candidates extracted.'))
        if np.sum(candidates):
            plt.hist(candidates, bins='auto')
            plt.title('Candidate summary for ' + date)
            plt.xlabel('Number of candidates per field')
            hist_file_name = log_file_name + '.pdf'
            plt.savefig(hist_file_name)
            logger.slack_client.files_upload(channels='pipeline', file=hist_file_name)

        logger.shutdown()
        sys.exit()

    except OSError as e:  # if OS error occurs, exit pipeline
        q.put(logger.critical('OS related error occurred during reduction: ' + str(e)))
        logger.shutdown()
        sys.exit(-1)

    except SystemError as e:  # if system error occurs, exit pipeline
        q.put(logger.critical('Interpreter-related error occurred during reduction: ' + str(e)))
        logger.shutdown()
        sys.exit(-1)


def cli():
    params = argparse.ArgumentParser(description='User parameters.')
    params.add_argument('--telescope', default=None, help='Telescope of data.')  # telescope argument required
    params.add_argument('--date', default=None, help='Date of files to process.')  # optional date argument
    params.add_argument('--cpu', default=None, help='Number of nodes to run in parallel.')  # optional parallel argument
    args = params.parse_args()

    main(telescope=args.telescope, date=args.date, cpu=args.cpu)
