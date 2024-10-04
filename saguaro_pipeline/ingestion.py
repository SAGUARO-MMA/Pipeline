import os
from astropy.time import Time
from astropy.table import Table
import pickle
import time

import ephem  # PyEphem module
from PIL import Image
from astropy.coordinates import SkyCoord
import numpy as np

from . import newsql, saguaro_logging
from importlib_resources import files
from tensorflow.keras import models
import argparse
import importlib
import datetime
import glob


def convert_mpcorb_to_monthly_catalog(filename_in, filename_out):
    raw_catalog_list = []
    f = open(filename_in, "r")
    for line in f:
        if line == "\n":
            continue  # in case there's a blank
        else:  # line in the original data file
            raw_catalog_list.append(line.rstrip())
    f.close()

    for n in range(100):
        if raw_catalog_list[n] == '-' * 160:
            start_line = n + 1
            # to define the start of the actual data table,
            # which comes after ~30 lines of header text

    cropped_catalog_list = []
    # crop off the header
    for n in range(len(raw_catalog_list) - start_line):
        cropped_catalog_list.append(raw_catalog_list[n + start_line])

    full_catalog = []

    for obj_mpc in cropped_catalog_list:
        abs_m_H = obj_mpc[8:14].strip()
        slope_G = obj_mpc[14:20].strip()
        epoch = obj_mpc[20:26].strip()
        mean_anomaly_M = obj_mpc[26:36].strip()
        peri = obj_mpc[37:47].strip()
        node = obj_mpc[48:58].strip()
        inclin = obj_mpc[59:69].strip()
        eccen = obj_mpc[70:80].strip()
        motion_n = obj_mpc[80:92].strip()
        a = obj_mpc[92:104].strip()
        unc_U = obj_mpc[105:107].strip()
        readable_designation = obj_mpc[166:194].strip()

        # MPC format has a "packed" date, allowing the epoch to be stored in
        # fewer digits. However, this must be converted to mm/dd/yyyy format
        # for XEphem.
        epoch_x = f'{int(epoch[3], 36):02d}/{int(epoch[4], 36):02d}.0/{int(epoch[0], 36):02d}{epoch[1:3]}'

        if unc_U == "":
            unc_U = "?"
        expanded_designation = readable_designation + " " + unc_U

        # Write XEphem format orbit to the full_catalog list.
        full_catalog.append(expanded_designation + ",e," + inclin + ","
                            + node + "," + peri + "," + a + "," + motion_n + "," + eccen + "," +
                            mean_anomaly_M + "," + epoch_x + "," + "2000" + ",H " + abs_m_H +
                            "," + slope_G + "\n")

    f2 = open(filename_out, "w")
    for obj in full_catalog:
        f2.write(obj)
    f2.close()


def movingobjectcatalog(obsmjd):
    catalog_list = []
    tobs = Time(obsmjd, format='mjd').strftime('%Y_%m')
    fnam = f"{tobs}_ORB.DAT"
    if not os.path.exists(fnam):
        os.system('wget -nv -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        convert_mpcorb_to_monthly_catalog('MPCORB.DAT', fnam)
    elif time.time() - os.path.getmtime(fnam)>(24*60*60):
        os.system('wget -nv -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        convert_mpcorb_to_monthly_catalog('MPCORB.DAT', fnam)
    with open(fnam) as f_catalog:
        for line in f_catalog:
            catalog_list.append(ephem.readdb(line))
    return catalog_list


def movingobjectfilter(s_catalog, s_ra, s_dec, obsmjd, filter_radius):
    """Searches for matches between (ra, dec, time) and the Minor Planet Center catalog. Takes about a minute to run."""
    tobs = Time(obsmjd, format='mjd')
    s_date = ephem.date(tobs.datetime)
    ras, decs = [], []
    for body in s_catalog:
        body.compute(s_date)
        ras.append(body.a_ra)
        decs.append(body.a_dec)
    catalog_coords = SkyCoord(ras, decs, unit='deg')
    s_coords = SkyCoord(s_ra, s_dec)
    _, separation, _ = s_coords.match_to_catalog_sky(catalog_coords)
    return separation.arcsec < filter_radius


def getsky(data):
    """
    Determine the sky parameters for a FITS data extension.
    data -- array holding the image data
    """

    # maximum number of interations for mean,std loop
    maxiter = 30

    # maximum number of data points to sample
    maxsample = 10000

    # size of the array
    ny, nx = data.shape

    # how many sampels should we take?
    if data.size > maxsample:
        nsample = maxsample
    else:
        nsample = data.size

    # create sample indicies
    xs = np.random.uniform(low=0, high=nx, size=nsample).astype('L')
    ys = np.random.uniform(low=0, high=ny, size=nsample).astype('L')

    # sample the data
    sample = data[ys, xs].copy()
    sample = sample.reshape(nsample)

    # determine the clipped mean and standard deviation
    mean = sample.mean()
    std = sample.std()
    oldsize = 0
    niter = 0
    while oldsize != sample.size and niter < maxiter:
        niter += 1
        wok = (sample < mean + 5 * std)
        sample = sample[wok]
        wok = (sample > mean - 5 * std)
        sample = sample[wok]
        mean = sample.mean()
        std = sample.std()
        return mean, std


def imgscale(data):
    sky, sig = getsky(data)
    depth = 256
    # set the color range
    zero = sky + (-2) * sig
    span = 4 * sig
    # scale the data to the requested display values
    # greys
    new_data = data - zero
    new_data *= (depth - 1) / span
    # black
    w = new_data < 0
    new_data[w] = 0
    # white
    w = new_data > (depth - 1)
    new_data[w] = (depth - 1)
    new_data = new_data + (256 - depth)
    return new_data


print('Loading classifier...')
with open(os.environ['ML_MODEL_OLD'], 'rb') as f:
    classifier = pickle.load(f)
classifier.n_jobs = 1
print('Classifier loaded.')
print('Loading moving object catalog...')
catalog = movingobjectcatalog(Time.now().mjd)
print('Moving object catalog loaded.')


def ingestion(transCatalog, log=None):
    if log is not None:
        log.info('Ingesting catalog.')
        log.info('Loading NN classifier...')
    ml_model_new = os.getenv('ML_MODEL_NEW', files('saguaro_pipeline').joinpath('model_onlyscorr16_ml'))
    model = models.load_model(ml_model_new, compile=False)
    model.compile(optimizer='Adam', metrics=['accuracy'], loss='binary_crossentropy')
    if log is not None:
        log.info('NN classifer loaded.')

    imgt0 = time.time()
    image_data = Table.read(transCatalog, unit_parse_strict='silent')
    if log is not None:
        log.info('Ingestion: '+str(len(image_data)) + ' candidates found.')
    basefile = os.path.basename(transCatalog)
    observation_id, dateobs = newsql.add_observation_record(basefile, image_data.meta, log=log)

    ra = image_data['ALPHAWIN_J2000'].quantity
    dec = image_data['DELTAWIN_J2000'].quantity
    image_data['CX'] = np.cos(ra) * np.cos(dec)
    image_data['CY'] = np.sin(ra) * np.cos(dec)
    image_data['CZ'] = np.sin(dec)

    # Moving Object Classification
    tmobjmatch_start = time.time()
    image_data['CLASSIFICATION'] = movingobjectfilter(catalog, ra, dec, image_data.meta['MJD'], 25.0).astype(int)
    tmobjmatch = time.time() - tmobjmatch_start

    tml_start = time.time()
    mldata = image_data['THUMBNAIL_D'][:, 27:37, 27:37]
    mldata_mean = np.nanmean(mldata, axis=(1, 2))[:, np.newaxis, np.newaxis]
    mldata_std = np.nanmean(mldata, axis=(1, 2))[:, np.newaxis, np.newaxis]
    mldata = mldata / mldata_mean * np.log(1. + mldata_mean / mldata_std)
    image_data['MLSCORE'] = classifier.predict_proba(mldata.reshape(-1, 100))[:, 0]
    tml = time.time() - tml_start

    tml_nn_start = time.time()
    scorr_data = image_data['THUMBNAIL_SCORR'][:, 24:40, 24:40]
    image_data['MLSCORE_BOGUS'], image_data['MLSCORE_REAL'] = model.predict(scorr_data, verbose=2).T
    tml_nn = time.time() - tml_nn_start

    ttingest_start = time.time()
    image_data['TARGETID'] = newsql.get_or_create_targets(image_data['ALPHAWIN_J2000'], image_data['DELTAWIN_J2000'])
    ttingest = time.time() - ttingest_start

    tcingest_start = time.time()
    newsql.ingestcandidates(image_data, observation_id, dateobs)
    tcingest = time.time() - tcingest_start

    pngpath = os.path.join(os.environ['THUMB_PATH'], basefile[4:8], basefile[8:10], basefile[10:12],
                           image_data.meta['OBJECT'])
    os.makedirs(pngpath, exist_ok=True)

    tpng, tpngsave = [], []
    for row in image_data:
        tpng_start = time.time()

        if np.mean(row[15]) != 0:
            data = imgscale(row[15])
        else:
            data = row[15]
        img = Image.fromarray(data)
        img = img.convert('L')

        if np.mean(row[16]) != 0:
            data = imgscale(row[16])
        else:
            data = row[16]
        ref = Image.fromarray(data)
        ref = ref.convert('L')

        if np.mean(row[17]) != 0:
            data = imgscale(row[17])
        else:
            data = row[17]
        diff = Image.fromarray(data)
        diff = diff.convert('L')

        if np.mean(row[18]) != 0:
            data = imgscale(row[18])
        else:
            data = row[18]
        scorr = Image.fromarray(data)
        scorr = scorr.convert('L')
        tpng.append((time.time() - tpng_start))

        tpngsave_start = time.time()
        visit = basefile.split('_')[4]
        img.save(f"{pngpath}/{row['NUMBER']}_{visit}_img.png")
        ref.save(f"{pngpath}/{row['NUMBER']}_{visit}_ref.png")
        diff.save(f"{pngpath}/{row['NUMBER']}_{visit}_diff.png")
        scorr.save(f"{pngpath}/{row['NUMBER']}_{visit}_scorr.png")
        tpngsave.append(time.time() - tpngsave_start)

    tcomp = time.time() - imgt0
    if log is not None:
        log.info(f'''Ingestion: {basefile}. Total time to
        match moving objects = {tmobjmatch:.4f} s,
        run old ML = {tml:.4f} s,
        run new ML = {tml_nn:.4f} s,
        make png = {np.sum(tpng):.4f} s,
        save png = {np.sum(tpngsave):.4f} s,
        ingest targets = {ttingest:.4f} s,
        ingest candidates = {tcingest:.4f} s.''')
        log.info(f'Ingestion: Time to complete {basefile} = {tcomp:.1f} s, {len(image_data) / tcomp:.1f} cand/s')


def cli():
    params = argparse.ArgumentParser(description='User parameters.')
    params.add_argument('--telescope', default=None, help='Telescope of data.')  # telescope argument required
    params.add_argument('--date', default=None, help='Date of files to process.')  # date argument required
    args = params.parse_args()

    t0 = time.time()

    if args.telescope is None or args.date is None:  # if no telescope or date is given, exit with error
        raise ValueError('No telescope or date given, please give telescope and date and re-run.')

    try:
        tel = importlib.import_module(f'{__package__}.{args.telescope}')  # import telescope setting file
    except ImportError:
        raise ValueError('No such telescope file, please check that the file is in the same directory as the pipeline.')

    if '-' in args.date:
        date = args.date.replace('-', '/')
    elif '.' in args.date:
        date = args.date.replace('.', '/')
    elif '/' in args.date:
        date = args.date
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d').strftime('%Y/%m/%d')

    log_path = tel.log_path()  # set logpath: where log is written
    os.makedirs(log_path, exist_ok=True)  # if log path does not exist, make path

    red_path = tel.red_path(date)  # set path where reduced science images are written to
    os.makedirs(red_path, exist_ok=True)  # if path does not exist, make path

    log_file_name = f'{log_path}/reingest_{datetime.datetime.utcnow().strftime("%Y%m%d_T%H%M%S")}'
    logger = saguaro_logging.initialize_logger(log_file_name)

    files = sorted(glob.glob(red_path + '/*_trans.fits*'))
    for f in files:
        ingestion(f, logger)
    logger.info(f'Total wall-time spent: {time.time() - t0} s')
