import os
from astropy.time import Time
import pickle
import time

import ephem  # PyEphem module
from PIL import Image
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.io import fits
import numpy as np

from . import newsql
from importlib_resources import files
from tensorflow.keras import models


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
        os.remove(fnam)
        os.system('wget -nv -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        convert_mpcorb_to_monthly_catalog('MPCORB.DAT', fnam)
    with open(fnam) as f_catalog:
        for line in f_catalog:
            catalog_list.append(line.rstrip())
    return catalog_list


def movingobjectfilter(s_catalog, s_ra, s_dec, obsmjd, filter_radius):
    DEG2RAD = 0.01745329252
    s_ra_radians, s_dec_radians = s_ra * DEG2RAD, s_dec * DEG2RAD
    tobs = Time(obsmjd, format='mjd')

    s_full_date = tobs.strftime("%Y/%m/%d %H:%M:%S")
    s_date = ephem.date(s_full_date)
    s_filtered_bodies = []

    for body in s_catalog:
        test_obj = ephem.readdb(body)
        test_obj.compute(s_date)
        test_obj_ra = test_obj.a_ra
        test_obj_dec = test_obj.a_dec
        test_obj_separation = 206264.806 * float(ephem.separation((test_obj_ra, test_obj_dec),
                                                                  (s_ra_radians, s_dec_radians)))
        if test_obj_separation < filter_radius:
            s_filtered_bodies.append(body)
    return s_filtered_bodies


def radectodecimal(ra, dec):
    c = SkyCoord(ra, dec, unit='deg')
    c = c.to_string('decimal')
    both = str.split(str(c))
    rad = float(both[0]) * 15.
    decd = float(both[1])
    return rad, decd


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


print('Loading classifier\n')
with open(os.environ['ML_MODEL_OLD'], 'rb') as f:
    classifier = pickle.load(f)
print('Classifier loaded\n')
print('Loading NN classifier\n')
ml_model_new = os.getenv('ML_MODEL_NEW', files('saguaro_pipeline').joinpath('model_onlyscorr16_ml'))
model = models.load_model(ml_model_new, compile=False)
model.compile(optimizer='Adam', metrics=['accuracy'], loss='binary_crossentropy')
print('NN classifer loaded\n')
print('Loading moving object catalog\n')
catalog = movingobjectcatalog(Time.now().mjd)
print('Moving object catalog loaded\n')


def ingestion(transCatalog, log=None):
    if log is not None:
        log.info('Ingesting catalog.')
    imgt0 = time.time()
    with fits.open(transCatalog) as hdul:
        hdul.info()
        hdr = hdul[1].header
        image_data = hdul[1].data
    if log is not None:
        log.info('Ingestion: '+str(len(image_data)) + ' candidates found.')
    print(str(len(image_data)) + ' candidates found.')
    basefile = os.path.basename(transCatalog)
    pngpath_main = f'{os.environ["THUMB_PATH"]}/{basefile[4:8]}/{basefile[8:10]}/{basefile[10:12]}'
    observation_id, dateobs = newsql.add_observation_record(basefile, hdr)
    resnumber = newsql.pipecandmatch(observation_id)
    tpng, tml, tml_nn, ttingest, tcingest, tmobjmatch, tpngsave = [], [], [], [], [], [], []
    print(observation_id, len(resnumber), len(image_data))
    if len(resnumber) < len(image_data):  # not fully ingested before

        # Moving Object Classification
        ra, dec = radectodecimal(hdr['RA'], hdr['DEC'])
        filtered_catalog=movingobjectfilter(catalog,ra,dec, float(hdr['MJD']), 2.5*3600.)

        pngpath = f"{pngpath_main}/{hdr['OBJECT']}"
        os.makedirs(pngpath, exist_ok=True)

        tpng, tml, ttingest, tcingest, tmobjmatch = [], [], [], [], []
        for row in image_data:
            rowt0 = time.time()

        #    print(row[0], resnumber)
            if str(row[0]) not in resnumber:
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

                diff_data = row[17]
                if np.mean(row[17]) != 0:
                    data = imgscale(row[17])
                else:
                    data = row[17]
                diff = Image.fromarray(data)
                diff = diff.convert('L')

                scorr_data = row[18][24:40,24:40]
                if np.mean(row[18]) != 0:
                    data = imgscale(row[18])
                else:
                    data = row[18]
                scorr = Image.fromarray(data)
                scorr = scorr.convert('L')
                tpng.append((time.time() - rowt0))

                visit = basefile.split('_')[4]
                img.save(f"{pngpath}/{row['NUMBER']}_{visit}_img.png")
                ref.save(f"{pngpath}/{row['NUMBER']}_{visit}_ref.png")
                diff.save(f"{pngpath}/{row['NUMBER']}_{visit}_diff.png")
                scorr.save(f"{pngpath}/{row['NUMBER']}_{visit}_scorr.png")
                tpngsave.append(time.time() - rowt0)

                asize = 64
                msize = 10
                mldata = diff_data[int(asize / 2 - msize / 2):int(asize / 2 + msize / 2),
                              int(asize / 2 - msize / 2):int(asize / 2 + msize / 2)]
                mldata = ((mldata / np.nanmean(mldata)) * np.log(1 + (np.nanmean(mldata) / np.nanstd(mldata))))
                try:
                    score = (classifier.predict_proba(mldata.reshape((1, -1))))[0][1]
                except:
                    score = 0

                tml.append(time.time() - rowt0)

                # Moving Object Classification
                mvobj = movingobjectfilter(filtered_catalog, float(row[7]), float(row[8]), float(hdr['MJD']), 25.0)
                if mvobj:
                    classification = 1
                else:
                    classification = 0
#                tmobjmatch.append(time.time() - rowt0)

                tml_nn_start = time.time()
                
                score_bogus, score_real = model.predict(scorr_data[None])[0]
                tml_nn.append(time.time() - tml_nn_start)

                ra = row['ALPHAWIN_J2000']
                dec = row['DELTAWIN_J2000']
                res = newsql.get_or_create_target(ra, dec)
                ttingest.append(time.time() - rowt0)

                cx = np.cos(np.radians(ra)) * np.cos(np.radians(dec))
                cy = np.sin(np.radians(ra)) * np.cos(np.radians(dec))
                cz = np.sin(np.radians(dec))

                newsql.ingestcandidates(row['NUMBER'], row['ELONGATION'], ra, dec, row['FWHM_TRANS'], row['S2N'],
                                        row['MAG_PSF'], row['MAGERR_PSF'], classification, cx, cy, cz, res['id'][0],
                                        score, score_bogus, score_real, observation_id, dateobs)
                tcingest.append(time.time() - rowt0)

    tcomp = time.time() - imgt0
    if log is not None:
        log.info('Ingestion: '+basefile+'  Average time to make png, save png, run ml, target ingest,candidateingest,total candidates,'+ str(np.mean(tpng))+','+str(np.mean(tpngsave))+','+str(np.mean(tml))+','+str(np.mean(tml_nn))+','+str(np.mean(ttingest))+','+str(np.mean(tcingest))+','+str(len(tpng)))
        log.info('Ingestion: Time to complete ' + basefile + ': '+str(tcomp)+' '+str(len(image_data) / tcomp)+' cand/sec')
