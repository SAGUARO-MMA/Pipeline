import os
import datetime
import pickle
import time

import ephem  # PyEphem module
from PIL import Image
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.io import fits
import numpy as np

import newsql


def movingobjectcatalog(obsmjd):
    catalog_list = []
    t0 = datetime.datetime(1858, 11, 17)
    tobs = t0 + datetime.timedelta(obsmjd)
    fnam = f"{tobs.year:04d}_{tobs.month:02d}_ORB.DAT"
    if not os.path.exists(fnam):
        os.system('wget -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        os.system('python MPCORB2MonthlyCatalog.py')
    with open(fnam) as f_catalog:
        for line in f_catalog:
            catalog_list.append(line.rstrip())
    f_catalog.close()
    return catalog_list


def movingobjectfilter(s_catalog, s_ra, s_dec, obsmjd, filter_radius):
    DEG2RAD = 0.01745329252
    s_ra_radians, s_dec_radians = s_ra * DEG2RAD, s_dec * DEG2RAD
    t0 = datetime.datetime(1858, 11, 17)
    tobs = t0 + datetime.timedelta(obsmjd)

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
    data -= zero
    data *= (depth - 1) / span
    # black
    w = data < 0
    data[w] = 0
    # white
    w = data > (depth - 1)
    data[w] = (depth - 1)
    data = data + (256 - depth)
    return data


def ingestion(transCatalog, log):
    log = None
    if log is not None:
        log.info('Ingesting catalog.')
    ml_model = '/home/saguaro/software/lundquist/rf_model.ml'
    print('Loading classifier\n')
    classifier = pickle.load(open(ml_model, 'rb'))
    print('Classifier loaded\n')
    imgt0 = time.time()
    hdul = fits.open(transCatalog)
    hdul.info()
    hdr = hdul[1].header
    image_data = hdul[1].data
    if log is not None:
        log.info(str(len(image_data)) + ' candidates found.')
    print(str(len(image_data)) + ' candidates found.')
    rawfile = transCatalog.replace('_red_trans.fits', '.arch')
    basefile = os.path.basename(transCatalog)
    pngpath_main = f'/home/saguaro/data/png/{basefile[4:8]}/{basefile[8:10]}/{basefile[10:12]}'
    resfile, resnumber = newsql.pipecandmatch(basefile)
    tpng, tpng2, tml, ttingest, tcingest, tglade, tmobjmatch, tgmatch, tpngsave = [], [], [], [], [], [], [], [], []
    print(resfile, len(resfile), len(image_data))
    if len(resfile) == 0 or len(resfile) < len(image_data):

        # Moving Object Classification
        #    catalog=movingobjectcatalog(float(hdr['MJD']))
        ra, dec = radectodecimal(hdr['RA'], hdr['DEC'])
        #    filtered_catalog=movingobjectfilter(catalog,ra,dec, float(hdr['MJD']), 2.5*3600.)

        tpng, tml, ttingest, tcingest, tglade, tmobjmatch, tgmatch = [], [], [], [], [], [], []
        for row in image_data:
            rowt0 = time.time()
            pngpath = pngpath_main + '/' + str(hdr['OBJECT'])

            print(row[0], resnumber)
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

                if np.mean(row[17]) != 0:
                    data = imgscale(row[17])
                else:
                    data = row[17]
                diff = Image.fromarray(data)
                diff = diff.convert('L')

                tpng.append((time.time() - rowt0))
                asize = 64
                msize = 10
                mldata = data[int(asize / 2 - msize / 2):int(asize / 2 + msize / 2),
                              int(asize / 2 - msize / 2):int(asize / 2 + msize / 2)]
                mldata = ((mldata / np.nanmean(mldata)) * np.log(1 + (np.nanmean(mldata) / np.nanstd(mldata))))
                try:
                    score = (classifier.predict_proba(mldata.reshape((1, -1))))[0][1]
                except:
                    score = 0

                tml.append(time.time() - rowt0)
                if np.mean(row[18]) != 0:
                    data = imgscale(row[18])
                else:
                    data = row[18]
                scorr = Image.fromarray(data)
                scorr = scorr.convert('L')

                tpng2.append((time.time() - rowt0))

                # Moving Object Classification
                #   mvobj=movingobjectfilter(filtered_catalog,float(row[7]),float(row[8]), float(hdr['MJD']), 25.0)
                #  if len(mvobj)==0:
                #      classification='0'
                #  else:
                #      classification='1'
                tmobjmatch.append(time.time() - rowt0)

                # Previously detected object search
                tgmatch.append(time.time() - rowt0)

                number = str(row[0])
                filename = str(basefile)
                elongation = str(row[6])
                ra = str(row[7])
                dec = str(row[8])
                fwhm = str(row[9])
                snr = str(row[10])
                mag = str(row[13])
                magerr = str(row[14])
                rawfilename = str(rawfile)
                obsdate = str(hdr['DATE-OBS'])
                field = str(hdr['OBJECT'])
                ncomb = str(hdr['NCOMBINE'])

                if 'MJDMID' not in hdr:
                    mmjd = str(float(hdr['MJD']) + float(hdr['EXPTIME']) / 2. / 86400.)
                else:
                    mmjd = str(hdr['MJDMID'])
                res = newsql.get_or_create_target(float(ra), float(dec))
                ttingest.append(time.time() - rowt0)

                cx = np.cos(np.radians(float(ra))) * np.cos(np.radians(float(dec)))
                cy = np.sin(np.radians(float(ra))) * np.cos(np.radians(float(dec)))
                cz = np.sin(np.radians(float(dec)))
                htm16ident = -1

                tglade.append(time.time() - rowt0)
                newsql.ingestcandidates(number, filename, elongation, ra, dec, fwhm, snr, mag, magerr, rawfilename,
                                        obsdate, field, 0, cx, cy, cz, htm16ident, res['id'][0], mmjd, score, ncomb)
                tcingest.append(time.time() - rowt0)
                os.makedirs(pngpath, exist_ok=True)
                visit = filename.split('_')[4]
                img.save(pngpath + '/' + str(row[0]) + '_' + visit + '_img.png', "PNG")
                ref.save(pngpath + '/' + str(row[0]) + '_' + visit + '_ref.png', "PNG")
                diff.save(pngpath + '/' + str(row[0]) + '_' + visit + '_diff.png', "PNG")
                scorr.save(pngpath + '/' + str(row[0]) + '_' + visit + '_scorr.png', "PNG")
                tpngsave.append(time.time() - rowt0)

    tcomp = time.time() - imgt0
    if log is not None:
        log.info('Time to complete ' + rawfile + ': ', tcomp, len(image_data) / tcomp, 'cand/sec')
