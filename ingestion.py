import multiprocessing
from multiprocessing import Pool, Manager
from multiprocessing import Queue, Process, cpu_count
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from astropy.io import fits
import io
import base64
import matplotlib.pyplot as plt
import sys
import newsql
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import argparse
from glob import glob
import datetime
import numpy as np
from numpy import *
from PIL import Image
import time
import sys
import operator
import ephem # PyEphem module
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets,svm,metrics
from sklearn.utils import shuffle
import re,ast
import pickle
from astropy.utils.exceptions import AstropyWarning
# from email import email_field




def gaialist(field):
    db = newsql.Dictdb()
    res=db.queryfetchall("select * from gaia where cssfield = '%s';" % (field))
    db.close()
    return res

def gaiamatch(gaia,arcsec,ra,dec):
    count=0
    if gaia:
        new_dict={}
        coords_cand = SkyCoord(ra*u.deg,dec*u.deg,frame='fk5')
        coords_gaia = SkyCoord(gaia['ra'], gaia['dec'],frame='fk5')
        idx, d2, d3 = coords_cand.match_to_catalog_sky(coords_gaia)
        for jj,mm in enumerate(idx):
            pm_tot = ((gaia['pmra'][mm]* np.cos(gaia['dec'][mm]))**2 + gaia['pmdec'][mm]**2) ** 0.5
            if d2[jj].deg*3600 < arcsec and pm_tot > 3 * (pm_tot) * (((2 * gaia['pmra_error'][mm]/gaia['pmra'][mm])**2 + (2 * gaia['pmdec_error'][mm]/gaia['pmdec'][mm])**2)**0.5) /2:
                return True
            else:
                res=False
    else:
        res=False
    return res

def movingobjectcatalog(obsmjd):
    catalog_list = []
    t0 = datetime.datetime(1858, 11, 17)
    tobs = t0 + datetime.timedelta(obsmjd)
    s_full_date=tobs.strftime("%Y/%m/%d %H:%M:%S")
    fnam=("%04i_%02i_ORB.DAT" %
                         (tobs.year, tobs.month))
    if not os.path.exists(fnam):
        os.system('wget -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        os.system('python MPCORB2MonthlyCatalog.py')
    with open("%04i_%02i_ORB.DAT" % (tobs.year, tobs.month)) as f_catalog:
        for line in f_catalog:
            catalog_list.append(line.rstrip())
    f_catalog.close()
    s_catalog=catalog_list
    return s_catalog

def movingobjectfilter(s_catalog,s_ra, s_dec, obsmjd, filter_radius):
    DEG2RAD = 0.01745329252
    s_ra_radians,s_dec_radians=s_ra*DEG2RAD,s_dec*DEG2RAD
    t0 = datetime.datetime(1858, 11, 17)
    tobs = t0 + datetime.timedelta(obsmjd)

    s_full_date=tobs.strftime("%Y/%m/%d %H:%M:%S")
    s_date = ephem.date(s_full_date)
    s_filtered_bodies = []

    t0=time.time()
    for body in s_catalog:
        test_obj = ephem.readdb(body)
        test_obj.compute(s_date)
        test_obj_ra = test_obj.a_ra
        test_obj_dec = test_obj.a_dec
        test_obj_separation = 206264.806*(float(ephem.separation((test_obj_ra,
            test_obj_dec), (s_ra_radians, s_dec_radians))))
        if (test_obj_separation < filter_radius):
            s_filtered_bodies.append(body)
    return s_filtered_bodies

def movingobjectmatch(s_catalog,s_ra, s_dec, obsmjd, list_radius):
    DEG2RAD = 0.01745329252
    s_ra_radians,s_dec_radians=s_ra*DEG2RAD,s_dec*DEG2RAD
    t0 = datetime.datetime(1858, 11, 17)
    tobs = t0 + datetime.timedelta(obsmjd)

    s_full_date=tobs.strftime("%Y/%m/%d %H:%M:%S")
    s_date = ephem.date(s_full_date)
    s_unsorted_bodies = []

    t0=time.time()
    for body in s_catalog:
        test_obj = ephem.readdb(body)
        test_obj.compute(s_date)
        test_obj_ra = test_obj.a_ra
        test_obj_dec = test_obj.a_dec
        test_obj_separation = 206264.806*(float(ephem.separation((test_obj_ra,
            test_obj_dec), (s_ra_radians, s_dec_radians))))
        if (test_obj_separation < list_radius):
            s_unsorted_bodies.append((test_obj.name,
                test_obj_separation, str(test_obj_ra),
                str(test_obj_dec), str(test_obj.mag)))
    matches=s_unsorted_bodies
    return s_unsorted_bodies

def radectodecimal(ra,dec):
    c=SkyCoord(ra,dec,unit='deg')
    c=c.to_string('decimal')
    both=str.split(str(c))
    rad=float(both[0])*15.
    decd=float(both[1])
    return rad,decd


def gladelist(field):
    db = newsql.Dictdb()
    res=db.queryfetchall("select * from glade where cssfield = '%s' AND dist > 5;" % (field))
    db.close()
    return res

def gladematch(gladelist,kpclimit,cx,cy,cz):
    if gladelist !={}:
        new_dict={}
        cx=np.float128(cx)
        cy=np.float128(cy)
        cz=np.float128(cz)
        idx=range(len(gladelist['ra']))
        gid,dist=[],[]
        for xx in idx:
            if cx*np.float128(gladelist['cx'][xx]) + cy*np.float128(gladelist['cy'][xx]) + cz*np.float128(gladelist['cz'][xx]) >= np.cos(kpclimit/(1000.)*(1.0/gladelist['dist'][xx])):
                gid.append(gladelist['id'][xx])
                dist.append(np.arccos(cx*np.float128(gladelist['cx'][xx]) + cy*np.float128(gladelist['cy'][xx]) + cz*np.float128(gladelist['cz'][xx]))*(gladelist['dist'][xx]*1000))
        if gid !=[]:
            val,pid=min((val,pid) for (pid,val) in enumerate(dist))
            res=gid[pid]
        else:
            res=-1
    else:
        res=-1
    return res

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
  ny,nx = data.shape

  # how many sampels should we take?
  if data.size > maxsample:
    nsample = maxsample
  else:
    nsample = data.size

  # create sample indicies
  xs = random.uniform(low=0, high=nx, size=nsample).astype('L')
  ys = random.uniform(low=0, high=ny, size=nsample).astype('L')

  # sample the data
  sample = data[ys,xs].copy()
  sample = sample.reshape(nsample)

  # determine the clipped mean and standard deviation
  mean = sample.mean()
  std = sample.std()
  fullsample=sample
  oldsize = 0
  niter = 0
  while oldsize != sample.size and niter < maxiter:
    niter += 1
    oldsize = sample.size
    wok = (sample < mean + 5*std)
    sample = sample[wok]
    wok = (sample > mean - 5*std)
    sample = sample[wok]
    mean = sample.mean()
    std = sample.std()
    return mean,std
def imgscale(data):
    sky, sig = getsky(data)
    depth=256
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




def ingestion(transCatalog,log):
    log=None
    if log !=None:log.info('Ingesting catalog.')
    ml_model = '/home/saguaro/software/lundquist/rf_model.ml'
    print('Loading classifier\n')
    classifier = pickle.load(open(ml_model, 'rb'))
    print('Classifier loaded\n')
    if True:
        imgt0=time.time()
        hdul=fits.open(transCatalog)
        hdul.info()
        hdr=hdul[1].header
        image_data=hdul[1].data
        if log !=None:log.info(str(len(image_data))+' candidates found.')
        print(str(len(image_data))+' candidates found.')
        rawfile=transCatalog.replace('_red_trans.fits','.arch')
        pngpath_main='/home/saguaro/data/png/'+transCatalog[4:8]+'/'+transCatalog[8:10]+'/'+transCatalog[10:12]
        tfile=time.time()-imgt0
        resfile,resnumber=newsql.pipecandmatch(os.path.basename(transCatalog))
        tpipecand=time.time()-imgt0
        tpng,tpng2,tml,ttingest,tcingest,tglade,tmobjmatch,tgmatch,tpngsave=[],[],[],[],[],[],[],[],[]
        field=str(hdr['OBJECT'])
        print(resfile,len(resfile),len(image_data))
        if len(resfile) == 0 or len(resfile) < len(image_data):
            
            ####  Moving Object Classification  ####'
        #    catalog=movingobjectcatalog(float(hdr['MJD']))
            ra,dec=radectodecimal(hdr['RA'],hdr['DEC'])
        #    filtered_catalog=movingobjectfilter(catalog,ra,dec, float(hdr['MJD']), 2.5*3600.)
            tmobj=time.time()-imgt0
            #gl=gladelist(str(hdr['OBJECT']))

            ###gaia stars in field###
            #gaiadict=gaialist(field)

            tggrab=time.time()-imgt0
            tpng,tml,ttingest,tcingest,tglade,tmobjmatch,tgmatch=[],[],[],[],[],[],[]
            for i in range(len(image_data)):
                row=image_data[i]
                rowt0=time.time()
                pngpath=pngpath_main+'/'+str(hdr['OBJECT'])

                print(row[0],resnumber)
                if str(row[0]) not in resnumber:
                    if np.mean(row[15])!=0: data=imgscale(row[15])
                    if np.mean(row[15])==0: data=row[15]
                    img = Image.fromarray(data)
                    img=img.convert('L')


                    if np.mean(row[16])!=0: data=imgscale(row[16])
                    if np.mean(row[16])==0: data=row[16]
                    ref = Image.fromarray(data)
                    ref=ref.convert('L')

                    if np.mean(row[17])!=0: data=imgscale(row[17])
                    if np.mean(row[17])==0: data=row[17]
                    diff = Image.fromarray(data)
                    diff=diff.convert('L')

                    tpng.append((time.time()-rowt0))
                    asize=64
                    msize=10
                    mldata=data[int(asize/2-msize/2):int(asize/2+msize/2),int(asize/2-msize/2):int(asize/2+msize/2)]
                    mldata=( (mldata/np.nanmean(mldata))*np.log(1+(np.nanmean(mldata)/np.nanstd(mldata))) )
                    try:
                        score=(classifier.predict_proba(mldata.reshape((1, -1))))[0][1]
                    except:
                        score=0

                    tml.append(time.time()-rowt0)
                    if np.mean(row[18])!=0: data=imgscale(row[18])
                    if np.mean(row[18])==0: data=row[18]
                    scorr = Image.fromarray(data)
                    scorr=scorr.convert('L')

                    tpng2.append((time.time()-rowt0))

                ####  Moving Object Classification  ####
                 #   mvobj=movingobjectmatch(filtered_catalog,float(row[7]),float(row[8]), float(hdr['MJD']), 25.0)
                  #  if len(mvobj)==0:
                  #      classification='0'
                  #  else:
                  #      classification='1'
                    classification='0'
                    tmobjmatch.append(time.time()-rowt0)

                ####  Previously detected object search  ####

                    #gmatch=gaiamatch(gaiadict,0.5,[float(row[7])],[float(row[8])])
                    gmatch=False
                    if gmatch == True:
                        classification='7'
                    else:
                        classification='0'

                    tgmatch.append(time.time()-rowt0)

                    basefile=os.path.basename(transCatalog)
                    if len(row)==19:number,filename,xwin,ywin,errx2win,erry2win,errxywin,elongation,ra,dec,fwhm,snr,fluxpsf,fluxpsferr,mag,magerr,rawfilename,obsdate,field,seqnum,ncomb=str(row[0]),str(basefile),str(row[1]),str(row[2]),str(row[3]),str(row[4]),str(row[5]),str(row[6]),str(row[7]),str(row[8]),str(row[9]),str(row[10]),str(row[11]),str(row[12]),str(row[13]),str(row[14]),str(rawfile),str(hdr['DATE-OBS']),str(hdr['OBJECT']),str(hdr['SEQNUM']),str(hdr['NCOMBINE'])


                    if not 'DATE-MID' in hdr:
                        datemid=str(datetime.datetime.strptime(hdr['DATE-OBS']+' '+hdr['TIME-OBS'],'%Y-%m-%d %H:%M:%S.%f')+datetime.timedelta(seconds=float(hdr['EXPTIME'])/2.))
                        dmid=datemid.replace(" ","T",1)
                    else:
                        dmid=str(hdr['DATE-MID'])
                    if not 'MJDMID' in hdr:
                        mmjd=str(float(hdr['MJD'])+float(hdr['EXPTIME'])/2./86400.)
                    else:
                        mmjd=str(hdr['MJDMID'])
                    # res=newsql.ingesttargets(float(ra),float(dec),field,classification)
                    res = {'classification': [0], 'targetid': ['NULL']}
                    ttingest.append(time.time()-rowt0)

                    cx = np.cos( np.radians(float(ra)) )*np.cos( np.radians(float(dec)))
                    cy = np.sin( np.radians(float(ra)) )*np.cos( np.radians(float(dec)))
                    cz = np.sin( np.radians(float(dec)) )

                    #match=gladematch(gl,50,cx,cy,cz)
                    match=-1
                    tglade.append(time.time()-rowt0)
                    ret=newsql.ingestcandidateswithidreturn(number,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,res['classification'][0],cx,cy,cz,res['targetid'][0],mmjd,score,ncomb,match)
                    tcingest.append(time.time()-rowt0)
                    if not os.path.exists(pngpath):os.makedirs(pngpath)
                    visit=filename.split('_')[4]
                    img.save(pngpath+'/'+str(row[0])+'_'+visit+'_img.png',"PNG")
                    ref.save(pngpath+'/'+str(row[0])+'_'+visit+'_ref.png',"PNG")
                    diff.save(pngpath+'/'+str(row[0])+'_'+visit+'_diff.png',"PNG")
                    scorr.save(pngpath+'/'+str(row[0])+'_'+visit+'_scorr.png',"PNG")
                    tpngsave.append(time.time()-rowt0)

        tcomp=time.time()-imgt0
        if log!=None:log.info('Time to complete '+rawfile+': ',tcomp,float(len(image_data))/tcomp,'cand/sec')
#        newsql.setingestedfiles(rawfile)
        return 
