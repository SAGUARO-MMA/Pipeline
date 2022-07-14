import multiprocessing
from multiprocessing import Pool, Manager
from multiprocessing import Queue, Process, cpu_count
import time
import os
from glob import glob
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
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets,svm,metrics
from sklearn.utils import shuffle
import re,ast
import pickle
from astropy.utils.exceptions import AstropyWarning
import smtplib

def email_field(gwname,field,trigger):
    from email.mime.text import MIMEText
    gmail_user=''
    gmail_password=''
    sent_from=gmail_user

    to=['']

    if trigger==True:
        subject = 'Field '+field+' has been ingested for '+gwname
        body = 'The field '+field+' that was triggered for '+gwname+' has been processed and ingested into the database. \n \n '

    else:
        subject = 'Field '+field+' has been ingested for '+gwname
        body = 'The field '+field+' was not triggered, but is within the 90% region for '+gwname+' and has been processed and ingested into the database.  \n \n '

    msg=MIMEText(body)
    msg['Subject']=subject
    msg['From']=gmail_user
    msg['To']=", ".join(to)
    email_text=msg.as_string()

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
        print ('Email sent!')
    except:
        print ('Something went wrong sending the email.')




def fieldcheck(field):
    print('started field check')
    res=gwfields()
#    print(res)
    gweventslist=[]

    gweventslist=set(res['gwname'])
    gwnames=res['gwname']
    fieldstriggered1=res['fieldstriggered']
    fieldstriggered2=res['fieldstriggered2']
    fieldstriggered3=res['fieldstriggered3']
    fields90=res['fields90']
    obsdate=res['ingestdate']
    zipped=zip(gwnames,fieldstriggered1,fieldstriggered2,fieldstriggered3,fields90,obsdate)

#Loop through all gwevents
#Sort list to find the most recent alert for each gwevent
    print('forloop')
    for i in gweventslist:
        tgw,tft1,tft2,tft3,tf90,tobs=[],[],[],[],[],[]
        print('second forloop')
        for a,b,c,d,e,f in zip(gwnames,fieldstriggered1,fieldstriggered2,fieldstriggered3,fields90,obsdate):
            if a==i:
                tgw.append(a)
                if b!=None:
                    tft1.append(b)
                else:
                    tft1.append('')
                if c!=None:
                    tft2.append(c)
                else:
                    tft2.append('')
                if d!=None:
                    tft3.append(d)
                else:
                    tft3.append('')
                if e!=None:
                    tf90.append(e)
                else:
                    tf90.append('')
                tobs.append(f)

        gwzip=zip(tgw,tft1,tft2,tft3,tf90,tobs)
        out=sorted(gwzip,key=lambda x:[-1])[0]
        gw,ft1,ft2,ft3,ft90,obs=out[0],out[1].split(','),out[2].split(','),out[3].split(','),out[4].split(','),out[5]
   
        ft_test=[x for x in [ft1,ft2,ft3] if field in x]
        ft90_test=[x for x in [ft90] if field in x]

        if ft_test !=[]:
            email_field(gw,field,trigger=True)
        elif ft90_test !=[]:
            email_field(gw,field,trigger=False)

def gwfields():
    db = newsql.Dictdb()
    res=db.queryfetchall("select gwname,fieldstriggered, fieldstriggered2, fieldstriggered3,fields90,ingestdate from gwevents where gwname in (select distinct gwname from gwevents where obstype='observation' and obsdate> current_timestamp- interval '5 days' and active =1) order by ingestdate desc ")
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
    try:
        f_catalog = file("%04i_%02i_ORB.DAT" %
                         (tobs.year, tobs.month), "r")
    except:
        os.system('wget -O MPCORB.DAT http://www.minorplanetcenter.org/iau/MPCORB/MPCORB.DAT')
        os.system('python MPCORB2MonthlyCatalog.py')
        f_catalog = file("%04i_%02i_ORB.DAT" %
                         (tobs.year, tobs.month), "r")

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
#    print(time.time()-t0)
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
#    print(time.time()-t0)
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




def proc(event):
    try:
        try:
            file = str(event.dest_path)
        except AttributeError:
            file = str(event.src_path) #get name of new file
    except AttributeError: #if event is a file
            file = event
    classifier = pickle.load(open('rf_model.ml', 'rb'))
    filelist=[file]
#    print(filelist,file.split('_')[-1],range(len(filelist)))
    if file.split('_')[-1] == 'trans.fits.fz':
        for ii in range(len(filelist)):
#            print('made it')
            imgt0=time.time()
            infile=filelist[ii]
            hdul=fits.open(infile)
            hdul.info()
            hdr=hdul[1].header
            image_data=hdul[1].data
            basefile=os.path.basename(infile)
 #           print('basefile:',basefile)
            rawfile=basefile.replace('_red_trans.fits.fz','.arch.fz')
            pngpath='/home/saguaro/data/png/'+basefile[4:8]+'/'+basefile[8:10]+'/'+basefile[10:12]
            tfile=time.time()-imgt0
            resfile,resnumber=newsql.pipecandmatch(basefile)
            tpipecand=time.time()-imgt0
            tpng,tpng2,tml,ttingest,tcingest,tglade,tmobjmatch,tgmatch,tpngsave=[],[],[],[],[],[],[],[],[]
#            print(len(resfile),len(image_data))
            field=str(hdr['OBJECT'])
            if len(resfile) == 0 or len(resfile) < len(image_data):
                print('####  Moving Object Classification  ####')
                catalog=movingobjectcatalog(float(hdr['MJD']))
                ra,dec=radectodecimal(hdr['RA'],hdr['DEC'])
                filtered_catalog=movingobjectfilter(catalog,ra,dec, float(hdr['MJD']), 2.5*3600.)
                tmobj=time.time()-imgt0
                gl=gladelist(str(hdr['OBJECT']))
                j=None
                try:
                    job2 = Gaia.launch_job_async("SELECT gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.astrometric_excess_noise,gaia_source.astrometric_excess_noise_sig,gaia_source.phot_g_mean_mag,gaia_source.bp_rp FROM gaiadr2.gaia_source WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',"+str(ra)+","+str(dec)+",1.7))=1;", dump_to_file=False)
                    j = job2.get_results()
                except:
                    print('Gaia search failed')
                tggrab=time.time()-imgt0
                tpng,tml,ttingest,tcingest,tglade,tmobjmatch,tgmatch=[],[],[],[],[],[],[]
                for i in range(len(image_data)):
                    row=image_data[i]
                    rowt0=time.time()
                    pngpath='/home/saguaro/data/png/'+basefile[4:8]+'/'+basefile[8:10]+'/'+basefile[10:12]+'/'+str(hdr['OBJECT'])

                    if str(row[0]) not in resnumber and float(row[9]) > 0.0:
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
                        score=(classifier.predict_proba(mldata.reshape((1, -1))))[0][1]

                        tml.append(time.time()-rowt0)
                        if np.mean(row[18])!=0: data=imgscale(row[18])
                        if np.mean(row[18])==0: data=row[18]
                        scorr = Image.fromarray(data)
                        scorr=scorr.convert('L')

                        tpng2.append((time.time()-rowt0))

                    ####  Moving Object Classification  ####
                        mvobj=movingobjectmatch(filtered_catalog,float(row[7]),float(row[8]), float(hdr['MJD']), 25.0)
                        if len(mvobj)==0:
                            classification='0'
                        else:
                            classification='1'

                        tmobjmatch.append(time.time()-rowt0)
                    ####  Previously detected object search  ####
                        gmatch=gaiamatch(j,0.5,[float(row[7])],[float(row[8])])
#                        gmatch = False
                        if gmatch == True:
                            classification='7'
                        else:
                            classification='0'

                        tgmatch.append(time.time()-rowt0)


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
  #                      print('ingesting.......')
                        res = newsql.get_or_create_target(float(ra), float(dec))
                        ttingest.append(time.time()-rowt0)
                        cx,cy,cz,htm16ident=newsql.coordinateident(float(ra),float(dec),lev=16)
                        cx = np.cos( np.radians(float(ra)) )*np.cos( np.radians(float(dec)))
                        cy = np.sin( np.radians(float(ra)) )*np.cos( np.radians(float(dec)))
                        cz = np.sin( np.radians(float(dec)) )
                        htm16ident=-1

                        match=gladematch(gl,50,cx,cy,cz)
                        tglade.append(time.time()-rowt0)
                        ret=newsql.ingestcandidateswithidreturn(number,filename,elongation,ra,dec,fwhm,snr,mag,magerr,rawfilename,obsdate,field,0,cx,cy,cz,htm16ident,res['id'][0],mmjd,score,ncomb,match)
                        tcingest.append(time.time()-rowt0)
                        if not os.path.exists(pngpath):os.makedirs(pngpath)
                        visit=filename.split('_')[4]
                        img.save(pngpath+'/'+str(row[0])+'_'+visit+'_img.png',"PNG")
                        ref.save(pngpath+'/'+str(row[0])+'_'+visit+'_ref.png',"PNG")
                        diff.save(pngpath+'/'+str(row[0])+'_'+visit+'_diff.png',"PNG")
                        scorr.save(pngpath+'/'+str(row[0])+'_'+visit+'_scorr.png',"PNG")
                        tpngsave.append(time.time()-rowt0)

#                        print('        Time to complete row: ',time.time()-rowt0)
#            print('tfile,tpipecand,tmobj,tggrab,pngavg,mlavg,pngavg2,mobjmatchavg,tgmatchavg,tingestavg,gladeavg,cingestavg,pngsave')
#            pngavg,pngavg2,mlavg,tingestavg,cingestavg,gladeavg,mobjmatchavg,tgmatchavg,pngsave=sum(tpng)/len(tpng),sum(tpng2)/len(tpng2),sum(tml)/len(tml),sum(ttingest)/len(ttingest),sum(tcingest)/len(tcingest),sum(tglade)/len(tglade),sum(tmobjmatch)/len(tmobjmatch),sum(tgmatch)/len(tgmatch),sum(tpngsave)/len(tpngsave)
#        print(tfile,tpipecand,tmobj,tggrab,pngavg,mlavg,pngavg2,tingestavg,cingestavg,gladeavg,mobjmatchavg,tgmatchavg,pngsave)
#            print(tfile,tpipecand,tmobj,tggrab,pngavg,mlavg,pngavg2,mobjmatchavg,tgmatchavg,tingestavg,gladeavg,cingestavg,pngsave)
#            print('fieldcheck',field)
            fieldcheck(field)
            tcomp=time.time()-imgt0
            print('    Time to complete '+rawfile+': ',tcomp,float(len(image_data))/tcomp,'cand/sec')
        newsql.setingestedfiles(rawfile)

class FileWatcher(FileSystemEventHandler,object):

    def __init__(self, queue): #parameters needed for action
        self._queue = queue

    def on_created(self, event):
        '''Action to take for new files.

        :param event: new event found
        :type event: event
        '''
        self._queue.apply_async(proc,[event])
#p=subprocess.Popen(['python','mocat_download.py'])
pool = Pool(20) #create pool with given CPUs and queue feeding into action function 
observer = Observer() #create observer
observer.schedule(FileWatcher(pool), '/home/saguaro/data/median/red/', recursive=True) #setup observer
observer.start() #start observe
while True:
    time.sleep(1)

