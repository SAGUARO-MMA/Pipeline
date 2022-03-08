import glob
from astropy.io import fits
import os, os.path, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval
import astropy.wcs as wcs
import argparse
import psycopg2
import psycopg2.extras
import datetime

params = argparse.ArgumentParser(description='User parameters.')
params.add_argument('--field', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--id', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--ra', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--dec', default=None, help='Telescope of data.') #telescope argument required to run pipeline
args = params.parse_args()

class Dictdb():
    def __init__(self, db="saguaro", user="saguaro"):
        connstring = open('/home/saguaro/software/webapptesting/sql.conn').readline()
        self.conn = psycopg2.connect(connstring)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def query(self, query):
        self.cur.execute(query)

    def queryfetchall(self,query):
        self.cur.execute(query)
        rows = self.cur.fetchall()
        ans = []
        for row in rows:
            ans.append(dict(row))
        ans1={k:[d.get(k) for d in ans] for k in {k for d in ans for k in d}}
        return ans1

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

def getcand(can,field):
    db = Dictdb()
    res=db.queryfetchall("select * from candidates where targetid = '%s' order by obsdate ASC;" % (can))
    db.close()
    return res

def gettarget(can,field):
    db = Dictdb()
    res=db.queryfetchall("select * from targets where targetid = '%s';" % (can))
    db.close()
    return res


def ingest(filename,ra,dec,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,ncombine):
    db = Dictdb()
    print("INSERT INTO candidates (filename, ra,dec,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,ncombine) VALUES (%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s);" % (filename,ra,dec,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,ncombine))
    db.query("INSERT INTO candidates (filename, ra,dec,mag,magerr,rawfilename,obsdate,field,classification,cx,cy,cz,htm16id,targetid,mjdmid,ncombine) VALUES (%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s);" % ("'"+filename+"'",ra,dec,mag,magerr,"'"+rawfilename+"'","'"+obsdate+"'","'"+field+"'",classification,cx,cy,cz,htm16id,targetid,mjdmid,ncombine))
    db.commit()
    db.close()

candidate_info = getcand(args.id,args.field)
alllimit=0
alldates=[]
if candidate_info=={}:
    candidate_info = gettarget(args.id,args.field)
    alllimit=1
else:
    for i in range(len(candidate_info['obsdate'])):
        alldates.append(candidate_info['obsdate'][i].strftime("%Y%m%d"))
    print(candidate_info['obsdate'])
    print(alldates)
files = np.sort(glob.glob('/home/saguaro/data/median/raw/*/*/*'+args.field+'*_med.fits.fz'))
print(files)
dirname = []
filename = []
non_date=None
for f in files:
    dirname.append(os.path.dirname(f))
    filename.append(os.path.basename(f))
zipped = list(zip(dirname,filename))
zipped.sort(key=lambda t: t[1])
sorted_files = [zipped[i][0]+'/'+zipped[i][1] for i in range(len(zipped))]
sorted_files=sorted_files[::-1]##added MJL
print('')
print('')
print(sorted_files,len(sorted_files),range(len(sorted_files)))
for i in range(len(sorted_files)):
    date = sorted_files[i].split('G96_')[1].split('_')[0]
    print(i,date,date not in alldates) #,canddaterev[i].strftime("%Y%m%d"))
    if date not in alldates:           # or date!=canddaterev[i].strftime("%Y%m%d"):
        print(date,' not in ',alldates)
        date = sorted_files[i].split('G96_')[1].split('_')[0]###editted from   date = f.split('G96_')[1].split('_')[0] MJL
        print('/home/saguaro/data/median/red/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+os.path.basename(sorted_files[i]).replace('_med','_med_red_Scorr'))
        print(os.path.exists('/home/saguaro/data/median/red/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+os.path.basename(sorted_files[i]).replace('_med','_med_red_Scorr')))
        if os.path.exists('/home/saguaro/data/median/red/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+os.path.basename(sorted_files[i]).replace('_med','_med_red_Scorr')):
            non_date = sorted_files[i]
            with fits.open(non_date) as hdr:
                header = hdr[1].header
                data = hdr[1].data
            zp = header['MAGZP']
            header['CTYPE1']= 'RA---TPV'
            header['CTYPE2']= 'DEC--TPV'
            x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
            write_path = '/home/saguaro/data/png/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+args.field+'/'
            if not os.path.exists(write_path):os.makedirs(write_path)
            if non_date:
                with fits.open(non_date) as hdr:
                    header = hdr[1].header
                    data = hdr[1].data
                zp = header['MAGZP']
                header['CTYPE1']= 'RA---TPV'
                header['CTYPE2']= 'DEC--TPV'
                x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
                thumbnail_new = data[int(y)-32:int(y)+32,int(x)-32:int(x)+32]
                plt.imshow(thumbnail_new, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
                plt.xticks([], [])
                plt.yticks([], [])
                plt.gca().invert_yaxis()
                plt.savefig(write_path+os.path.basename(non_date).replace('_med.fits.fz','')+'_img.png',bbox_inches='tight')
                plt.close()
                with fits.open('/home/saguaro/data/median/ref/'+args.field+'/'+args.field+'_wcs.fits.fz') as hdr:
                    header = hdr[1].header
                    data = hdr[1].data
                zp = header['MAGZP']
                header['CTYPE1']= 'RA---TPV'
                header['CTYPE2']= 'DEC--TPV'
                x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
                thumbnail_ref = data[int(y)-32:int(y)+32,int(x)-32:int(x)+32]
                plt.imshow(thumbnail_ref, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
                plt.xticks([], [])
                plt.yticks([], [])
                plt.gca().invert_yaxis()
                plt.savefig(write_path+os.path.basename(non_date).replace('_med.fits.fz','')+'_ref.png',bbox_inches='tight')
                plt.close()
                print(write_path+os.path.basename(non_date).replace('_med.fits.fz','')+'_ref.png')
                with fits.open('/home/saguaro/data/median/red/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+os.path.basename(non_date).replace('_med','_med_red_Scorr')) as hdr:
                    header = hdr[1].header
                    data = hdr[1].data
                zp = header['MAGZP']
                header['CTYPE1']= 'RA---TPV'
                header['CTYPE2']= 'DEC--TPV'
                x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
                thumbnail_scorr = data[int(y)-32:int(y)+32,int(x)-32:int(x)+32]
                plt.imshow(thumbnail_scorr, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
                plt.xticks([], [])
                plt.yticks([], [])
                plt.gca().invert_yaxis()

                plt.savefig(write_path+os.path.basename(non_date).replace('_med.fits.fz','')+'_scorr.png',bbox_inches='tight')
                plt.close()
                trans_cat = '/home/saguaro/data/median/red/'+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/'+os.path.basename(non_date).replace('_med','_med_red_trans')
                with fits.open(trans_cat) as hdr:
                    mjd = hdr[1].header['MJD']
                    ncombine = hdr[1].header['NCOMBINE']
                    data = hdr[1].data
                upper_mag = np.max(data['MAG_PSF'])
                ingest(os.path.basename(non_date).replace('_med','_med_red_trans'),args.ra,args.dec,upper_mag,0,os.path.basename(non_date),date[0:4]+'-'+date[4:6]+'-'+date[6:8],args.field,10,candidate_info['cx'][0],candidate_info['cy'][0],candidate_info['cz'][0],candidate_info['htm16id'][0],args.id,mjd,ncombine)
            
            
