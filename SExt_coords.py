import glob
import os
import numpy as np
from astropy.coordinates import SkyCoord
import argparse
import datetime
from astropy.io import fits
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.visualization import ImageNormalize, ZScaleInterval
import astropy.wcs as wcs

params = argparse.ArgumentParser(description='User parameters.')
params.add_argument('--file', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--id', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--ra', type=float, default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--dec', type=float, default=None, help='Telescope of data.') #telescope argument required to run pipeline
args = params.parse_args()

write_path = '/home/saguaro/data/coordinates/'
if not os.path.exists(write_path):
    os.makedirs(write_path)

date = datetime.datetime.strptime(args.file.split('_')[1],'%Y%m%d').strftime('%Y/%y%b%d/')
files = sorted(glob.glob('/home/data/css/G96/'+date+args.file.split('_med')[0]+'*.sext.gz'))

radius = 0.01
ra, dec, mag, flag = np.loadtxt(files[0],usecols=(11,12,3,7),unpack=True)
ra_ref = ra[(ra<args.ra+2*radius)&(ra>args.ra-2*radius)&(dec<args.dec+2*radius)&(dec>args.dec-2*radius)]
dec_ref = dec[(ra<args.ra+2*radius)&(ra>args.ra-2*radius)&(dec<args.dec+2*radius)&(dec>args.dec-2*radius)]
mag_ref = mag[(ra<args.ra+2*radius)&(ra>args.ra-2*radius)&(dec<args.dec+2*radius)&(dec>args.dec-2*radius)]
ref_cat = SkyCoord(ra_ref*u.deg, dec_ref*u.deg,frame='fk5')

diff = {}
for j in range(len(files)-1):
    r, d = np.loadtxt(files[j+1],usecols=(11,12),unpack=True)
    ra_new = r[(r<args.ra+2*radius)&(r>args.ra-2*radius)&(d<args.dec+2*radius)&(d>args.dec-2*radius)]
    dec_new = d[(r<args.ra+2*radius)&(r>args.ra-2*radius)&(d<args.dec+2*radius)&(d>args.dec-2*radius)]
    cat = SkyCoord(ra_new*u.deg, dec_new*u.deg,frame='fk5')
    idx, d2, d3 = ref_cat.match_to_catalog_sky(cat)
    for i,n in enumerate(idx):
        if mag_ref[i]<=19:
            add = True
        else:
            c1 = SkyCoord(ra_ref[i]*u.deg, dec_ref[i]*u.deg, frame='fk5')
            c2 = SkyCoord(args.ra*u.deg, args.dec*u.deg, frame='fk5')
            sep = c1.separation(c2)
            if sep.deg < 3./3600:
                add = True
            else:
                add = False
        if add:
            try:
                diff[i]
            except KeyError:
                diff.update({i:[]})
                diff[i].append(ra_ref[i])
                diff[i].append(dec_ref[i])
            if d2[i].deg < 8./3600:
                diff[i].append(d2[i].value*3600)
            else:
                diff[i].append(np.nan)   

for i in diff:
    std = np.nanstd(diff[i][2:4])
    diff[i].append(std)
    if len(diff[i]) >=2:
        if std > 0.3:
            diff[i].append(1)
        else:
            diff[i].append(0)
    else:
        diff[i].append(-1)

f = files[0].replace('.sext.gz','.arch_h')
try:
    with fits.open(f) as hdr:
        header = hdr[0].header
    f = files[0].replace('.sext.gz','.calb.fz')
    with fits.open(f) as hdr:
        hdr.verify('fix+ignore')
        data = hdr[1].data
except:
    f = files[0].replace('.sext.gz','.arch.fz')
    with fits.open(f) as hdr:
        hdr.verify('fix+ignore')
        header = hdr[1].header
        data = hdr[1].data
header['CTYPE1']= 'RA---TPV'
header['CTYPE2']= 'DEC--TPV'
x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
thumbnail = data[int(y)-64:int(y)+64,int(x)-64:int(x)+64]
plt.imshow(thumbnail, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
plt.xticks([], [])
plt.yticks([], [])
plt.vlines(63.5,70,80,'r',linewidth=1.5)
plt.hlines(63.5,70,80,'r',linewidth=1.5)
for i in diff:
    xi, yi = (wcs.WCS(header)).all_world2pix(np.float(diff[i][0]),np.float(diff[i][1]),1)
    plt.gca().add_patch(patches.Circle((xi-int(x)+63,yi-int(y)+63),radius=4,edgecolor='g',facecolor='none',linewidth=2))
    plt.text(xi-int(x)+63+4,yi-int(y)+63+4,str(i),color='k')
plt.gca().invert_yaxis()
plt.text(130,124,'Object  | RA             | Dec          | d1 (arcsec) | d2 (arcsec) | d3 (arcsec) | std (arcsec) | MO |')
plt.text(130,120,'--------------------------------- ------------------------------------------------------------------------------------------------')
for i in diff:
    try:
        plt.text(130,116-5*i,'%5.0f      | %6.5f  | %6.5f  |%10.3f      |%10.3f      |%10.3f      |%10.3f      |%4.0f  |'%(i,diff[i][0],diff[i][1],diff[i][2],diff[i][3],diff[i][4],diff[i][5],diff[i][6]))
    except:
        plt.text(130,116-5*i,'Failed for '+str(i)+', diff = '+str(diff[i]))
if len(diff) == 0:
    plt.text(130,116,'Only one image found, can not run comparison.')
plt.text(130,110-5*i,'---------------------------------------------------------------------------------------------------------------------------------')
plt.savefig(write_path+args.id+'.png',bbox_inches='tight',pad_inches=0)
plt.close()
