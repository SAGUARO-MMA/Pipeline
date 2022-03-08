import glob
from astropy.io import fits
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval
import astropy.wcs as wcs
import argparse
import datetime

params = argparse.ArgumentParser(description='User parameters.')
params.add_argument('--file', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--id', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--ra', default=None, help='Telescope of data.') #telescope argument required to run pipeline
params.add_argument('--dec', default=None, help='Telescope of data.') #telescope argument required to run pipeline
args = params.parse_args()

write_path = '/home/saguaro/data/thumbnails/'+args.id+'/'
if not os.path.exists(write_path):
    os.makedirs(write_path)

date = datetime.datetime.strptime(args.file.split('_')[1],'%Y%m%d').strftime('%Y/%y%b%d/')
files = glob.glob('/home/data/css/G96/'+date+args.file.split('_med')[0]+'*.calb.fz')

if len(files) == 0:
    files = glob.glob('/home/data/css/G96/'+date+args.file.split('_med')[0]+'*.arch.fz')

print(files)
#header = fits.open(f.replace('calb.fz','arch_h'))[0].header

for f in files:    
    if not os.path.exists(write_path+os.path.basename(f).replace('.calb.fz','.png')) or not os.path.exists(write_path+os.path.basename(f).replace('arch.fz','.png')):
        with fits.open(f) as hdr:
            hdr.verify('fix+ignore')
            if 'calb.fz' in f:
                header = fits.open(f.replace('calb.fz','arch_h'))[0].header
                typ = 'new'
            else:
                header = hdr[1].header
                typ = 'old'
            header['CTYPE1']= 'RA---TPV'
            header['CTYPE2']= 'DEC--TPV'
            data = hdr[1].data
#            hdul = fits.HDUList([fits.PrimaryHDU(hdr[1].data,header)])
#            data = hdul[1].data

        x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
        xmin=int(x)-64
        xmax=int(x)+64
        ymin=int(y)-64
        ymax=int(y)+64
        xdiff=0
        ydiff=0
        if x<64:xmin,xdiff=0,64-int(x)
        if x>5280-64:xmax,xdiff=5280,int(x)-5280
        if y<64:ymin,ydiff=0,64-int(y)
        if y>5280-64:ymax,ydiff=5280,int(y)-5280
        thumbnail=(data[ymin:ymax,xmin:xmax])
        plt.imshow(thumbnail, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().invert_yaxis()
        plt.vlines(63.5-xdiff,70-ydiff,80-ydiff,'r',linewidth=1.5)
        plt.hlines(63.5-ydiff,70-xdiff,80-xdiff,'r',linewidth=1.5)
        plt.vlines(10,10,20,'r',linewidth=3)
        plt.hlines(10,10,20,'r',linewidth=3)
        plt.text(12,20,'N',fontsize=13)
        plt.text(22,10,'E',fontsize=13)
        if typ == 'new':
            plt.savefig(write_path+os.path.basename(f).replace('.calb.fz','.png'),bbox_inches='tight',pad_inches=0)
        else:
            plt.savefig(write_path+os.path.basename(f).replace('.arch.fz','.png'),bbox_inches='tight',pad_inches=0)
        plt.close()

