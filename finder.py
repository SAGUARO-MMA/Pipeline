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

write_path = '/home/saguaro/data/finders/'

date = datetime.datetime.strptime(args.file.split('_')[1],'%Y%m%d').strftime('%Y/%y%b%d/')
sdate=datetime.datetime.strptime(args.file.split('_')[1],'%Y%m%d').strftime('%Y/%m/%d/')

f='/home/saguaro/data/median/raw/'+date+args.file.split('_med')[0]+'_med.fits.fz'
r='/home/saguaro/data/median/ref/'+args.file.split('_')[3]+'/'+args.file.split('_')[3]+'_wcs.fits.fz'
s='/home/saguaro/data/median/red/'+sdate+args.file.split('_med')[0]+'_med_red_Scorr.fits.fz'

print(s)

images=[f,r,s]

fig,_axs=plt.subplots(nrows=1,ncols=3)
fig.subplots_adjust(hspace=0)
axs=_axs.flatten()


    
for i in range(len(images)):
    if not os.path.exists(write_path+args.id+'_finder.png'):
        with fits.open(images[i]) as hdr:
            header = hdr[1].header
            data = hdr[1].data
        header['CTYPE1']= 'RA---TPV'
        header['CTYPE2']= 'DEC--TPV'
        x, y = (wcs.WCS(header)).all_world2pix(np.float(args.ra),np.float(args.dec),1)
        xmin=int(x)-64
        xmax=int(x)+64
        ymin=int(y)-64
        ymax=int(y)+64
        if x<64:xmin=0
        if x>5280-64:xmax=5280
        if y<64:ymin=0
        if y>5280-64:ymax=5280
        thumbnail=(data[ymin:ymax,xmin:xmax])

        axs[i].imshow(thumbnail, cmap='gray', norm=ImageNormalize(data, interval=ZScaleInterval()))
        axs[i].invert_yaxis()
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])
        axs[i].vlines(63.5,70,80,'r',linewidth=1.5)
        axs[i].hlines(63.5,70,80,'r',linewidth=1.5)
        axs[i].vlines(10,10,26,'r',linewidth=3)
        axs[i].hlines(10,10,26,'r',linewidth=3)
        axs[i].text(12,26,'N',color='#000000',fontsize=13)
        axs[i].text(26,10,'E',color='#000000',fontsize=13)
        if i==0:
            axs[i].hlines(10,98,118,'r',linewidth=3)
            axs[i].text(98,16,'30"',color='#000000',fontsize=13)

        if i==2:
            axs[i].text(12,110,'RA:'+str('%.6f'%float(args.ra)),color='#000000',fontsize=13)
            axs[i].text(12,96,'DEC:'+str('%.6f'%float(args.dec)),color='#000000',fontsize=13)

plt.subplots_adjust(wspace=0,hspace=0)
plt.savefig(write_path+args.id+'_finder.png',bbox_inches='tight',pad_inches=0)
plt.close()

