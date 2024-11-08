# SAGUARO Pipeline

The SAGUARO pipeline is designed to run every night, automatically started by a cron job, to reduce incoming data.

The main pipeline script is `saguaro-pipe`, and runs together with the telescope setting file (e.g. `css.py` for the 1.5 m CSS telescope on Mt. Lemmon) to reduce new data through a watchdog observer (the location of the folder with new data is defined in the setting file). The pipeline will reduce the data according to the options defined in the setting file as well as creating a mask for each image.

The pipeline will then look for a reference file (the location of which is defined in the setting file) which it can use for subtraction using [ZOGY](https://github.com/KerryPaterson/ZOGY). If a reference image is found, the pipeline submits the relevant files as a subtraction job to ZOGY. If no reference is found, the image is submitting to ZOGY as a reference job. Once ZOGY has completed, the pipeline will move the needed files to the relevant folders. The pipeline includes a complete log of its operations, and can be run on multiple CPUs.

The parameters which can be set when starting the pipeline are:
1) **telescope** - the name of the setting file the pipeline will load (in the same folder)
2) **date** - the date of the data you wish to run on (will automatically run the same night's data if not set, i.e., it will run in real time)
3) **cpu** - the number of processes you wish to run across, i.e., how many images to process at the same time (max number is the number of CPUs)

These parameters can be set use `--` when calling the python script, e.g., `saguaro-pipe --telescope css --date 2020/01/02 --cpu 4`.

Other scripts housed here includes the median watcher (`median-watcher`) for the CSS telescope which is used to create a median combined image (from the 4 individual exposures) for `saguaro-pipe` to reduce for subtraction.

## Installation
You can now pip install the pipeline. First activate Anaconda, if you haven't already:
```bash
source /dataraid6/sassy/anaconda/bin/activate
```

Then create a new Python 3.8[^1] environment and install pip:
```bash
conda create -n saguaro-mma python=3.8 pip
```

[^1]: Python 3.8 is required to support scikit-learn 0.22.1, which is what we used to produce the old machine-learning model.

Activate that environment:
```bash
conda activate saguaro-mma
```

Finally, install the pipeline in that environment:
```bash
pip install git+https://github.com/SAGUARO-MMA/Pipeline
```

This will also install [ZOGY](https://github.com/KerryPaterson/ZOGY), which has non-Python dependencies: [PSFEx](http://www.astromatic.net/software/psfex), [Source Extractor](http://www.astromatic.net/software/sextractor), and [SWarp](http://www.astromatic.net/software/swarp). The pipeline itself also requres [CFITSIO](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/fitsio.html), which contains `fpack` and `funpack`. These can all be installed with APT:
```bash
sudo apt install psfex sextractor swarp libcfitsio-bin
```

APT installs some of these with slightly different names than the pipeline uses, so make these aliases:
```bash
cd /usr/bin/
sudo ln -s sextractor sex
sudo ln -s SWarp swarp
```

You might also have to install some or all of the following linear algebra packages, depending on what comes with your system:
```bash
sudo apt install libatlas-base-dev liblapack-dev libblas-dev libfftw3-devlibplplot-dev
```

When using the pipeline in the future, make sure to activate the environment first:
```bash
conda activate saguaro-mma
```

## Environment Variables
The pipeline needs the following environment variables to run:
```bash
ML_MODEL_NEW=/dataraid6/sassy/Pipeline/model_onlyscorr16_ml  # optional, the default is included in the package
ML_MODEL_OLD=/dataraid6/sassy/Pipeline/rf_model.ml
SAGUARO_ROOT=/dataraid6/sassy
SLACK_API_TOKEN=xoxb-********
THUMB_PATH=/dataraid6/sassy/data/png
```

as well as the PostgreSQL credentials for ingestion:
```bash
PGHOST
PGDATABASE
PGUSER
PGPASSWORD
PGPORT
```

You could add all of these to your `.bashrc` file (e.g., `export SAGUARO_ROOT=/dataraid6/sassy`) so you don't have to set them every time you use the pipeline.

## Cronjobs
In order to run the pipeline automatically, we use the following cronjobs:
```
[list all the environment variables from above]

30 17 * * * /dataraid6/sassy/anaconda/envs/saguaro-mma/bin/median-watcher > median_watcher.log 2>&1
30 17 * * * /dataraid6/sassy/anaconda/envs/saguaro-mma/bin/saguaro-pipe --telescope css --cpu 16 > saguaro_pipe.log 2>&1
```

Then, because we don't want to fill up the disk, we do a weekly backup of the data products to a network drive (as root). We do this at noon so that the pipeline is not running at the same time:
```
0 12 * * 0 cd bash /root/bin/css.sh >> /root/bin/css.log 2>&1
```

The content of css.sh is subject to local configurations but, at it's simplest, we use:
```
echo "CSS start: " `date`

# FITS cleanup
cd /dataraid6/sassy/data/css 
rsync -arv --remove-source-files --progress -e "ssh -i /root/.ssh/id_rsa" inc log raw red pndaly@10.130.133.220:/mnt/tank/dsand/saguaro/data/css
if [ $? != 0 ]; then
  echo "ERROR: failed to rsync /mnt/dsand/saguaro/data/css"
  exit -3
else
  rm -rv tmp/*
fi

# new thumbnail cleanup
cd /dataraid6/sassy/data
rsync -arv --remove-source-files -e "ssh -i /root/.ssh/id_rsa" png pndaly@10.130.133.220:/mnt/tank/dsand/saguaro/data
if [ $? != 0 ]; then
  echo "ERROR: failed to rsync /mnt/dsand/saguaro/data/png"
  exit -3
fi

# old thumbnail compression (to save inodes)
cd /mnt/dsand/saguaro/data/png/$(date -d "-13 months" +%Y)
LAST_MONTH=`date -d "-13 months" +%m`
if [ ! -e $LAST_MONTH.tar.gz ]; then
  tar --remove-files -acvf $LAST_MONTH.tar.gz $LAST_MONTH
fi

echo "CSS end:   " `date`
```


