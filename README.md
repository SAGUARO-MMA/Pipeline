# Pipeline

The SAUARO pipeline is designed to run every night, automatically started by a cron job, to reduce incoming data.

The main pipeline script is saguaro_pipe.py, and runs together with the telescope setting file (e.g. css.py for the 1.5 m CSS telescope on Mt Lemmon) to reduce new data through a watchdog observer (the location of the folder with new data is defined in the setting file). The pipeline will reduce the data according to the options defined in the setting file as well as creating a mask for each image.

The pipeline will then look for a reference file (the location of which is defined in the setting file) which it can use for subtraction using ZOGY (). If a reference image is found, the pipeline submits the relevant files as a subtraction job to ZOGY. If no reference is found, the image is submitting to ZOGY as a reference job. Once ZOGY has completed, the pipeline will move the needed files to the relevant folders. The pipeline includes a complete log of its operations, and can be run on multiple CPUs.

The parameters which can be set when starting the pipeline are:
1) telescope - the name of the setting file the pipeline will load (in the same folder)
2) date - the date of the data you wish to run on (will automatically run the same night's data if not set i.e. it will run in real time)
3) cpu - the number of processes you wish to run across i.e. how many images to process at the same time (max number is the number of CPUs)

These parameters can be set use "--" when calling the python script e.g. python sauargo_pipe.py --telescope css --date 2020/01/02 --cpu 4

Other scripts housed here includes the medain watcher (median_watcher.py) for the CSS telescope which is used to create a median combined image (from the 4 individual exposures) for sauargo_pipe.py to reduce for subtraction.
