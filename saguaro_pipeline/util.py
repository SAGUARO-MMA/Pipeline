from datetime import datetime, timedelta
import importlib
import os
import subprocess
import time


def copying(file):
    """
    Function that waits until the given file size is no longer changing before returning.
    This ensures the file has finished copying before the file is accessed.
    """
    copying_file = True  # file is copying
    size_earlier = -1  # set inital size of file
    while copying_file:
        size_now = os.path.getsize(file)  # get current size of file
        if size_now == size_earlier:  # if the size of the file has not changed, return
            return
        else:  # if the size of the file has changed
            size_earlier = os.path.getsize(file)  # get new size of file
            time.sleep(1)  # wait


def funpack_file(file):
    """
    Funpack file and return new name.
    """
    subprocess.call(['funpack', '-D', file])
    return file.replace('.fz', '')


def scheduled_exit(start_time, tel):
    """
    Checks current time against the scheduled exit time for the night pipeline.
    If the current time has past the scheduled exit time, return True.
    """
    if start_time.hour < tel.stop_hour():
        stop_date = start_time.date()
    else:
        stop_date = start_time.date() + timedelta(days=1)
    stop_time = datetime(stop_date.year, stop_date.month, stop_date.day, tel.stop_hour())
    return datetime.utcnow() > stop_time
