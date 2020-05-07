import os
from pathlib import Path
import pickle

from remake import TaskControl, Task, remake_task_control

from cosmic.datasets.aphrodite.aphrodite_downloader import AphroditeDownloader

DATADIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/aphrodite_data/025deg/')
# DATADIR = Path('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/aphrodite_data/025deg/')


def download_year(inputs, outputs, year):
    with open(os.path.expandvars('$HOME/.aphrodite_credentials.pkl'), 'rb') as f:
        aphrodite_credentials = pickle.load(f)
    downloader = AphroditeDownloader(DATADIR, **aphrodite_credentials)
    downloader.download(year)


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    for year in AphroditeDownloader.ALL_YEARS:
        output_path = DATADIR / Path(AphroditeDownloader.FILE_TPL.format(year=year)).stem
        tc.add(Task(download_year, [], [output_path], func_args=(year,), atomic_write=False))
    return tc
