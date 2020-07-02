import os
from pathlib import Path
import pickle

from remake import TaskControl, Task, remake_task_control

from cosmic.config import PATHS
from cosmic.datasets.aphrodite import AphroditeDownloader, ALL_YEARS, FILE_TPL


def download_year(inputs, outputs, year, datadir):
    with open(os.path.expandvars('$HOME/.aphrodite_credentials.pkl'), 'rb') as f:
        aphrodite_credentials = pickle.load(f)
    downloader = AphroditeDownloader(datadir, **aphrodite_credentials)
    downloader.download(year)


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    datadir = PATHS['datadir'] / 'aphrodite_data/025deg'
    for year in ALL_YEARS:
        output_path = datadir / FILE_TPL.format(year=year)
        tc.add(Task(download_year, [], [output_path], func_args=(year, datadir), atomic_write=False))
    return tc
