import os
import gzip
from pathlib import Path
import shutil

from cosmic.util import sysrun


class AphroditeDownloader:
    URL = 'http://aphrodite.st.hirosaki-u.ac.jp/product/APHRO_V1901/APHRO_MA/025deg_nc/'
    FILE_TPL = 'APHRO_MA_025deg_V1901.{year}.nc.gz'
    ALL_YEARS = list(range(1998, 2016))

    def __init__(self, datadir, username, password):
        self.datadir = Path(datadir)
        if not self.datadir.exists():
            raise Exception(f'{datadir} does not exist')
        self.username = username
        self.password = password
        self.output = []

    def download_all(self):
        for year in self.ALL_YEARS:
            self.download(year)

    def download(self, year):
        cwd = Path.cwd()
        os.chdir(self.datadir)

        filename = self.FILE_TPL.format(year=year)
        path = Path(filename)
        newpath = Path(path.stem)
        if newpath.exists():
            print(f'Skipping (already exists): {newpath}')
            os.chdir(cwd)
            return
        url = self.URL + filename
        self.output.append(sysrun(f'wget --user={self.username} --password={self.password} {url}'))
        with gzip.open(filename, 'rb') as f_in:
            with open(newpath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        path.unlink()

        os.chdir(cwd)

