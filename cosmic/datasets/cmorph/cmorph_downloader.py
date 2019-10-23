import os
import datetime as dt
import logging
from pathlib import Path
from ftplib import FTP


logging.basicConfig(level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def range_years_months_days(start_date, end_date):
    current_date = start_date
    years_months_days = []
    while current_date < end_date:
        years_months_days.append((current_date.year, current_date.month, current_date.day))
        current_date += dt.timedelta(days=1)
    return years_months_days


class CmorphDownloader():
    DATASET_TYPES = ['0p25deg_3hrly']

    def __init__(self, download_dir):
        self.download_dir = Path(download_dir)
        if not self.download_dir.exists():
            logger.info(f'Creating {self.download_dir}')
            self.download_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, ftp_filepath, local_filepath):
        ftp = FTP('ftp.cpc.ncep.noaa.gov')
        ftp.login()
        with open(local_filepath, 'wb') as fp:
            ftp.retrbinary(f'RETR {ftp_filepath}', fp.write)
        ftp.quit()

    def download_range_0p25deg_3hrly(self, start_date=dt.datetime(1998, 1, 1), end_date=dt.datetime.now()):
        for year, month, day in range_years_months_days(start_date, end_date):
            self.download_0p25deg_3hrly(year, month, day)

    def download_0p25deg_3hrly(self, year, month, day):
        filename = Path(f'CMORPH_V1.0_ADJ_0.25deg-3HLY_{year}{month:02}{day:02}.bz2')
        compressed_filepath = self.download_dir / filename

        if compressed_filepath.exists():
            logger.info(f'{compressed_filepath} exists: skipping')
            return

        logger.info(f'downloading file: {filename.stem}')
        ftp_filepath = f'/precip/CMORPH_V1.0/CRT/0.25deg-3HLY/{year}/{year}{month:02}/{filename}'

        self._download(ftp_filepath, compressed_filepath)
