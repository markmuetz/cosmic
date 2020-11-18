import os
import sys
import datetime as dt
import logging
from pathlib import Path
from ftplib import FTP


logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def range_years_months_days(start_date, end_date):
    current_date = start_date
    years_months_days = []
    while current_date < end_date:
        years_months_days.append((current_date.year, current_date.month, current_date.day))
        current_date += dt.timedelta(days=1)
    return years_months_days


def range_years_months(start_date, end_date):
    curr_year_month = (start_date.year, start_date.month)
    end_year_month = (end_date.year, end_date.month)
    years_months = []
    while curr_year_month < end_year_month:
        years_months.append(curr_year_month)
        next_year, next_month = curr_year_month[0], curr_year_month[1] + 1
        if next_month == 13:
            next_year += 1
            next_month = 1
        curr_year_month = (next_year, next_month)
    return years_months


class CmorphDownloader():
    def __init__(self, download_dir):
        self.download_dir = Path(download_dir)
        if not self.download_dir.exists():
            logger.info(f'Creating {self.download_dir}')
            self.download_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, ftp_filepath, local_filepath):
        print(f'downloading {ftp_filepath} to {local_filepath}')
        ftp = FTP('ftp.cpc.ncep.noaa.gov')
        ftp.login()
        with open(local_filepath, 'wb') as fp:
            ftp.retrbinary(f'RETR {ftp_filepath}', fp.write)
        ftp.quit()

    def download_range_0p25deg_3hrly(self,
                                     start_date=dt.datetime(1998, 1, 1),
                                     end_date=dt.datetime.now()):
        for year, month, day in range_years_months_days(start_date, end_date):
            self.download_0p25deg_3hrly(year, month, day)

    def download_0p25deg_3hrly(self, year, month, day):
        # filename = Path(f'CMORPH_V1.0_ADJ_0.25deg-3HLY_{year}{month:02}{day:02}.bz2')
        filename = Path(f'CMORPH_V1.0_ADJ_0.25deg-3HLY_{year}{month:02}{day:02}.bz2')
        compressed_filepath = self.download_dir / filename

        if compressed_filepath.exists():
            logger.info(f'{compressed_filepath} exists: skipping')
            return

        logger.info(f'downloading file: {filename.stem}')
        ftp_filepath = f'/precip/CMORPH_V1.0/CRT/0.25deg-3HLY/{year}/{year}{month:02}/{filename}'

        self._download(ftp_filepath, compressed_filepath)

    def download_8km_30min(self, year, month, filepath):
        filename = Path(f'CMORPH_V1.0_ADJ_8km-30min_{year}{month:02}.tar')

        if filepath.exists():
            logger.info(f'{filepath} exists: skipping')
            return

        logger.info(f'downloading file: {filename.stem}')
        ftp_filepath = f'/precip/CMORPH_V1.0/CRT/8km-30min/{year}/{filename}'

        self._download(ftp_filepath, filepath)
