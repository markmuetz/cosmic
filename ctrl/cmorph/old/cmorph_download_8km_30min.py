import sys
import datetime as dt
from pathlib import Path

from cosmic.util import load_module
from cosmic.datasets.cmorph.cmorph_downloader import CmorphDownloader

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data/8km-30min/raw')


def main(year):
    for month in range(1, 13):
        print(f'{year}, {month}')
        dl = CmorphDownloader(BASEDIR / f'precip_{year}{month:02}')
        end_year = year
        end_month = month + 1
        if end_month == 13:
            end_year += 1
            end_month = 1
        dl.download_range_8km_30min(dt.datetime(year, month, 1), 
                                    dt.datetime(end_year, end_month, 1))


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config.SCRIPT_ARGS[config_key])
    # main(int(sys.argv[1]))
