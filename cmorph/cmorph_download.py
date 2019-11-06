import sys
import datetime as dt

from cosmic.util import load_config
from cosmic.datasets.cmorph.cmorph_downloader import CmorphDownloader


def main(year):
    for month in range(1, 13):
        print(f'{year}, {month}')
        dl = CmorphDownloader(f'cmorph_data/raw/precip_{year}{month:02}')
        end_year = year
        end_month = month + 1
        if end_month == 13:
            end_year += 1
            end_month = 1
        dl.download_range_0p25deg_3hrly(dt.datetime(year, month, 1), 
                                        dt.datetime(end_year, end_month, 1))


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = int(sys.argv[2])
    main(config.SCRIPT_ARGS[config_key])
