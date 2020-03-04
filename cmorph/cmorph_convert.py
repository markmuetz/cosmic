import sys
import datetime as dt

from cosmic.util import load_module
from cosmic.datasets.cmorph.cmorph_convert import convert_to_netcdf4_month


def main(basedir, year):
    for month in range(1, 13):
        print(f'{year}, {month}')
        data_dir = basedir / f'raw/precip_{year}{month:02}'
        output_dir = basedir / f'precip_{year}{month:02}'
        end_year = year
        end_month = month + 1
        if end_month == 13:
            end_year += 1
            end_month = 1
        convert_to_netcdf4_month(data_dir, output_dir, year, month)


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config.BASEDIR, config.SCRIPT_ARGS[config_key])
