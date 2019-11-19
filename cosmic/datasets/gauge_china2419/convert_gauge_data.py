from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

BASEDIR = Path('/home/markmuetz/Datasets/gauge_china2419/pre/SURF_CLI_CHN_PRE_MUT_HOMO/SURF_CLI_CHN_PRE_MUT_HOMO')


def read_station_precip_data(basedir):
    filenames = sorted((basedir / 'datasets/DAY').glob('SURF_CLI_CHN_PRE_MUT_HOMO-DAY-?????.txt'))

    dfs = []
    for filename in filenames:
        station_id = filename.stem[-5:]
        df_station = pd.read_csv(filename, header=None, delim_whitespace=True, names=['year', 'month', 'day', 'precip'])
        df_station['station_id'] = int(station_id)
        dfs.append(df_station)
        
    df_precip = pd.concat(dfs)
    df_precip['datetime'] = pd.to_datetime(df_precip[['year', 'month', 'day']])
    return df_precip


def read_station_info(basedir):
    df_station_info = pd.read_excel(basedir / 'documents' / 'SURF_CLI_CHN_PRE_MUT_HOMO_STATION.xls', 
                                    names=['station_id', 'station_name', 'province', 'level', 'lat_0p01deg', 'lon_0p01deg', 'alt_m', 'alt_p', 'start_date', 'end_date'],
                                    skiprows=1) 
    assert df_station_info.lat_0p01deg.str.endswith('N').sum() == 2419
    assert df_station_info.lon_0p01deg.str.endswith('E').sum() == 2419

    lat_deg_sec = df_station_info.lat_0p01deg.str[:-1].astype(float)
    lon_deg_sec = df_station_info.lon_0p01deg.str[:-1].astype(float)

    df_station_info['lat'] = (lat_deg_sec // 100) + ((lat_deg_sec % 100) / 60)
    df_station_info['lon'] = (lon_deg_sec // 100) + ((lon_deg_sec % 100) / 60)
    return df_station_info


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('basedir', action='store_const', const=BASEDIR)
    args = parser.parse_args()
    basedir = Path(args.basedir)

    df_station_info = read_station_info(basedir)
    df_precip = read_station_precip_data(basedir)

    df_station_info.to_hdf(basedir / 'station_data.hdf', 'station_info')
    df_precip.to_hdf(basedir / 'station_data.hdf', 'precip')
