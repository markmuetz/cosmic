from pathlib import Path

import pandas as pd

BASEDIR = Path('/home/markmuetz/Datasets/gauge_china2419/pre/SURF_CLI_CHN_PRE_MUT_HOMO/SURF_CLI_CHN_PRE_MUT_HOMO/datasets/DAY/')


def read_station_precip_data():
    filenames = sorted(BASEDIR.glob('SURF_CLI_CHN_PRE_MUT_HOMO-DAY-?????.txt'))

    dfs = []
    for filename in filenames:
        station_id = filename.stem[-5:]
        df_station = pd.read_csv(filename, header=None, delim_whitespace=True, names=['year', 'month', 'day', 'precip'])
        df_station['station_id'] = int(station_id)
        dfs.append(df_station)
        
    df_precip = pd.concat(dfs)
    df_precip['datetime'] = pd.to_datetime(df_precip[['year', 'month', 'day']])
    return df_precip

