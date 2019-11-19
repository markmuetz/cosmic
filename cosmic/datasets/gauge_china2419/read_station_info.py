from pathlib import Path

import pandas as pd

BASEDIR = Path('/home/markmuetz/Datasets/gauge_china2419/pre/SURF_CLI_CHN_PRE_MUT_HOMO/SURF_CLI_CHN_PRE_MUT_HOMO/documents')


def read_station_info():
    df_station_info = pd.read_excel(BASEDIR / 'SURF_CLI_CHN_PRE_MUT_HOMO_STATION.xls', 
                                    names=['station_id', 'station_name', 'province', 'level', 'lat_0p01deg', 'lon_0p01deg', 'alt_m', 'alt_p', 'start_date', 'end_date'],
                                    skiprows=1) 
    assert df_station_info.lat_0p01deg.str.endswith('N').sum() == 2419
    assert df_station_info.lon_0p01deg.str.endswith('E').sum() == 2419

    lat_deg_sec = df_station_info.lat_0p01deg.str[:-1].astype(float)
    lon_deg_sec = df_station_info.lon_0p01deg.str[:-1].astype(float)

    df_station_info['lat'] = (lat_deg_sec // 100) + ((lat_deg_sec % 100) / 60)
    df_station_info['lon'] = (lon_deg_sec // 100) + ((lon_deg_sec % 100) / 60)
    return df_station_info
