import datetime as dt
from pathlib import Path

import iris
from metpy.interpolate import interpolate_to_grid
import numpy as np

from .plot_gauge_data import load_jja_gauge_data


def fmt_daterange_jja_year(year):
    return f'{year}06-{year}08'


DATASETS = {
    'cmorph_0p25': ['200906-200908'],
    'cmorph_8km_N1280': [fmt_daterange_jja_year(y) for y in range(1998, 2019)],
    'aphrodite': ['200906-200908'],
    'gauge_china_2419': ['200906-200908'],
    'u-ak543': ['200806-200808'],
    'u-al508': ['200806-200808'],
    'u-am754': ['200806-200808'],
}


def extract_dataset(datadir, dataset, daterange):
    constraint_china = (iris.Constraint(coord_values={'latitude': lambda cell: 18 < cell < 41})
                        & iris.Constraint(coord_values={'longitude': lambda cell: 97.5 < cell < 125}))

    if dataset == 'cmorph_0p25':
        datadir = Path(f'{datadir}/cmorph_data/0.25deg-3HLY')
        season = 'jja'
        filename = f'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_0p1.nc'
        amount_jja = iris.load_cube(f'{datadir}/{filename}',
                                    f'amount_of_precip_{season}')
        amount_jja_china = amount_jja.collapsed('time', iris.analysis.MEAN).extract(constraint_china)
    elif dataset == 'cmorph_8km_N1280':
        datadir = Path(f'{datadir}/cmorph_data/8km-30min')
        season = 'jja'
        filename = f'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_0p1.N1280.nc'
        amount_jja = iris.load_cube(f'{datadir}/{filename}',
                                    f'amount_of_precip_{season}')
        amount_jja_china = amount_jja.collapsed('time', iris.analysis.MEAN).extract(constraint_china)
    elif dataset == 'aphrodite':
        datadir = Path(f'{datadir}/aphrodite_data/025deg')
        amount = iris.load_cube(str(datadir / 'APHRO_MA_025deg_V1901.2009.nc'),
                                ' daily precipitation analysis interpolated onto 0.25deg grids')
        epoch2009 = dt.datetime(2009, 1, 1)
        time_index = np.array([epoch2009 + dt.timedelta(minutes=m) for m in amount.coord('time').points])
        jja = ((time_index >= dt.datetime(2009, 6, 1)) & (time_index < dt.datetime(2009, 9, 1)))
        amount_jja = amount[jja]
        amount_jja_mean = amount_jja.collapsed('time', iris.analysis.MEAN)
        amount_jja_china = amount_jja_mean.extract(constraint_china)
    elif dataset == 'gauge_china_2419':
        df_station_info, df_precip, df_precip_jja, df_precip_station_jja = load_jja_gauge_data(datadir)
        if False:
            lat = df_precip_station_jja.lat * 1.5
        else:
            lat = df_precip_station_jja.lat
        lon = df_precip_station_jja.lon
        precip = df_precip_station_jja.precip

        # Front/back pad each array to include a value that fits with a native 0.25 deg grid.
        lon = np.array([75.125] + list(lon) + [134.375])
        lat = np.array([16.375] + list(lat) + [80.375])
        precip = np.array([0] + list(precip) + [0])

        gx, gy, griddata = interpolate_to_grid(lon, lat, precip,
                                               interp_type='cressman', minimum_neighbors=1,
                                               hres=0.25, search_radius=0.48)
        lat_coord = iris.coords.Coord(gy[:, 0], standard_name='latitude', units='degrees')
        lon_coord = iris.coords.Coord(gx[0], standard_name='longitude', units='degrees')
        coords = [(lat_coord, 0), (lon_coord, 1)]
        amount_jja_mean = iris.cube.Cube(griddata,
                                         long_name='precipitation', units='mm hr-1',
                                         dim_coords_and_dims=coords)
        amount_jja_china = amount_jja_mean.extract(constraint_china)
    elif dataset[:2] == 'u-':
        # UM run:
        runid = dataset[2:7]
        datadir = Path(f'{datadir}/u-{runid}/ap9.pp')
        season = 'jja'
        filename = f'{runid}a.p9{season}.{daterange}.asia_precip.ppt_thresh_0p1.nc'
        amount_jja = iris.load_cube(f'{datadir / filename}',
                                    f'amount_of_precip_{season}')
        amount_jja_china = amount_jja.collapsed('time', iris.analysis.MEAN).extract(constraint_china)

    iris.save(amount_jja_china, f'data/{dataset}_china_amount.{daterange}.nc')
