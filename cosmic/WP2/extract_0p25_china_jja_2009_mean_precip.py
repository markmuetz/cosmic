from argparse import ArgumentParser
import datetime as dt
from pathlib import Path

import iris
from metpy.interpolate import interpolate_to_grid
import numpy as np

from plot_gauge_data import load_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('basepath')
    parser.add_argument('dataset')
    args = parser.parse_args()

    constraint_china = (iris.Constraint(coord_values={'latitude':lambda cell: 18 < cell < 41}) 
                        & iris.Constraint(coord_values={'longitude':lambda cell: 97.5 < cell < 125}))

    if Path('data/cmorph_0p25_china_jja_2009_amount.nc').exists():
        amount_jja_china_ref = iris.load_cube('data/cmorph_0p25_china_jja_2009_amount.nc')

    if args.dataset == 'cmorph_0p25':
        datadir = Path(f'{args.basepath}/cmorph_data/0.25deg-3HLY')
        season = 'jja'
        filename = f'cmorph_ppt_{season}.200906-200908.asia_precip.ppt_thresh_0p1.nc'
        amount_jja = iris.load_cube(f'{datadir}/{filename}',
                                    f'amount_of_precip_{season}')
        amount_jja_china = amount_jja.collapsed('time', iris.analysis.MEAN).extract(constraint_china)
    elif args.dataset == 'aphrodite':
        datadir = Path(f'{args.basepath}/aphrodite_data/025deg')
        amount = iris.load_cube(str(datadir / 'APHRO_MA_025deg_V1901.2009.nc'), 
                                ' daily precipitation analysis interpolated onto 0.25deg grids')
        epoch2009 = dt.datetime(2009, 1, 1)
        time_index = np.array([epoch2009 + dt.timedelta(minutes=m) for m in amount.coord('time').points])
        jja = ((time_index >= dt.datetime(2009, 6, 1)) & (time_index < dt.datetime(2009, 9, 1)))
        amount_jja = amount[jja]
        amount_jja_mean = amount_jja.collapsed('time', iris.analysis.MEAN)
        amount_jja_china = amount_jja_mean.extract(constraint_china)
    elif args.dataset == 'gauge_china_2419':
        datadir = Path(f'{args.basepath}')
        df_station_info, df_precip, df_precip_jja, df_precip_station_jja = load_data(datadir)
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

    iris.save(amount_jja_china, f'data/{args.dataset}_china_jja_2009_amount.nc')



    print(args)
    lon_min, lon_max = (97.5, 125)
    lat_min, lat_max = (18, 41)

