import datetime as dt
from pathlib import Path
import bz2

import numpy as np
import iris


def convert_to_netcdf4_month(data_dir, year, month):
    lat = np.linspace(-59.875, 59.875, 480)
    lon = np.linspace(0.125, 360 - 0.125, 1440)

    epoch = dt.datetime(1970, 1, 1)
    start_time = dt.datetime(year, month, 1, 1, 30)
    if month < 12:
        end_time = dt.datetime(year, month + 1, 1)
    else:
        end_time = dt.datetime(year + 1, 1, 1)
    curr_time = start_time

    times = []
    while curr_time < end_time:
        times.append((curr_time - epoch).total_seconds() / 3600)
        curr_time += dt.timedelta(hours=3)

    lat_coord = iris.coords.Coord(lat, standard_name='latitude', units='degrees')
    lon_coord = iris.coords.Coord(lon, standard_name='longitude', units='degrees')
    time_coord = iris.coords.Coord(times, standard_name='time', 
                                   units=('hours since 1970-01-01 00:00:00'))

    data = load_raw_0p25deg_3hrly_year(data_dir, year, month, '??')

    coords = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cmorph_ppt_cube = iris.cube.Cube(data.reshape(len(times), 480, 1440), 
                                     long_name='precipitation', units='mm hr-1',  
                                     dim_coords_and_dims=coords)

    iris.save(cmorph_ppt_cube, f'data/cmorph_ppt_{year}{month:02}.nc', zlib=True)


def extract_asia(data_dir, year):
    constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
                       & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))
    filenames = sorted(Path(data_dir).glob(f'cmorph_ppt_{year}??.nc'))

    for filename in filenames:
        asia_cmorph_ppt_cube = iris.load_cube(str(filename), constraint_asia)
        output_filename = filename.parent / (filename.stem + '.asia.nc')
        iris.save(asia_cmorph_ppt_cube, str(output_filename), zlib=True)


def load_raw_0p25deg_3hrly_year(data_dir, year, month, day):
    if isinstance(year, int):
        year = f'{year:04}'
    if isinstance(month, int):
        month = f'{month:02}'
    if isinstance(day, int):
        day = f'{day:02}'

    filenames = sorted(Path(data_dir).glob(f'CMORPH_V1.0_ADJ_0.25deg-3HLY_{year}{month}{day}.bz2'))
    data = []
    for filename in filenames:
        print(filename)
        data.append(load_raw_0p25deg_3hrly(filename))
    return np.ma.masked_array(data)


def load_raw_0p25deg_3hrly(filename):
    with bz2.open(filename) as fp:
        buf = fp.read()
        raw_cmorph_data = np.frombuffer(buf, '<f4')

    masked_cmorph_data = np.ma.masked_array(raw_cmorph_data, 
                                            raw_cmorph_data == -999).reshape(8, 480, 1440)

    return masked_cmorph_data
