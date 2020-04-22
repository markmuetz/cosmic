import datetime as dt
from pathlib import Path
import bz2
import tarfile

import numpy as np
import iris


def convert_cmorph_0p25deg_3hrly_to_netcdf4_month(data_dir, output_dir, year, month):
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

    data = _load_raw_0p25deg_3hrly_year(data_dir, year, month, '??')

    coords = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    cmorph_ppt_cube = iris.cube.Cube(data.reshape(len(times), 480, 1440),
                                     long_name='precipitation', units='mm hr-1',
                                     dim_coords_and_dims=coords)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iris.save(cmorph_ppt_cube, output_dir / f'cmorph_ppt_{year}{month:02}.nc', zlib=True)


def convert_cmorph_8km_30min_to_netcdf4_month(data_dir, output_dir, year, month):
    lon0 = 0.036378335
    dlon = 0.072756669
    nlon = 4948
    lat0 = -59.963614
    dlat = 0.072771377
    nlat = 1649
    lat = np.linspace(lat0, lat0 + dlat * (nlat - 1), nlat)
    lon = np.linspace(lon0, lon0 + dlon * (nlon - 1), nlon)

    epoch = dt.datetime(1970, 1, 1)
    start_time = dt.datetime(year, month, 1, 0, 15)
    if month < 12:
        month_end_time = dt.datetime(year, month + 1, 1, 0, 30)
    else:
        month_end_time = dt.datetime(year + 1, 1, 1, 0, 30)
    curr_time = start_time

    lat_coord = iris.coords.Coord(lat, standard_name='latitude', units='degrees')
    lon_coord = iris.coords.Coord(lon, standard_name='longitude', units='degrees')
    end_time = start_time + dt.timedelta(days=1)

    day = 1
    while end_time < month_end_time:
        times = []
        while curr_time < end_time:
            times.append((curr_time - epoch).total_seconds() / 3600)
            curr_time += dt.timedelta(minutes=30)
        assert curr_time == end_time

        time_coord = iris.coords.Coord(times, standard_name='time',
                                       units=('hours since 1970-01-01 00:00:00'))

        data = _load_raw_8km_3min_year(data_dir, year, month, day)

        coords = [(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
        cmorph_ppt_cube = iris.cube.Cube(data.reshape(len(times), 1649, 4948),
                                         long_name='precipitation', units='mm hr-1',
                                         dim_coords_and_dims=coords)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        iris.save(cmorph_ppt_cube, output_dir / f'cmorph_ppt_{year}{month:02}{day:02}.nc', zlib=True)
        end_time += dt.timedelta(days=1)
        day += 1


def extract_asia(data_dir, year):
    constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
                       & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))
    # N.B. run in dir for one month: only loads data for one month.
    filenames = sorted(Path(data_dir).glob(f'cmorph_ppt_{year}??.nc'))

    for filename in filenames:
        asia_cmorph_ppt_cube = iris.load_cube(str(filename), constraint_asia)
        output_filename = filename.parent / (filename.stem + '.asia.nc')
        iris.save(asia_cmorph_ppt_cube, str(output_filename), zlib=False)


def extract_asia_8km_30min(data_dir, year, month):
    constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
                       & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))
    # Different from above.
    # N.B. run in dir for one month: only loads data for one month.
    filenames = sorted(Path(data_dir).glob(f'cmorph_ppt_{year}????.nc'))

    asia_cmorph_ppt_cube = iris.load([str(f) for f in filenames], constraint_asia).concatenate_cube()
    output_filename = data_dir / (f'cmorph_ppt_{year}{month:02}.asia.nc')
    # Compression saves A LOT of space: 5.0G -> 67M.
    iris.save(asia_cmorph_ppt_cube, str(output_filename), zlib=True)


def extract_europe_8km_30min(data_dir, year, month):
    # N.B. Max. lat of CMORPH is 60 N.
    constraint_eu = (iris.Constraint(coord_values={'latitude': lambda cell: 28 < cell < 67}))
                     # Cannot constrain on longitude across a boundary!
                     # & iris.Constraint(coord_values={'longitude': lambda cell: -22 < cell < 37}))
    # Different from above.
    # N.B. run in dir for one month: only loads data for one month.
    filenames = sorted(Path(data_dir).glob(f'cmorph_ppt_{year}????.nc'))

    eu_cmorph_ppt_cube = iris.load([str(f) for f in filenames], constraint_eu).concatenate_cube()
    output_filename = data_dir / (f'cmorph_ppt_{year}{month:02}.europe.nc')
    iris.save(eu_cmorph_ppt_cube.intersection(longitude=(-22, 37)), str(output_filename), zlib=True)


def _load_raw_0p25deg_3hrly_year(data_dir, year, month):
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
        data.append(_load_raw_0p25deg_3hrly(filename))
    return np.ma.masked_array(data)


def _load_raw_8km_3min_year(data_dir, year, month, day):
    filename = Path(data_dir) / f'CMORPH_V1.0_ADJ_8km-30min_{year}{month:02}.tar'
    data = []
    with tarfile.open(filename) as tar:
        for member in [m for m in sorted(tar.getmembers(), key=lambda ti: ti.name) if m.isfile()]:
            # member.name == '199801/CMORPH_V1.0_ADJ_8km-30min_1998010408.bz2'
            if member.name[-8:-6] == f'{day:02}':
                print(member)
                data.append(_load_raw_8km_30min(tar.extractfile(member)))
    return np.ma.masked_array(data)


def _load_raw_0p25deg_3hrly(filename):
    with bz2.open(filename) as fp:
        buf = fp.read()
        raw_cmorph_data = np.frombuffer(buf, '<f4')

    # Data is in mm/3hr, convert to mm/hr.
    masked_cmorph_data = np.ma.masked_array(raw_cmorph_data,
                                            raw_cmorph_data == -999).reshape(8, 480, 1440) / 3

    return masked_cmorph_data


def _load_raw_8km_30min(fp):
    with bz2.open(fp) as bzfp:
        buf = bzfp.read()
        raw_cmorph_data = np.frombuffer(buf, '<f4')

    # Data is in mm/hr
    masked_cmorph_data = np.ma.masked_array(raw_cmorph_data,
                                            raw_cmorph_data == -999).reshape(2, 1649, 4948)

    return masked_cmorph_data
