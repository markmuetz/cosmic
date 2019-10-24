# coding: utf-8
import sys
from pathlib import Path

import numpy as np
import iris

DEFAULT_DATADIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/u-ak543/ap9.pp/')
DEFAULT_PRECIP_THRESH = 0.1  # mm hr-1

DEFAULT_DIR_TPL = 'precip_{year}{month:02}'
DEFAULT_FILE_TPL = '{runid}{split_stream}{year}{month:02}.{loc}_precip.nc'
DEFAULT_OUTPUT_FILE_TPL = '{runid}{split_stream}{season}.{loc}_precip.ppt_thresh_{thresh_text}.nc'

            
def gen_nc_precip_filenames(datadir, season, start_year, end_year, 
                            dir_tpl=DEFAULT_DIR_TPL, 
                            file_tpl=DEFAULT_FILE_TPL, 
                            skip_spinup=True, 
                            **file_kwargs):
    nc_season = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            if skip_spinup and year == start_year and month == 1:
                continue  # skip spinup
            nc_asia_precip = (datadir / 
                              dir_tpl.format(year=year, month=month) / 
                              file_tpl.format(year=year, month=month, **file_kwargs))
            if not nc_asia_precip.exists():
                continue
            if month in range(3, 6) and season == 'mam':
                nc_season.append(nc_asia_precip)
            elif month in range(6, 9) and season == 'jja':
                nc_season.append(nc_asia_precip)
            elif month in range(9, 12) and season == 'son':
                nc_season.append(nc_asia_precip)
            elif month in [12, 1, 2] and season == 'djf':
                nc_season.append(nc_asia_precip)
    print(f'number of months in season: {len(nc_season)}')
    return nc_season


def calc_precip_amount_freq_intensity_low_mem(season, season_cube, precip_thresh, 
                                              num_per_day=24, convert_kgpm2ps1_to_mmphr=True):
    print('calc season_mean')
    season_mean = season_cube.collapsed('time', iris.analysis.MEAN)
    print('calc season_std')
    season_std = season_cube.collapsed('time', iris.analysis.STD_DEV)
    season_mean.rename('precip_flux_mean')
    season_std.rename('precip_flux_std')

    factor = 3600 if convert_kgpm2ps1_to_mmphr else 1

    assert season_cube.shape[0] % num_per_day == 0, 'Cube has wrong time dimension'
    num_days = season_cube.shape[0] // num_per_day

    season_freq_data = np.zeros((num_per_day, season_cube.shape[1], season_cube.shape[2]))
    season_amount_data = np.zeros((num_per_day, season_cube.shape[1], season_cube.shape[2]))
    # season_intensity_data = np.zeros((num_per_day, season_cube.shape[1], season_cube.shape[2]))

    for i in range(num_days):
        # N.B. only load slice into memory.
        cube_slice = season_cube[i * num_per_day: (i + 1) * num_per_day].data * factor

        freq_keep = cube_slice >= precip_thresh
        season_freq_data += freq_keep
        season_amount_data[freq_keep] = season_amount_data[freq_keep] + cube_slice[freq_keep]
        # season_intensity_data = np.ma.masked_array(reshaped_data, mask=freq_mask).mean(axis=0)
    season_freq_data /= num_days
    season_amount_data /= num_days
    season_intensity_data = season_amount_data / season_freq_data

    # max_diff = np.max(np.abs(season_intensity_data * season_freq_data - season_amount_data))
    # print(f'max diff: {max_diff}')

    hourly_coords = [(season_cube[:num_per_day].coord('time'), 0),
                     (season_cube.coord('latitude'), 1),
                     (season_cube.coord('longitude'), 2)]

    print('build freq cube')
    season_hourly_freq = iris.cube.Cube(season_freq_data,
                                        long_name=f'freq_of_precip_{season}',
                                        units='',
                                        dim_coords_and_dims=hourly_coords)

    print('build amount cube')
    season_hourly_amount = iris.cube.Cube(season_amount_data,
                                          long_name=f'amount_of_precip_{season}',
                                          units='mm hr-1',
                                          dim_coords_and_dims=hourly_coords)

    print('build intensity cube')
    season_hourly_intensity = iris.cube.Cube(season_intensity_data,
                                             long_name=f'intensity_of_precip_{season}',
                                             units='mm hr-1',
                                             dim_coords_and_dims=hourly_coords)

    analysis_cubes = iris.cube.CubeList([season_mean, season_std,
                                         season_hourly_freq, season_hourly_amount, 
                                         season_hourly_intensity])
    return analysis_cubes



def calc_precip_amount_freq_intensity(season, season_cube, precip_thresh, 
                                      num_per_day=24, convert_kgpm2ps1_to_mmphr=True):
    print('calc season_mean')
    season_mean = season_cube.collapsed('time', iris.analysis.MEAN)
    print('calc season_std')
    season_std = season_cube.collapsed('time', iris.analysis.STD_DEV)
    season_mean.rename('precip_flux_mean')
    season_std.rename('precip_flux_std')

    # Reshape ndarray so that axis 1 is one day.
    # I.e. reshaped_data[0] is the 1 hourly data for one day (num_per_day of them) for all lat/lon.
    # Using -1 tells reshape to infer the dimension from the others.
    # Convert from kg m-2 s-1 to mm hr-1 by multiplying by 3600 (# s/hr)
    # N.B. I am not sure how this will deal with larger datasets as dereferencing data
    # causes all data to be loaded into memory.
    print('reshape data')
    factor = 3600 if convert_kgpm2ps1_to_mmphr else 1
    reshaped_data = season_cube.data.reshape(-1, num_per_day, 
                                             season_cube.shape[1], 
                                             season_cube.shape[2]) * factor

    # The freq, amount and intensity must all be collapsed on the first dimension.
    print('calc freq')
    freq_mask = reshaped_data < precip_thresh
    season_freq_data = 1 - freq_mask.sum(axis=0) / freq_mask.shape[0]
    # N.B. this is a *thresholded* amount. It will be very similar to the mean, but not identical.
    print('calc amount')
    # Keep units as mm hr-1, by dividing by number of hours.
    season_amount_data = (np.ma.masked_array(reshaped_data, mask=freq_mask).sum(axis=0) / 
                          reshaped_data.shape[0])

    print('calc intensity')
    season_intensity_data = np.ma.masked_array(reshaped_data, mask=freq_mask).mean(axis=0)
    max_diff = np.max(np.abs(season_intensity_data * season_freq_data - season_amount_data))
    print(f'max diff: {max_diff}')

    hourly_coords = [(season_cube[:num_per_day].coord('time'), 0),
                     (season_cube.coord('latitude'), 1),
                     (season_cube.coord('longitude'), 2)]

    print('build freq cube')
    season_hourly_freq = iris.cube.Cube(season_freq_data,
                                        long_name=f'freq_of_precip_{season}',
                                        units='',
                                        dim_coords_and_dims=hourly_coords)

    print('build amount cube')
    season_hourly_amount = iris.cube.Cube(season_amount_data,
                                          long_name=f'amount_of_precip_{season}',
                                          units='mm hr-1',
                                          dim_coords_and_dims=hourly_coords)

    print('build intensity cube')
    season_hourly_intensity = iris.cube.Cube(season_intensity_data,
                                             long_name=f'intensity_of_precip_{season}',
                                             units='mm hr-1',
                                             dim_coords_and_dims=hourly_coords)

    analysis_cubes = iris.cube.CubeList([season_mean, season_std,
                                         season_hourly_freq, season_hourly_amount, 
                                         season_hourly_intensity])
    return analysis_cubes

def save_analysis_cubes(datadir, season, precip_thresh, analysis_cubes,
                        output_file_tpl=DEFAULT_OUTPUT_FILE_TPL, 
                        **output_file_kwargs):

    print('save analysis cubes')
    thresh_text = str(precip_thresh).replace('.', 'p')
    output_filepath = datadir / output_file_tpl.format(season=season, 
                                                       thresh_text=thresh_text, 
                                                       **output_file_kwargs)
    iris.save(analysis_cubes, str(output_filepath))
    return season_cube, analysis_cubes


def default_main(season):
    runid = 'ak543'
    split_stream = 'a.p9'
    loc = 'asia'

    nc_season = gen_nc_precip_filenames(DEFAULT_DATADIR, season, 2005, 2010,
                                        runid=runid, split_stream=split_stream, loc=loc)
    season_cube = iris.load([str(p) for p in nc_season]).concatenate_cube()
    analysis_cubes = calc_precip_amount_freq_intensity(DEFAULT_DATADIR, season, season_cube, 
                                                       DEFAULT_PRECIP_THRESH)
    save_analysis_cubes(DEFAULT_DATADIR, season, DEFAULT_PRECIP_THRESH,
                        runid=runid, 
                        split_stream=split_stream,
                        loc=loc)


if __name__ == '__main__':
    season = sys.argv[1]
