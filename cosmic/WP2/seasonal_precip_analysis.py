import os
import sys
import logging
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import iris

DEFAULT_DATADIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/u-ak543/ap9.pp/')
DEFAULT_PRECIP_THRESH = 0.1  # mm hr-1

DEFAULT_DIR_TPL = 'precip_{year}{month:02}'
DEFAULT_FILE_TPL = '{runid}{split_stream}{year}{month:02}.{loc}_precip.nc'
DEFAULT_OUTPUT_FILE_TPL = '{runid}{split_stream}{season}.{loc}_precip.ppt_thresh_{thresh_text}.nc'

logging.basicConfig(level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)

            
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
    logger.debug(f'number of months in season: {len(nc_season)}')
    return nc_season


def calc_precip_amount_freq_intensity(season, season_cube, precip_thresh, 
                                      num_per_day=24, convert_kgpm2ps1_to_mmphr=True,
                                      calc_method='reshape', ignore_mask=True):
    if not ignore_mask:
        raise NotImplementedError('Results not 100% reliable, use at own risk')

    factor = 3600 if convert_kgpm2ps1_to_mmphr else 1
    assert season_cube.shape[0] % num_per_day == 0, 'Cube has wrong time dimension'
    num_days = season_cube.shape[0] // num_per_day

    logger.debug('calc season_mean')
    season_mean = season_cube.collapsed('time', iris.analysis.MEAN)
    logger.debug('calc season_std')
    season_std = season_cube.collapsed('time', iris.analysis.STD_DEV)
    season_mean.rename('precip_flux_mean')
    season_std.rename('precip_flux_std')

    start = timer()

    if calc_method == 'reshape':
        # Reshape ndarray so that axis 1 is one day.
        # I.e. reshaped_data[0] is the 1 hourly data for one day (num_per_day of them) for all lat/lon.
        # Using -1 tells reshape to infer the dimension from the others.
        # Convert from kg m-2 s-1 to mm hr-1 by multiplying by 3600 (# s/hr)
        # This method causes all data to be loaded into memory.
        reshaped_data = season_cube.data.reshape(num_days, num_per_day, 
                                                 season_cube.shape[1], 
                                                 season_cube.shape[2]) * factor
        if ignore_mask:
            reshaped_data = reshaped_data.filled(0)

        # The freq, amount and intensity must all be collapsed on the first dimension.
        freq_mask = reshaped_data < precip_thresh
        season_freq_data = 1 - freq_mask.sum(axis=0) / num_days
        # N.B. this is a *thresholded* amount. It will be very similar to the mean, but not identical.
        # Keep units as mm hr-1, by dividing by number of hours.
        season_amount_data = (np.ma.masked_array(reshaped_data, mask=freq_mask).sum(axis=0) / 
                              num_days).filled(0)

        season_intensity_data = np.ma.masked_array(reshaped_data, mask=freq_mask).mean(axis=0).filled(0)

        max_diff = np.max(np.abs(season_intensity_data * season_freq_data - season_amount_data))
        logger.debug(f'max diff: {max_diff}')
    elif calc_method == 'low_mem':
        # Use a moving window over the array to calc freq and amount.
        # Will make use of freq * intensity = amount to calc intensity.
        data_shape = (num_per_day, season_cube.shape[1], season_cube.shape[2])
        season_freq_data = np.zeros(data_shape)
        season_amount_data = np.zeros(data_shape)
        if ignore_mask:
            data_mask = np.zeros(data_shape, dtype=bool)
        else:
            data_mask = np.ones(data_shape, dtype=bool)

        for i in range(num_days):
            if i % 10 == 0:
                logger.debug(f'calc for day {i}')
            # N.B. only load slice into memory because slices *cube*, not *cube.data*.
            sliced_data = season_cube[i * num_per_day: (i + 1) * num_per_day].data * factor
            if ignore_mask:
                sliced_data = sliced_data.filled(0)
            else:
                data_mask &= sliced_data.mask

            # freq_keep should not be masked. Do not want to add a masked var as it will keep the mask.
            freq_keep = (sliced_data >= precip_thresh).data
            season_freq_data += freq_keep
            season_amount_data[freq_keep] = season_amount_data[freq_keep] + sliced_data[freq_keep]

        if ignore_mask:
            season_amount_data = season_amount_data / num_days
            season_freq_data = season_freq_data / num_days
            season_intensity_data = (np.ma.masked_array(season_amount_data,
                                                        season_freq_data == 0) / season_freq_data).filled(0)
        else:
            season_amount_data = np.ma.masked_array(season_amount_data / num_days,
                                                    (season_freq_data == 0) | data_mask)
            season_intensity_data = np.ma.masked_array(season_amount_data / season_freq_data,
                                                       (season_freq_data == 0) | data_mask)
    logger.debug(f'performed {calc_method} in {timer() - start:.02f}s')

    hourly_coords = [(season_cube[:num_per_day].coord('time'), 0),
                     (season_cube.coord('latitude'), 1),
                     (season_cube.coord('longitude'), 2)]

    logger.debug('build freq cube')
    season_hourly_freq = iris.cube.Cube(season_freq_data,
                                        long_name=f'freq_of_precip_{season}',
                                        units='',
                                        dim_coords_and_dims=hourly_coords)

    logger.debug('build amount cube')
    season_hourly_amount = iris.cube.Cube(season_amount_data,
                                          long_name=f'amount_of_precip_{season}',
                                          units='mm hr-1',
                                          dim_coords_and_dims=hourly_coords)

    logger.debug('build intensity cube')
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

    logger.debug('save analysis cubes')
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
