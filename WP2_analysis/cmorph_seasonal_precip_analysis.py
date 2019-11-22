import sys
from pathlib import Path

import iris

from cosmic.util import load_config
import cosmic.WP2.seasonal_precip_analysis as spa


def fmt_year_month(year, month):
    return f'{year}{month:02}'


def main(cmorph_dataset, resolution, start_year_month, end_year_month, season, precip_thresh):
    datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data/{cmorph_dataset}')
    if resolution:
        file_tpl = 'cmorph_ppt_{year}{month:02}.asia.{resolution}.nc'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,  
                                                file_tpl=file_tpl, resolution=resolution)
    else:
        file_tpl = 'cmorph_ppt_{year}{month:02}.asia.nc'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,  
                                                file_tpl=file_tpl)

    season_cube = iris.load([str(p) for p in nc_season]).concatenate_cube()

    if cmorph_dataset == '8km-30min':
        num_per_day = 48
    elif cmorph_dataset == '0.25deg-3HLY':
        num_per_day = 8

    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                           num_per_day=num_per_day,
                                                           convert_kgpm2ps1_to_mmphr=False,
                                                           calc_method='low_mem')

    daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
    if resolution:
        output_file_tpl = 'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_{thresh_text}.{resolution}.nc'
        spa.save_analysis_cubes(datadir, season, precip_thresh, analysis_cubes,
                                output_file_tpl=output_file_tpl, daterange=daterange, resolution=resolution)
    else:
        output_file_tpl = 'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_{thresh_text}.nc'
        spa.save_analysis_cubes(datadir, season, precip_thresh, analysis_cubes,
                                output_file_tpl=output_file_tpl, daterange=daterange)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config.CMORPH_DATASET, config.RESOLUTION, config.START_YEAR_MONTH, config.END_YEAR_MONTH, 
         *config.SCRIPT_ARGS[config_key])
