import sys
from pathlib import Path

import iris

from cosmic.util import load_config
import cosmic.WP2.seasonal_precip_analysis as spa


def fmt_year_month(year, month):
    return f'{year}{month:02}'


def main(start_year_month, end_year_month, 
         split_stream, loc, runid, precip_thresh, season):
    suite = f'u-{runid}'
    datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{suite}/ap9.pp/')
    nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                            runid=runid, split_stream=split_stream, loc=loc)
    season_cube = iris.load([str(p) for p in nc_season]).concatenate_cube()
    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh)

    daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
    output_file_tpl = '{runid}{split_stream}{season}.{daterange}.{loc}_precip.ppt_thresh_{thresh_text}.nc'

    spa.save_analysis_cubes(datadir, season, precip_thresh, analysis_cubes, 
                            output_file_tpl=output_file_tpl,
                            runid=runid, split_stream=split_stream, loc=loc, daterange=daterange)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config.START_YEAR_MONTH, config.END_YEAR_MONTH, 
         config.SPLIT_STREAM, config.LOC,
         *config.SCRIPT_ARGS[config_key])
