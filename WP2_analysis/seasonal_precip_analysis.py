import sys
from pathlib import Path

import iris

from cosmic.util import load_config
import cosmic.WP2.seasonal_precip_analysis as spa


def main(runid, start_year_month, end_year_month, 
         split_stream, loc, precip_thresh, season):
    suite = f'u-{runid}'
    datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{suite}/ap9.pp/')
    nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                            runid=runid, split_stream=split_stream, loc=loc)
    season_cube = iris.load([str(p) for p in nc_season]).concatenate_cube()
    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh)
    spa.save_analysis_cubes(datadir, season, precip_thresh, analysis_cubes,
                            runid=runid, split_stream=split_stream, loc=loc)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config.RUNID, config.START_YEAR_MONTH, config.END_YEAR_MONTH, 
         config.SPLIT_STREAM, config.LOC, config.PRECIP_THRESH,
         config.SCRIPT_ARGS[config_key])
