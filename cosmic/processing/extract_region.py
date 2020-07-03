# coding: utf-8
import os
import sys
import logging
from pathlib import Path

import iris
from iris.experimental import equalise_cubes

from cosmic.config import CONSTRAINT_ASIA, CONSTRAINT_EU
from cosmic.util import load_module

logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def UM_gen_region_precip_filepath(runid, stream, year, month, region, output_dir):
    return (output_dir /
            f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}.{region}_precip.nc')


def UM_extract_region_precip(runid, stream, year, month, nc_dirpath, region='asia', stratiform=False):
    output_filepath = UM_gen_region_precip_filepath(runid, stream, year, month, region, nc_dirpath)
    done_filename = (output_filepath.parent / (output_filepath.name + '.done'))

    if done_filename.exists():
        logger.info(f'Skipping: {done_filename.name} exists')
        return
    nc_filename_glob = (nc_dirpath /
                        f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}??.precip.nc')

    # N.B. loading 10 files per month.
    if region == 'asia':
        region_precip_cubes = iris.load(str(nc_filename_glob), CONSTRAINT_ASIA)
    elif region == 'europe':
        region_precip_cubes = iris.load(str(nc_filename_glob), CONSTRAINT_EU)

    rainfall_flux_name = 'rainfall_flux'
    snowfall_flux_name = 'snowfall_flux'
    if stratiform:
        rainfall_flux_name = 'stratiform_' + rainfall_flux_name
        snowfall_flux_name = 'stratiform_' + snowfall_flux_name
    region_rainfall = (region_precip_cubes.extract(iris.Constraint(name=rainfall_flux_name))
                       .concatenate_cube())
    region_snowfall = (region_precip_cubes.extract(iris.Constraint(name=snowfall_flux_name))
                       .concatenate_cube())

    region_total_precip = region_rainfall + region_snowfall
    region_total_precip.rename('precipitation_flux')

    if region == 'europe':
        iris.save(region_total_precip.intersection(longitude=(-22, 37)), str(output_filepath))
    else:
        iris.save(region_total_precip, str(output_filepath))
    done_filename.touch()


def HadGEM_gen_asia_precip_filepath(model, expt, variant, year, season, output_dir):
    return (output_dir /
            f'{model}.{expt}.{variant}.{year}.{season}.asia_precip.nc')


def HadGEM3_extract_asia_precip(model, nc_dirpath, output_dir, year, season='JJA',
                                expt='highresSST-present', variant='r1i1p1f1'):
    output_filepath = HadGEM_gen_asia_precip_filepath(model, expt, variant, year, season, output_dir)
    done_filename = (output_filepath.parent / (output_filepath.name + '.done'))

    if done_filename.exists():
        logger.info(f'Skipping: {done_filename.name} exists')
        return
    # e.g.:
    # pr_E1hr_HadGEM3-GC31-HM_highresSST-present_r1i1p1f1_gn_201404010030-201406302330.nc
    # pr_E1hr_HadGEM3-GC31-MM_highresSST-present_r1i1p1f1_gn_201401010030-201412302330.nc
    # pr_E1hr_HadGEM3-GC31-LM_highresSST-present_r1i1p1f1_gn_201401010030-201412302330.nc
    nc_filename_glob = (nc_dirpath / f'pr_E1hr_{model}_{expt}_{variant}_gn_{year}????????-{year}????????.nc')

    if season == 'JJA':
        constraint_season = iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
    elif season == 'SON':
        constraint_season = iris.Constraint(time=lambda cell: 9 <= cell.point.month <= 11)
    elif season == 'DJF':
        constraint_season = iris.Constraint(time=lambda cell: ((cell.point.month == 12) or
                                                               (cell.point.month == 1) or
                                                               (cell.point.month == 2)))
    elif season == 'MAM':
        constraint_season = iris.Constraint(time=lambda cell: 3 <= cell.point.month <= 5)

    asia_precip_season_cubes = iris.load(str(nc_filename_glob), CONSTRAINT_ASIA & constraint_season)
    equalise_cubes.equalise_attributes(asia_precip_season_cubes)
    asia_precip_season_cube = asia_precip_season_cubes.concatenate_cube()

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    iris.save(asia_precip_season_cube, str(output_filepath))
    done_filename.touch()


def main(config, region, nc_dirpath):
    logger.info(nc_dirpath)
    year = int(nc_dirpath.stem[-6:-2])
    month = int(nc_dirpath.stem[-2:])
    UM_extract_region_precip(config.RUNID, config.STREAM, year, month, nc_dirpath, region, config.STRATIFORM)


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config, *config.SCRIPT_ARGS[config_key])
