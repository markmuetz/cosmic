# coding: utf-8
import os
import sys
import logging
from pathlib import Path

import iris
from iris.experimental import equalise_cubes

from cosmic.util import load_module

logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)

# lat/lon constraints come from Asia HydroBASINS level 1 bounds,
# rounded up or down as appropriate.
CONSTRAINT_ASIA = (iris.Constraint(coord_values={'latitude': lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude': lambda cell: 56.9 < cell < 151.1}))


def UM_gen_asia_precip_filepath(runid, stream, year, month, output_dir):
    return (output_dir / 
            f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}.asia_precip.nc')


def UM_extract_asia_precip(runid, stream, year, month, nc_dirpath, stratiform=False):
    output_filepath = UM_gen_asia_precip_filepath(runid, stream, year, month, nc_dirpath)
    done_filename = (output_filepath.parent / (output_filepath.name + '.done'))

    if done_filename.exists():
        logger.info(f'Skipping: {done_filename.name} exists')
        return
    nc_filename_glob = (nc_dirpath /
                        f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}??.precip.nc')

    # N.B. loading 10 files per month.
    asia_precip_cubes = iris.load(str(nc_filename_glob), CONSTRAINT_ASIA)

    rainfall_flux_name = 'rainfall_flux'
    snowfall_flux_name = 'snowfall_flux'
    if stratiform:
        rainfall_flux_name += 'stratiform_'
        snowfall_flux_name += 'stratiform_'
    asia_rainfall = (asia_precip_cubes.extract(iris.Constraint(name=rainfall_flux_name))
                     .concatenate_cube())
    asia_snowfall = (asia_precip_cubes.extract(iris.Constraint(name=snowfall_flux_name))
                     .concatenate_cube())

    asia_total_precip = asia_rainfall + asia_snowfall
    asia_total_precip.rename('precipitation_flux')

    iris.save(asia_total_precip, str(output_filepath))
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


def main(config, nc_dirpath):
    logger.info(nc_dirpath)
    year = int(nc_dirpath.stem[-6:-2])
    month = int(nc_dirpath.stem[-2:])
    UM_extract_asia_precip(config.RUNID, config.STREAM, year, month, nc_dirpath, config.STRATIFORM)


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config, config.SCRIPT_ARGS[config_key])
