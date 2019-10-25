# coding: utf-8
import os
import sys
import logging
from pathlib import Path

import iris

from cosmic.util import load_config

logging.basicConfig(stream=sys.stdout, level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def gen_asia_precip_filepath(runid, stream, year, month, output_dir):
    return (output_dir / 
            f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}.asia_precip.nc')


def extract_asia_precip(runid, stream, year, month, nc_dirpath, stratiform=False):
    output_filepath = gen_asia_precip_filepath(runid, stream, year, month, nc_dirpath)
    done_filename = (output_filepath.parent / (output_filepath.name + '.done'))

    if done_filename.exists():
        logger.info(f'Skipping: {done_filename.name} exists')
        return
    # lat/lon constraints come from Asia HydroBASINS level 1 bounds, 
    # rounded up or down as appropriate.
    constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1}) 
                       & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))

    nc_filename_glob = (nc_dirpath / 
                        f'{runid[2:]}{stream[0]}.{stream[1:]}{year}{month:02}??.precip.nc')

    # N.B. loading 10 files per month.
    asia_precip_cubes = iris.load(str(nc_filename_glob), constraint_asia)

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


def main(config, nc_dirpath):
    logger.info(nc_dirpath)
    year = int(nc_dirpath.stem[-6:-2])
    month = int(nc_dirpath.stem[-2:])
    extract_asia_precip(config.RUNID, config.STREAM, year, month, nc_dirpath, config.STRATIFORM)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config, config.SCRIPT_ARGS[config_key])
