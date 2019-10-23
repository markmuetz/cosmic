import sys
import logging
from pathlib import Path
from timeit import default_timer as timer

import iris

from cosmic.util import load_config


MONTH_MAP = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}

logging.basicConfig(level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def gen_nc_filepath(diagtype, filepath, remap_month=False):
    start, stream, extension = filepath.name.split('.')
    if remap_month:
        new_stream = stream[:-3] + f'{MONTH_MAP[stream[-3:]]:02}'
    else:
        new_stream = stream
    new_filename = '.'.join([start, new_stream, diagtype, 'nc'])
    new_filepath = filepath.parent / new_filename

    return new_filepath


def convert_pp_to_nc(pp_filepath, nc_filepath, attrs={}):
    done_filename = (nc_filepath.parent / (nc_filepath.name + '.done'))

    if done_filename.exists():
        logger.info(f'Skipping: {done_filename.name} exists')
        return

    start = timer()
    logger.info(f'Convert: {pp_filepath} -> {nc_filepath.name}')

    pp = iris.load(str(pp_filepath))
    middle = timer()
    logger.info(f'  loaded in:    {middle - start:.02f}s')
    if attrs:
        for cube in pp:
            cube.attributes.update(attrs)

    iris.save(pp, str(nc_filepath))
    end = timer()
    logger.info(f'  converted in: {end - start:.02f}s')

    done_filename.touch()


def main(config, pp_filepath):
    logger.info(pp_filepath)
    nc_filepath = gen_nc_filepath(config.DIAGTYPE, pp_filepath)
    convert_pp_to_nc(pp_filepath, nc_filepath, config.IRIS_CUBE_ATTRS)
    if config.DELETE_PP:
        logger.info(f'Deleting {pp_filepath}')
        pp_filepath.unlink()


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config, config.SCRIPT_ARGS[config_key])

