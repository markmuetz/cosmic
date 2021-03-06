import sys

from cosmic.util import load_module
from cosmic.datasets.cmorph.cmorph_convert import extract_europe_8km_30min


def main(basedir, year, month):
    print(f'{year}, {month}')
    output_dir = basedir / f'precip_{year}{month:02}'
    extract_europe_8km_30min(output_dir, year, month)


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config.BASEDIR, *config.SCRIPT_ARGS[config_key])
