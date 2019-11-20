import sys
from pathlib import Path

import iris

from cosmic.util import load_config, regrid


def main(year, month):
    target_filepath = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/u-al508/ap9.pp/precip_200501/al508a.p9200501.asia_precip.nc')

    cmorph_datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')
    cmorph_filepath = cmorph_datadir / f'8km-30min/precip_{year}{month:02}/cmorph_ppt_{year}{month:02}.asia.nc'

    output_filepath = cmorph_filepath.parent / (cmorph_filepath.stem + '.N1280.nc')
    print(f'Regrid {cmorph_filepath} -> {output_filepath}')
    print(f'  using {target_filepath} resolution')

    coarse_cube = regrid(cmorph_filepath, target_filepath)
    iris.save(coarse_cube, str(output_filepath), zlib=True)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(*config.SCRIPT_ARGS[config_key])
