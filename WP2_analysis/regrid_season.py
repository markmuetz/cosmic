import sys
from pathlib import Path

from cosmic.util import load_config
from cosmic.WP2.regrid_model_to_cmorph import regrid


def main(runid, loc, season):
    suite = f'u-{runid}'
    datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{suite}/ap9.pp/')
    filepath = datadir / f'{runid}a.p9{season}.{loc}_precip.ppt_thresh_0p1.nc'
    cmorph_datadir = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')
    cmorph_filepath = cmorph_datadir / f'cmorph_ppt_{season}.{loc}_precip.ppt_thresh_0p1.nc'
    output_filepath = datadir / (filepath.stem + '.cmorph_res.nc')

    coarse_cube = regrid(filepath, cmorph_filepath)
    iris.save(coarse_cube, output_filepath)


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config.RUNID, config.LOC, config.SCRIPT_ARGS[config_key])
