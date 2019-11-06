import os
os.environ['COSMIC_LOGLEVEL'] = 'DEBUG'

import itertools
from pathlib import Path

import numpy as np
import iris

import cosmic.WP2.seasonal_precip_analysis as spa

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')


# @profile
def main():
    jja = list(itertools.chain(*[BASEDIR.glob(f'precip_????{m:02}/*.asia.nc') for m in [6, 7, 8]]))
    season_cube = iris.load([str(p) for p in jja]).concatenate_cube()
    cubes = spa.calc_precip_amount_freq_intensity('jja', season_cube, 0.1, num_per_day=8, convert_kgpm2ps1_to_mmphr=False, calc_method='reshape')
    print(cubes)


if __name__ == '__main__':
    main()
