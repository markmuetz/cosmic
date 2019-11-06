import os
os.environ['COSMIC_LOGLEVEL'] = 'DEBUG'

import itertools
from pathlib import Path

import numpy as np
import iris

import cosmic.WP2.seasonal_precip_analysis as spa

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')


jja = list(itertools.chain(*[BASEDIR.glob(f'precip_2000{m:02}/*.asia.nc') for m in [6, 7, 8]]))
season_cube1 = iris.load([str(p) for p in jja]).concatenate_cube()

cubes1 = spa.calc_precip_amount_freq_intensity('jja', season_cube1, 0.1, num_per_day=8, convert_kgpm2ps1_to_mmphr=False, calc_method='reshape')

season_cube2 = iris.load([str(p) for p in jja]).concatenate_cube()
cubes2 = spa.calc_precip_amount_freq_intensity('jja', season_cube2, 0.1, num_per_day=8, convert_kgpm2ps1_to_mmphr=False, calc_method='low_mem')

for c1, c2 in zip(cubes1, cubes2):
    print(f'{c1.name()}:{c2.name()}')
    print(f'  {np.allclose(c1.data, c2.data)}')
    # print(f'  masks: {c1.data.mask.sum()}, {c2.data.mask.sum()}')

a1 = cubes1[3]
a2 = cubes2[3]
f1 = cubes1[2]
f2 = cubes2[2]
i1 = cubes1[4]
i2 = cubes2[4]

ppt_mean = cubes1[0]
print('Should be close to zero')
print(f'min diff: {np.min(ppt_mean.data - a1.data.sum(axis=0) / 8)}')
