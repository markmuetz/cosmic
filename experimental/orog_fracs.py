# coding: utf-8
import sys
from itertools import product

import pandas as pd

df = pd.read_hdf('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/orog_precip/experiments/combine_fracs.asia.hdf')
df.orog_frac *= 100
df.orog_precip_frac *= 100
df.land_frac *= 100

dfg = df.groupby(['model', 'dist_thresh', 'month'], sort=False).first()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']]
with open('orog_fracs.csv', 'w') as f:
    dfg.to_csv(f, float_format='%.1f')
dfg_mean = df.groupby(['model', 'dist_thresh'], sort=False).mean()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']]
with open('orog_fracs_mean.csv', 'w') as f:
    dfg_mean.to_csv(f, float_format='%.1f')

