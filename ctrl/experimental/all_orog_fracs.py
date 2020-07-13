# coding: utf-8
import pandas as pd

from cosmic.config import PATHS

df = pd.read_hdf(str(PATHS['datadir'] / 'orog_precip' / 'experiments' / 'combine_fracs.asia.hdf'))
diag_df = pd.read_hdf(str(PATHS['datadir'] / 'orog_precip' / 'experiments' / 'diag_combine_fracs.asia.hdf'))

print(df.groupby(['model', 'dist_thresh'], sort=False).mean()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']])
print(diag_df.groupby(['model'], sort=False).mean()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']])
