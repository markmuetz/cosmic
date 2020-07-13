# coding: utf-8
import pandas as pd

from cosmic.config import PATHS

df = pd.read_hdf(str(PATHS['datadir'] / 'orog_precip' / 'experiments' / 'combine_fracs.asia.hdf'))
df.orog_frac *= 100
df.orog_precip_frac *= 100
df.land_frac *= 100

diag_df = pd.read_hdf(str(PATHS['datadir'] / 'orog_precip' / 'experiments' / 'diag_combine_fracs.asia.hdf'))
diag_df.orog_frac *= 100
diag_df.orog_precip_frac *= 100
diag_df.land_frac *= 100

# Hack units of these -- should be calculated properly in the first place.
# N.B. can work out the factor 28000 by dividing e.g. direct method ocean_total by diag method.
# They should be the same!
diag_df.ocean_total *= 28000
diag_df.land_total *= 28000
diag_df.orog_total *= 28000
diag_df.non_orog_total *= 28000

df_mean = df[df.dist_thresh == 100].groupby(['model'], sort=False).mean()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']]
df_mean['method'] = 'direct'
diag_df_mean = diag_df.groupby(['model'], sort=False).mean()[['orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']]
diag_df_mean['method'] = 'diag'


df_output = pd.concat([df_mean, diag_df_mean])[['method', 'orog_frac', 'orog_precip_frac', 'land_frac', 'ocean_total', 'land_total', 'orog_total', 'non_orog_total']]
print(df_output)
output_path = str(PATHS['figsdir'] / 'orog_precip' / 'experiments' / 'all_orog_fracs.csv')
print(output_path)
with open(output_path, 'w') as f:
    df_output.to_csv(f, float_format='%.1f')
