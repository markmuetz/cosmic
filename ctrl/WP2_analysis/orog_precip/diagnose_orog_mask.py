import sys
from itertools import product

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from remake import Task, TaskControl, remake_task_control
from cosmic import util
from cosmic.config import CONSTRAINT_ASIA, PATHS
from orog_precip_paths import (land_sea_mask, extended_rclim_mask, precip_path_tpl,
                               diag_orog_precip_path_tpl, diag_orog_precip_frac_path_tpl,
                               diag_combine_frac_path, fmtp)


def calc_orog_precip(inputs, outputs, index_month):
    extended_rclim_mask = iris.load_cube(str(inputs['extended_rclim_mask']), CONSTRAINT_ASIA)
    lsm_asia = iris.load_cube(str(inputs['land_sea_mask']), CONSTRAINT_ASIA)
    precip_asia = iris.load_cube(str(inputs['precip']))
    precip_asia_mean = precip_asia.collapsed('time', iris.analysis.MEAN)
    # Need to regrid to mask resolution.
    lsm_asia_coarse = util.regrid(lsm_asia, extended_rclim_mask)
    precip_asia_mean_coarse = util.regrid(precip_asia_mean, extended_rclim_mask)

    orog_precip_asia = precip_asia_mean_coarse.copy()
    orog_precip_asia.rename('orog_' + precip_asia_mean_coarse.name())
    nonorog_precip_asia = precip_asia_mean_coarse.copy()
    nonorog_precip_asia.rename('non_orog_' + precip_asia_mean_coarse.name())
    ocean_precip_asia = precip_asia_mean_coarse.copy()
    ocean_precip_asia.rename('ocean_' + precip_asia_mean_coarse.name())

    orog_precip_asia.data = (precip_asia_mean_coarse.data *
                             lsm_asia_coarse.data *
                             extended_rclim_mask[index_month].data)
    nonorog_precip_asia.data = (precip_asia_mean_coarse.data *
                                lsm_asia_coarse.data *
                                (1 - extended_rclim_mask[index_month].data))
    ocean_precip_asia.data = (precip_asia_mean_coarse.data *
                              (1 - lsm_asia_coarse.data))

    iris.save(iris.cube.CubeList([orog_precip_asia,
                                  nonorog_precip_asia,
                                  ocean_precip_asia]), str(outputs[0]))


def calc_orog_precip_fracs(inputs, outputs, index_month):
    # TODO: area weighting.
    orog_mask = iris.load_cube(str(inputs['extended_rclim_mask']))
    lsm = iris.load_cube(str(inputs['land_sea_mask']))
    orog_precip_cubes = iris.load(str(inputs['orog_precip']))

    lsm_coarse = util.regrid(lsm, orog_mask)

    orog_mask_asia = orog_mask.extract(CONSTRAINT_ASIA)
    lsm_coarse_asia = lsm_coarse.extract(CONSTRAINT_ASIA)

    orog_precip = orog_precip_cubes.extract_strict('orog_precipitation_flux')
    non_orog_precip = orog_precip_cubes.extract_strict('non_orog_precipitation_flux')
    land_precip = orog_precip + non_orog_precip
    ocean_precip = orog_precip_cubes.extract_strict('ocean_precipitation_flux')

    orog_frac = (orog_mask_asia[index_month].data * lsm_coarse_asia.data).sum() / lsm_coarse_asia.data.sum()
    non_orog_frac = ((1 - orog_mask_asia[index_month].data) * lsm_coarse_asia.data).sum() / lsm_coarse_asia.data.sum()

    land_precip_total = land_precip.data.sum()
    ocean_precip_total = ocean_precip.data.sum()
    orog_precip_total = orog_precip.data.sum()
    non_orog_precip_total = non_orog_precip.data.sum()

    land_precip_frac = land_precip_total / (ocean_precip_total + land_precip_total)
    orog_precip_frac = orog_precip_total / land_precip_total
    non_orog_precip_frac = non_orog_precip_total / land_precip_total

    df = pd.DataFrame({
        'orog_frac': [orog_frac],
        'non_orog_frac': [non_orog_frac],
        'land_total': [land_precip_total],
        'ocean_total': [ocean_precip_total],
        'land_frac': [land_precip_frac],
        'orog_total': [orog_precip_total],
        'non_orog_total': [non_orog_precip_total],
        'orog_precip_frac': [orog_precip_frac],
        'non_orog_precip_frac': [non_orog_precip_frac],
    })
    df.to_hdf(str(outputs[0]), 'orog_fracs')


def combine_orog_precip_fracs(inputs, outputs, variables, columns):
    dfs = []
    for input_path in inputs:
        df = pd.read_hdf(str(input_path))
        dfs.append(df)
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined['dataset'] = [str(p) for p in inputs]
    df_combined = pd.concat([df_combined, pd.DataFrame(variables, columns=columns)], axis=1)
    df_combined.to_hdf(str(outputs[0]), 'combined_orog_fracs')


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    # /gws/nopw/j04/cosmic/mmuetz/data/era_interim_orog_precip

    # years = [2006]
    years = [2005, 2006, 2007, 2008]
    models = ['al508', 'ak543']
    months = [6, 7, 8]

    for model, year, month in product(models, years, months):
        # al508a.p9200606.asia_precip.nc
        precip_path = fmtp(precip_path_tpl, model=model, year=year, month=month)
        orog_precip_inputs = {
            'extended_rclim_mask': extended_rclim_mask,
            'land_sea_mask': land_sea_mask,
            'precip': precip_path
        }
        diag_orog_precip_path = fmtp(diag_orog_precip_path_tpl, model=model, year=year, month=month)
        tc.add(Task(calc_orog_precip,
                    orog_precip_inputs,
                    [diag_orog_precip_path],
                    func_args=(month - 1, )))

        orog_precip_fracs_inputs = {
            'extended_rclim_mask': extended_rclim_mask,
            'land_sea_mask': land_sea_mask,
            'orog_precip': diag_orog_precip_path
        }
        diag_orog_precip_frac_path = fmtp(diag_orog_precip_frac_path_tpl, model=model, year=year, month=month)
        tc.add(Task(calc_orog_precip_fracs,
                    orog_precip_fracs_inputs,
                    [diag_orog_precip_frac_path],
                    func_args=(month - 1, )))

    variables = list(product(models, months))
    columns = ['model', 'month']
    combine_inputs = [fmtp(diag_orog_precip_frac_path_tpl, model=model, year=year, month=month)
                      for model, month in variables]
    combine_fracs_output = [diag_combine_frac_path]
    tc.add(Task(combine_orog_precip_fracs,
                combine_inputs,
                combine_fracs_output,
                func_args=(variables, columns)
               ))

    return tc
