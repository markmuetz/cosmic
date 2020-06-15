import sys

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np

from remake import Task, TaskControl, remake_task_control
from cosmic import util
from cosmic.config import CONSTRAINT_ASIA, PATHS


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
    ocean_precip_asia.rename('oceang_' + precip_asia_mean_coarse.name())

    orog_precip_asia.data = (precip_asia_mean_coarse.data *
                             lsm_asia_coarse.data *
                             extended_rclim_mask[index_month].data)
    nonorog_precip_asia.data = (precip_asia_mean_coarse.data *
                                lsm_asia_coarse.data *
                                (1 - extended_rclim_mask[index_month].data))
    ocean_precip_asia.data = (precip_asia_mean_coarse.data *
                              (1 - lsm_asia_coarse.data) *
                              extended_rclim_mask[index_month].data)

    iris.save(iris.cube.CubeList([orog_precip_asia,
                                  nonorog_precip_asia,
                                  ocean_precip_asia]), str(outputs[0]))


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    land_sea_mask = PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.landfrac'
    extended_rclim_mask = PATHS['datadir'] / 'experimental' / 'extended_orog_mask.nc'
    # /gws/nopw/j04/cosmic/mmuetz/data/era_interim_orog_precip

    year = 2006
    for month in [6, 7, 8]:
        # al508a.p9200606.asia_precip.nc
        precip_path = (PATHS['datadir'] / 'u-al508' / 'ap9.pp' /
                       f'precip_{year}{month:02}' /
                       f'al508a.p9{year}{month:02}.asia_precip.nc')
        orog_precip_inputs = {
            'extended_rclim_mask': extended_rclim_mask,
            'land_sea_mask': land_sea_mask,
            'precip': precip_path
        }
        orog_precip_path = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                            f'u-al508_diagnose_orog.mean.{year}{month:02}.asia.nc')
        tc.add(Task(calc_orog_precip,
                    orog_precip_inputs,
                    [orog_precip_path],
                    func_args=(month - 1, )))

    return tc
