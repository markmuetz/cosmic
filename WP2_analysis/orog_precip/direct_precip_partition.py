from itertools import product

import iris
import numpy as np

from remake import Task, TaskControl, remake_task_control
from cosmic import util
from cosmic.config import CONSTRAINT_ASIA
from orog_precip_paths import (orog_path, land_sea_mask, cache_key_tpl, surf_wind_path_tpl,
                               orog_mask_path_tpl, precip_path_tpl, orog_precip_path_tpl, fmtp)


def gen_dist_cache(inputs, outputs, dist_thresh):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    orog_asia = orog.extract(CONSTRAINT_ASIA)
    lat_asia = orog_asia.coord('latitude').points
    lon_asia = orog_asia.coord('longitude').points
    Lon_asia, Lat_asia = np.meshgrid(lon_asia, lat_asia)
    cache_mask = util.CalcLatLonDistanceMask.gen_cache_mask(Lat_asia, Lon_asia, dist_thresh)
    util.CalcLatLonDistanceMask.save_cache_mask(outputs[0], cache_mask)


def gen_orog_mask(inputs, outputs, dotprod_val_thresh, dist_thresh):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    orog_asia = orog.extract(CONSTRAINT_ASIA)
    grad_orog = util.calc_uniform_lat_lon_grad(orog)
    grad_orog_asia = grad_orog.extract(CONSTRAINT_ASIA)

    cache_key = inputs['cache_key']

    surf_wind_asia = iris.load(str(inputs['surf_wind']))
    u = surf_wind_asia.extract_strict('x_wind')
    v = surf_wind_asia.extract_strict('y_wind')

    dotprod = u.copy()
    dotprod.data = np.zeros_like(dotprod.data, dtype=float)

    for tindex in range(u.shape[0]):
        print(tindex)
        dotprod2d = util.calc_2d_dot_product(iris.cube.CubeList([u[tindex], v[tindex]]),
                                             grad_orog_asia)
        # Note, dotprod[tindex].data will not assign it!
        dotprod.data[tindex] = dotprod2d.data

    dotprod.rename('surf_wind x del orog')

    dotprod_thresh = dotprod.copy()
    # iris does not like bools.
    dotprod_thresh.data = (dotprod.data > dotprod_val_thresh).astype(np.single)
    dotprod_thresh.rename(f'surf_wind x del orog > thresh')
    dotprod_thresh.units = None
    dotprod_thresh.attributes['dotprod_val_thresh'] = dotprod_val_thresh

    lat_asia = orog_asia.coord('latitude').points
    lon_asia = orog_asia.coord('longitude').points
    Lon_asia, Lat_asia = np.meshgrid(lon_asia, lat_asia)

    dist_asia = util.CalcLatLonDistanceMask(Lat_asia, Lon_asia, dist_thresh,
                                            circular_lon=False, cache_key=cache_key)
    mask_asia = dist_asia.calc_close_to_mask_3d(dotprod_thresh)
    mask_asia.data = mask_asia.data.astype(np.single)
    mask_asia.rename(f'expanded surf_wind x del orog > thresh')
    mask_asia.units = None
    mask_asia.attributes['dotprod_val_thresh'] = dotprod_val_thresh
    mask_asia.attributes['dist_thresh'] = dist_thresh

    iris.save(iris.cube.CubeList([dotprod, dotprod_thresh, mask_asia]), str(outputs[0]))


def calc_orog_precip(inputs, outputs):
    lsm_asia = iris.load_cube(str(inputs['land_sea_mask']), CONSTRAINT_ASIA)
    mask_asia = iris.load_cube(str(inputs['orog_mask']), f'expanded surf_wind x del orog > thresh')
    precip_asia = iris.load_cube(str(inputs['precip']))
    assert mask_asia.shape == precip_asia.shape

    orog_precip_asia = precip_asia.copy()
    orog_precip_asia.rename('orog_' + precip_asia.name())
    nonorog_precip_asia = precip_asia.copy()
    nonorog_precip_asia.rename('non_orog_' + precip_asia.name())
    ocean_precip_asia = precip_asia.copy()
    ocean_precip_asia.rename('ocean_' + precip_asia.name())

    for i in range(mask_asia.shape[0]):
        orog_precip_asia.data[i] = precip_asia[i].data * lsm_asia.data * mask_asia[i].data
        nonorog_precip_asia.data[i] = precip_asia[i].data * lsm_asia.data * (1 - mask_asia[i].data)
        ocean_precip_asia.data[i] = precip_asia[i].data * (1 - lsm_asia.data)
    iris.save(iris.cube.CubeList([orog_precip_asia,
                                  nonorog_precip_asia,
                                  ocean_precip_asia]), str(outputs[0]))


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    models = ['al508', 'ak543']
    dist_threshs = [50]
    dotprod_threshs = [0.05]
    # dist_threshs = [20, 100]
    # dotprod_threshs = [0.05, 0.1]

    for dist_thresh in dist_threshs:
        cache_key = fmtp(cache_key_tpl, dist_thresh=dist_thresh)
        tc.add(Task(gen_dist_cache,
                    {'orog': orog_path},
                    [cache_key],
                    func_args=(dist_thresh, )))

    for model, dotprod_thresh, dist_thresh in product(models, dotprod_threshs, dist_threshs):
        cache_key = fmtp(cache_key_tpl, dist_thresh=dist_thresh)
        year = 2006
        for month in [6, 7, 8]:
            surf_wind_path = fmtp(surf_wind_path_tpl, model=model, year=year, month=month)
            orog_mask_path = fmtp(orog_mask_path_tpl, model=model, year=year, month=month,
                                  dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
            inputs = {'orog': orog_path, 'cache_key': cache_key, 'surf_wind': surf_wind_path}

            tc.add(Task(gen_orog_mask, inputs, [orog_mask_path],
                        func_args=(dotprod_thresh, dist_thresh)))

            precip_path = fmtp(precip_path_tpl, model=model, year=year, month=month)
            orog_precip_path = fmtp(orog_precip_path_tpl, model=model, year=year, month=month,
                                    dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
            orog_precip_inputs = {
                'orog_mask': orog_mask_path,
                'land_sea_mask': land_sea_mask,
                'precip': precip_path
            }
            tc.add(Task(calc_orog_precip, orog_precip_inputs, [orog_precip_path]))

    return tc
