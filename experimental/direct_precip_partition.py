import sys

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np

from remake import Task, TaskControl, remake_task_control
from cosmic import util
from cosmic.config import CONSTRAINT_ASIA, PATHS


def gen_orog_mask(inputs, outputs, dotprod_val_thresh, dist_thresh):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    orog_asia = orog.extract(CONSTRAINT_ASIA)
    grad_orog = util.calc_uniform_lat_lon_grad(orog)
    grad_orog_asia = grad_orog.extract(CONSTRAINT_ASIA)

    surf_wind_asia = iris.load(str(inputs['surf_wind']))
    u = surf_wind_asia.extract_strict('x_wind')
    v = surf_wind_asia.extract_strict('y_wind')

    dotprod = u.copy()
    dotprod.data = np.zeros_like(dotprod.data, dtype=float)

    for tindex in range(u.shape[0]):
        print(tindex)
        dotprod2d = util.calc_2d_dot_product(iris.cube.CubeList([u[tindex], v[tindex]]), grad_orog_asia)
        # Note, dotprod[tindex].data will not assign it!
        dotprod.data[tindex] = dotprod2d.data

    dotprod.rename('surf_wind x del orog')
    dotprod_thresh = dotprod.copy()
    # iris does not like bools.
    dotprod_thresh.data = (dotprod.data > dotprod_val_thresh).astype(np.single)
    dotprod_thresh.rename(f'surf_wind x del orog > {dotprod_val_thresh}')

    lat_asia = dotprod_thresh.coord('latitude').points
    lon_asia = dotprod_thresh.coord('longitude').points
    Lon_asia, Lat_asia = np.meshgrid(lon_asia, lat_asia)

    dist_asia = util.CalcLatLonDistanceMask(Lat_asia, Lon_asia, dist_thresh)
    mask_asia = dist_asia.calc_close_to_mask_3d(dotprod_thresh)
    mask_asia.data = mask_asia.data.astype(np.single)
    mask_asia.rename(f'expanded surf_wind x del orog > {dotprod_val_thresh}')
    iris.save(iris.cube.CubeList([dotprod, dotprod_thresh, mask_asia]), str(outputs[0]))


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    orog_input = PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.orog'
    year = 2006
    for month in [6, 7, 8]:
        surf_wind_input = (PATHS['datadir'] / 'u-al508' / 'ap9.pp' /
                           f'surface_wind_{year}{month:02}' /
                           f'al508a.p9{year}{month:02}.asia.nc')
        inputs = {'orog': orog_input, 'surf_wind': surf_wind_input}

        dotprod_thresh = 0.1
        dist_thresh = 100
        outputs = [PATHS['datadir'] / 'orog_precip' / 'experiments' /
                   f'u-al508_direct_orog_mask.dp_{dotprod_thresh}.dist_{dist_thresh}.{year}{month:02}.asia.nc']
        tc.add(Task(gen_orog_mask, inputs, outputs, func_args=(dotprod_thresh, dist_thresh)))
    return tc
