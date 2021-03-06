from itertools import product

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (orog_precip_path_tpl, orog_precip_fig_tpl, orog_precip_mean_fields_tpl, fmtp)


def _extract_precip_fields(orog_precip_cubes):
    orog_precip = orog_precip_cubes.extract_strict('orog_precipitation_flux')
    nonorog_precip = orog_precip_cubes.extract_strict('non_orog_precipitation_flux')
    ocean_precip = orog_precip_cubes.extract_strict('ocean_precipitation_flux')

    assert orog_precip.units == 'kg m-2 s-1'
    assert nonorog_precip.units == 'kg m-2 s-1'
    assert ocean_precip.units == 'kg m-2 s-1'

    orog_precip_mean = orog_precip[0].copy()
    nonorog_precip_mean = nonorog_precip[0].copy()
    ocean_precip_mean = ocean_precip[0].copy()

    orog_precip_mean.units = 'mm day-1'
    orog_precip_mean.rename(orog_precip_mean.name() + '_mean')
    nonorog_precip_mean.units = 'mm day-1'
    nonorog_precip_mean.rename(nonorog_precip_mean.name() + '_mean')
    ocean_precip_mean.units = 'mm day-1'
    ocean_precip_mean.rename(ocean_precip_mean.name() + '_mean')

    orog_precip_mean.data = orog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    nonorog_precip_mean.data = nonorog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    ocean_precip_mean.data = ocean_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    return nonorog_precip_mean, ocean_precip_mean, orog_precip_mean


@remake_required(depends_on=[_extract_precip_fields])
def extract_precip_mean_fields(inputs, outputs):
    orog_precip_cubes = iris.load([str(p) for p in inputs]).concatenate()
    nonorog_precip_mean, ocean_precip_mean, orog_precip_mean = _extract_precip_fields(orog_precip_cubes)
    iris.save(iris.cube.CubeList([nonorog_precip_mean, ocean_precip_mean, orog_precip_mean]), str(outputs[0]))


@remake_required(depends_on=[configure_ax_asia, _extract_precip_fields])
def plot_mean_orog_precip(inputs, outputs):
    cubes = iris.load(str(inputs[0]))
    orog_precip_mean = cubes.extract_strict('orog_precipitation_flux_mean')
    nonorog_precip_mean = cubes.extract_strict('non_orog_precipitation_flux_mean')
    ocean_precip_mean = cubes.extract_strict('ocean_precipitation_flux_mean')

    extent = util.get_extent_from_cube(orog_precip_mean)

    for i, (name, precip) in enumerate([('orog', orog_precip_mean.data),
                                        ('non orog', nonorog_precip_mean.data),
                                        ('ocean/water', ocean_precip_mean.data)]):
        plt.figure(figsize=(10, 7.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        configure_ax_asia(ax)
        im = ax.imshow(precip, origin='lower', extent=extent, norm=mpl.colors.LogNorm(),
                       vmin=1e-2, vmax=1e2)
        plt.colorbar(im, orientation='horizontal', label=f'{name} (mm day$^{{-1}}$)', pad=0.1)
        plt.savefig(outputs[i])

    plt.figure(figsize=(10, 7.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    configure_ax_asia(ax)
    im = ax.imshow(100 * orog_precip_mean.data / (orog_precip_mean.data + nonorog_precip_mean.data),
                   origin='lower', extent=extent, vmin=0, vmax=100)
    plt.colorbar(im, orientation='horizontal', label='% orog', pad=0.1)
    plt.savefig(outputs[3])


@remake_required(depends_on=[configure_ax_asia])
def plot_compare_mean_orog_precip(inputs, outputs, models, months):
    inputs1 = [path for (key, path) in inputs.items() if key[0] == models[0]]
    inputs2 = [path for (key, path) in inputs.items() if key[0] == models[1]]

    orog_precip_cubes1 = iris.load([str(p) for p in inputs1]).concatenate()
    orog_precip_cubes2 = iris.load([str(p) for p in inputs2]).concatenate()
    extent1, nonorog_precip_mean1, ocean_precip_mean1, orog_precip_mean1 = _extract_precip_fields(orog_precip_cubes1)
    extent2, nonorog_precip_mean2, ocean_precip_mean2, orog_precip_mean2 = _extract_precip_fields(orog_precip_cubes2)
    assert extent1 == extent2
    extent = extent1

    for i, (name, precip1, precip2) in enumerate([('orog', orog_precip_mean1, orog_precip_mean2),
                                                  ('non orog', nonorog_precip_mean1, nonorog_precip_mean2),
                                                  ('ocean/water', ocean_precip_mean1, ocean_precip_mean2)]):
        plt.figure(figsize=(10, 7.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        configure_ax_asia(ax)
        im = ax.imshow(precip2 / precip1 * 100, origin='lower', extent=extent,
                       vmin=0, vmax=200, cmap='bwr_r')
        plt.colorbar(im, orientation='horizontal', label=f'% change {models[1]}/{models[0]} (%)', pad=0.1)
        plt.savefig(outputs[i])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    models = ['al508', 'ak543']
    # models = ['ak543']
    dist_threshs = [100]
    dotprod_threshs = [0.05]
    months = [6, 7, 8]
    # dist_threshs = [20, 100]
    # dotprod_threshs = [0.05, 0.1]
    year = 2006

    for model, dotprod_thresh, dist_thresh in product(models, dotprod_threshs, dist_threshs):
        # 1 model at a time.
        orog_precip_paths = [fmtp(orog_precip_path_tpl, model=model, year=year, month=month,
                                  dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
                             for month in months]

        orog_precip_mean_fields = [fmtp(orog_precip_mean_fields_tpl, model=model, year=year, season='jja',
                                        dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)]
        tc.add(Task(extract_precip_mean_fields, orog_precip_paths, orog_precip_mean_fields))

    for model, dotprod_thresh, dist_thresh in product(models, dotprod_threshs, dist_threshs):
        orog_precip_mean_fields = [fmtp(orog_precip_mean_fields_tpl, model=model, year=year, season='jja',
                                        dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)]
        orog_precip_figs = [fmtp(orog_precip_fig_tpl, model=model, year=year, season='jja',
                                 dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh,
                                 precip_type=precip_type)
                            for precip_type in ['orog', 'non_orog', 'ocean', 'orog_frac']]
        tc.add(Task(plot_mean_orog_precip, orog_precip_mean_fields, orog_precip_figs))

    return tc
    for dotprod_thresh, dist_thresh in product(dotprod_threshs, dist_threshs):
        # Compare 2 models.
        orog_precip_paths = {(model, month): fmtp(orog_precip_path_tpl, model=model, year=year, month=month,
                                                  dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
                             for model in models
                             for month in months}
        orog_precip_figs = [fmtp(orog_precip_fig_tpl, model='-'.join(models), year=year, season='jja',
                                 dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh,
                                 precip_type=precip_type)
                            for precip_type in ['orog', 'non_orog', 'ocean']]
        tc.add(Task(plot_compare_mean_orog_precip, orog_precip_paths, orog_precip_figs,
                    func_args=(models, months)))

    return tc

