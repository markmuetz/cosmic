import cosmic.headless_matplotlib  # uses 'agg' backend if HEADLESS env var set.

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (diag_orog_precip_path_tpl, diag_orog_precip_fig_tpl, fmtp)


@remake_required(depends_on=[configure_ax_asia])
def plot_mean_orog_precip(inputs, outputs):
    orog_precip_cubes = iris.load([str(p) for p in inputs]).concatenate()
    orog_precip = orog_precip_cubes.extract_strict('orog_precipitation_flux')
    nonorog_precip = orog_precip_cubes.extract_strict('non_orog_precipitation_flux')
    ocean_precip = orog_precip_cubes.extract_strict('ocean_precipitation_flux')

    assert orog_precip.units == 'kg m-2 s-1'
    assert nonorog_precip.units == 'kg m-2 s-1'
    assert ocean_precip.units == 'kg m-2 s-1'

    extent = util.get_extent_from_cube(orog_precip)
    orog_precip_mean = orog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    nonorog_precip_mean = nonorog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    ocean_precip_mean = ocean_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1

    for i, (name, precip) in enumerate([('orog', orog_precip_mean),
                                        ('non orog', nonorog_precip_mean),
                                        ('ocean/water', ocean_precip_mean)]):
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
    im = ax.imshow(100 * orog_precip_mean / (orog_precip_mean + nonorog_precip_mean),
                   origin='lower', extent=extent, vmin=0, vmax=100)
    plt.colorbar(im, orientation='horizontal', label='% orog', pad=0.1)
    plt.savefig(outputs[3])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    year = 2006
    models = ['al508', 'ak543']

    for model in models:
        diag_orog_precip_paths = [fmtp(diag_orog_precip_path_tpl, model=model, year=year, month=month)
                                  for month in [6, 7, 8]]

        diag_orog_precip_figs = [fmtp(diag_orog_precip_fig_tpl, model=model, year=year, season='jja',
                                      precip_type=precip_type)
                                 for precip_type in ['orog', 'non_orog', 'ocean', 'orog_frac']]
        tc.add(Task(plot_mean_orog_precip, diag_orog_precip_paths, diag_orog_precip_figs))

    return tc

