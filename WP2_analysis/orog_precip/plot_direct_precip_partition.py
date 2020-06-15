import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cosmic import util
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (orog_precip_path_tpl, orog_precip_fig_tpl, fmtp)


def _configure_ax_asia(ax, extent=None, tight_layout=True):
    ax.coastlines(resolution='50m')

    xticks = range(60, 160, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${t}\\degree$ E' for t in xticks])
    ax.set_xticks(np.linspace(58, 150, 47), minor=True)

    yticks = range(20, 60, 20)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'${t}\\degree$ N' for t in yticks])
    ax.set_yticks(np.linspace(2, 56, 28), minor=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True, which='both')
    if extent is not None:
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
    else:
        ax.set_xlim((58, 150))
        ax.set_ylim((2, 56))
    if tight_layout:
        plt.tight_layout()


@remake_required(depends_on=[_configure_ax_asia])
def plot_mean_orog_precip(inputs, outputs):
    orog_precip_cubes = iris.load([str(p) for p in inputs]).concatenate()
    orog_precip = orog_precip_cubes.extract_strict('orog precipitation_flux')
    nonorog_precip = orog_precip_cubes.extract_strict('non orog precipitation_flux')
    ocean_precip = orog_precip_cubes.extract_strict('ocean precipitation_flux')

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
        _configure_ax_asia(ax)
        im = ax.imshow(precip, origin='lower', extent=extent, norm=mpl.colors.LogNorm())
        plt.colorbar(im, orientation='horizontal', label=f'{name} (mm day$^{{-1}}$)', pad=0.1)
        plt.savefig(outputs[i])

    plt.figure(figsize=(10, 7.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _configure_ax_asia(ax)
    im = ax.imshow(100 * orog_precip_mean / (orog_precip_mean + nonorog_precip_mean),
                   origin='lower', extent=extent, vmin=0, vmax=100)
    plt.colorbar(im, orientation='horizontal', label='% orog', pad=0.1)
    plt.savefig(outputs[3])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    dotprod_thresh = 0.1
    dist_thresh = 100

    year = 2006
    orog_precip_paths = [fmtp(orog_precip_path_tpl, year=year, month=month,
                              dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
                         for month in [6, 7, 8]]

    orog_precip_figs = [fmtp(orog_precip_fig_tpl, year=year, season='jja',
                             dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh,
                             precip_type=precip_type)
                        for precip_type in ['orog', 'non_orog', 'ocean', 'orog_frac']]
    tc.add(Task(plot_mean_orog_precip, orog_precip_paths, orog_precip_figs))

    return tc

