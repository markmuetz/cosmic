import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (diag_orog_precip_path_tpl, diag_orog_precip_fig_tpl, fmtp)
from orog_precip_paths import (orog_precip_path_tpl, orog_precip_fig_tpl, orog_precip_mean_fields_tpl, fmtp)
from orog_precip_paths import D23_fig2, D23_fig3


@remake_required(depends_on=[configure_ax_asia])
def plot_fig2(inputs, outputs):

    m1_cubes = iris.load([str(inputs[f'diag_{month:02}']) for month in [6, 7, 8]]).concatenate()
    m2_cubes = iris.load(str(inputs['direct']))

    m1_orog_precip = m1_cubes.extract_strict('orog_precipitation_flux')
    m1_nonorog_precip = m1_cubes.extract_strict('non_orog_precipitation_flux')
    assert m1_orog_precip.units == 'kg m-2 s-1'
    assert m1_nonorog_precip.units == 'kg m-2 s-1'

    m1_extent = util.get_extent_from_cube(m1_orog_precip)
    m1_orog_precip_mean = m1_orog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    m1_nonorog_precip_mean = m1_nonorog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1

    m2_orog_precip_mean = m2_cubes.extract_strict('orog_precipitation_flux_mean')
    m2_nonorog_precip_mean = m2_cubes.extract_strict('non_orog_precipitation_flux_mean')
    m2_extent = util.get_extent_from_cube(m2_orog_precip_mean)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax, precip, extent in zip(axes.flatten(),
                                  [m1_nonorog_precip_mean, m2_nonorog_precip_mean.data,
                                   m1_orog_precip_mean, m2_orog_precip_mean.data],
                                  [m1_extent, m2_extent, m1_extent, m2_extent]):
        configure_ax_asia(ax, tight_layout=False)
        im = ax.imshow(precip, origin='lower', extent=extent, norm=mpl.colors.LogNorm(),
                       vmin=1e-2, vmax=1e2)

    for ax in axes[0, :].flatten():
        ax.get_xaxis().set_ticks([])
    for ax in axes[:, 1].flatten():
        ax.get_yaxis().set_ticks([])
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])

    axes[0, 0].set_title('M1')
    axes[0, 1].set_title('M2')
    axes[0, 0].set_ylabel('non-orographic')
    axes[1, 0].set_ylabel('orographic')

    plt.colorbar(im, cax=cax, orientation='horizontal', label='precip. (mm day$^{-1}$)')
    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.07)

    plt.savefig(outputs[0])


@remake_required(depends_on=[configure_ax_asia])
def plot_fig3(inputs, outputs):

    m1_cubes = iris.load([str(inputs[f'diag_{month:02}']) for month in [6, 7, 8]]).concatenate()
    m2_cubes = iris.load(str(inputs['direct']))

    m1_orog_precip = m1_cubes.extract_strict('orog_precipitation_flux')
    m1_nonorog_precip = m1_cubes.extract_strict('non_orog_precipitation_flux')
    m1_ocean_precip = m1_cubes.extract_strict('ocean_precipitation_flux')
    assert m1_orog_precip.units == 'kg m-2 s-1'
    assert m1_nonorog_precip.units == 'kg m-2 s-1'
    assert m1_ocean_precip.units == 'kg m-2 s-1'

    m1_extent = util.get_extent_from_cube(m1_orog_precip)
    m1_orog_precip_mean = m1_orog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    m1_nonorog_precip_mean = m1_nonorog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1
    m1_ocean_precip_mean = m1_nonorog_precip.data.mean(axis=0) * 3600 * 24  # kg m-2 s-1 -> mm day-1

    m2_orog_precip_mean = m2_cubes.extract_strict('orog_precipitation_flux_mean')
    m2_nonorog_precip_mean = m2_cubes.extract_strict('non_orog_precipitation_flux_mean')
    m2_ocean_precip_mean = m2_cubes.extract_strict('ocean_precipitation_flux_mean')
    m2_extent = util.get_extent_from_cube(m2_orog_precip_mean)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax, (nonorog, orog, ocean), extent in zip(axes.flatten(),
                                                  [(m1_nonorog_precip_mean,
                                                    m1_orog_precip_mean,
                                                    m1_ocean_precip_mean),
                                                   (m2_nonorog_precip_mean.data,
                                                    m2_orog_precip_mean.data,
                                                    m2_ocean_precip_mean.data)],
                                                  [m1_extent, m2_extent]):
        configure_ax_asia(ax, tight_layout=False)
        im = ax.imshow(100 * orog / (orog + nonorog),
                       origin='lower', extent=extent, vmin=0, vmax=100)

    axes[1].get_yaxis().set_ticks([])
    cax = fig.add_axes([0.1, 0.12, 0.8, 0.03])

    axes[0].set_title('M1')
    axes[1].set_title('M2')

    plt.colorbar(im, cax=cax, orientation='horizontal', label='% orog')
    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.07)

    plt.savefig(outputs[0])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    year = 2006
    model = 'al508'
    dist_thresh = 100
    dotprod_thresh = 0.05

    inputs = {f'diag_{month:02}': fmtp(diag_orog_precip_path_tpl, model=model, year=year, month=month)
              for month in [6, 7, 8]}

    inputs['direct'] = fmtp(orog_precip_mean_fields_tpl, model=model, year=year, season='jja',
                            dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)

    tc.add(Task(plot_fig2, inputs, [D23_fig2]))
    tc.add(Task(plot_fig3, inputs, [D23_fig3]))

    return tc

