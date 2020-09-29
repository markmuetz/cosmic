import string

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (diag_orog_precip_path_tpl, diag_orog_precip_fig_tpl, fmtp)
from orog_precip_paths import (orog_precip_path_tpl, orog_precip_fig_tpl, orog_precip_mean_fields_tpl, fmtp)
from orog_precip_paths import D23_fig2, D23_fig3, D23_fig4, D23_fig5


@remake_required(depends_on=[configure_ax_asia])
def plot_fig2(inputs, outputs):

    m1_cubes = iris.load([str(inputs[f'al508_diag_{month:02}']) for month in [6, 7, 8]]).concatenate()
    m2_cubes = iris.load(str(inputs['al508_direct']))

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
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(-0.07, 0.95, f'({c})', size=12, transform=ax.transAxes)

    cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])

    axes[0, 0].set_title('M1')
    axes[0, 1].set_title('M2')
    axes[0, 0].set_ylabel('non-orographic')
    axes[1, 0].set_ylabel('orographic')

    plt.colorbar(im, cax=cax, orientation='horizontal', label='precip. (mm day$^{-1}$)')
    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.09)

    plt.savefig(outputs[0])


def plot_fig3(inputs, outputs):
    m1_cubes = iris.load([str(inputs[f'al508_diag_{month:02}']) for month in [6, 7, 8]]).concatenate()
    m2_cubes = iris.load(str(inputs['al508_direct']))

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
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(-0.07, 0.95, f'({c})', size=12, transform=ax.transAxes)

    axes[0].set_title('M1')
    axes[1].set_title('M2')

    plt.colorbar(im, cax=cax, orientation='horizontal', label='% orog.')
    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.09)

    plt.savefig(outputs[0])


def plot_fig4(inputs, outputs):
    al508_cubes = iris.load(str(inputs['al508_direct']))
    ak543_cubes = iris.load(str(inputs['ak543_direct']))

    al508_orog_precip_mean = al508_cubes.extract_strict('orog_precipitation_flux_mean')
    al508_nonorog_precip_mean = al508_cubes.extract_strict('non_orog_precipitation_flux_mean')
    al508_ocean_precip_mean = al508_cubes.extract_strict('ocean_precipitation_flux_mean')
    al508_extent = util.get_extent_from_cube(al508_orog_precip_mean)

    ak543_orog_precip_mean = ak543_cubes.extract_strict('orog_precipitation_flux_mean')
    ak543_nonorog_precip_mean = ak543_cubes.extract_strict('non_orog_precipitation_flux_mean')
    ak543_ocean_precip_mean = ak543_cubes.extract_strict('ocean_precipitation_flux_mean')
    ak543_extent = util.get_extent_from_cube(ak543_orog_precip_mean)

    al508_mean = al508_orog_precip_mean.data + al508_nonorog_precip_mean.data + al508_ocean_precip_mean.data
    ak543_mean = ak543_orog_precip_mean.data + ak543_nonorog_precip_mean.data + ak543_ocean_precip_mean.data

    al508_opf = 100 * al508_orog_precip_mean.data / (al508_orog_precip_mean.data + al508_nonorog_precip_mean.data)
    ak543_opf = 100 * ak543_orog_precip_mean.data / (ak543_orog_precip_mean.data + ak543_nonorog_precip_mean.data)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), subplot_kw={'projection': ccrs.PlateCarree()})

    extent = al508_extent
    for ax in axes.flatten():
        configure_ax_asia(ax, tight_layout=False)
    im0 = axes[0, 0].imshow(al508_mean,
                            origin='lower', extent=extent, norm=mpl.colors.LogNorm(),
                            vmin=1e-2, vmax=1e2)
    im1 = axes[1, 0].imshow(ak543_mean,
                            origin='lower', extent=extent, norm=mpl.colors.LogNorm(),
                            vmin=1e-2, vmax=1e2)

    im3 = axes[0, 1].imshow(al508_opf,
                            origin='lower', extent=extent,
                            vmin=0, vmax=100)
    im4 = axes[1, 1].imshow(ak543_opf,
                            origin='lower', extent=extent,
                            vmin=0, vmax=100)

    for ax in axes.flatten()[:2]:
        ax.get_xaxis().set_ticks([])
    for ax in axes[:, 1].flatten():
        ax.get_yaxis().set_ticks([])
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(-0.07, 0.95, f'({c})', size=12, transform=ax.transAxes)

    axes[0, 0].set_title('total precip.')
    axes[0, 1].set_title('% orog.')
    axes[0, 0].set_ylabel('N1280')
    axes[1, 0].set_ylabel('N1280-EC')

    cax1 = fig.add_axes([0.1, 0.1, 0.4, 0.03])
    cax2 = fig.add_axes([0.56, 0.1, 0.4, 0.03])

    plt.colorbar(im1, cax=cax1, orientation='horizontal', label='precip. (mm day$^{-1}$)')
    plt.colorbar(im3, cax=cax2, orientation='horizontal', label='% orog.')
    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.09)
    plt.savefig(outputs[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax in axes.flatten():
        configure_ax_asia(ax, tight_layout=False)
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(-0.07, 0.95, f'({c})', size=12, transform=ax.transAxes)

    levels = [-20, -10, -5, -1, 1, 5, 10, 20]
    colour_levels = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99]

    norm1 = mpl.colors.BoundaryNorm(levels, ncolors=256)
    colours = [(i / (len(colour_levels) - 1), mpl.cm.bwr_r(v)) for i, v in enumerate(colour_levels)]
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('frac_list', colours)
    im2 = axes[0].imshow(ak543_mean - al508_mean,
                            origin='lower', extent=extent, cmap=cmap1, norm=norm1)

    levels = [-100, -50, -10, -1, 1, 10, 50, 100]
    colour_levels = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99]
    norm2 = mpl.colors.BoundaryNorm(levels, ncolors=256)
    colours = [(i / (len(colour_levels) - 1), mpl.cm.bwr_r(v)) for i, v in enumerate(colour_levels)]
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('frac_list', colours)
    im5 = axes[1].imshow(ak543_opf - al508_opf,
                            origin='lower', extent=extent,
                            vmin=-100, vmax=100, cmap=cmap2, norm=norm2)

    axes[1].get_yaxis().set_ticks([])
    axes[0].set_title('$\Delta$ total precip.')
    axes[1].set_title('$\Delta$ % orog.')

    cax3 = fig.add_axes([0.1, 0.12, 0.4, 0.03])
    cax4 = fig.add_axes([0.56, 0.12, 0.4, 0.03])
    plt.suptitle('N1280-EC $\minus$ N1280')
    plt.colorbar(im2, cax=cax3, orientation='horizontal', label='$\Delta$ precip. (mm day$^{-1}$)', extend='both')
    plt.colorbar(im5, cax=cax4, orientation='horizontal', label='$\Delta$ % orog.')

    plt.subplots_adjust(left=0.1, top=0.96, right=0.98, bottom=0.2, hspace=0.07, wspace=0.09)
    plt.savefig(outputs[1])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    year = 2006
    model = 'al508'
    dist_thresh = 100
    dotprod_thresh = 0.05

    inputs = {f'al508_diag_{month:02}': fmtp(diag_orog_precip_path_tpl, model=model, year=year, month=month)
              for month in [6, 7, 8]}

    inputs['al508_direct'] = fmtp(orog_precip_mean_fields_tpl, model='al508', year=year, season='jja',
                                  dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)
    inputs['ak543_direct'] = fmtp(orog_precip_mean_fields_tpl, model='ak543', year=year, season='jja',
                                  dotprod_thresh=dotprod_thresh, dist_thresh=dist_thresh)

    tc.add(Task(plot_fig2, inputs, [D23_fig2]))
    tc.add(Task(plot_fig3, inputs, [D23_fig3]))
    tc.add(Task(plot_fig4, inputs, [D23_fig4, D23_fig5]))

    return tc
