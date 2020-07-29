import string
from collections import defaultdict
import itertools
import pickle
from logging import getLogger

import iris
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib import colors
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from cosmic.util import load_cmap_data, rmse_mask_out_nan, mae_mask_out_nan, get_extent_from_cube
from cosmic.mid_point_norm import MidPointNorm
from remake import Task, TaskControl, remake_required, remake_task_control

from cosmic.config import PATHS, STANDARD_NAMES
from cosmic.plotting_util import configure_ax_asia

from basin_weighted_config import DATASETS, HB_NAMES, PRECIP_MODES
import basin_weighted_analysis

logger = getLogger('remake.basin_weighted_analysis')


@remake_required(depends_on=[configure_ax_asia])
def plot_hydrobasins_files(inputs, outputs, hb_name):
    hb_size = gpd.read_file(str(inputs['shp']))
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_title(f'scale:{hb_name}, #basins:{len(hb_size)}')
    hb_size.plot(ax=ax)
    hb_size.geometry.boundary.plot(ax=ax, color=None, edgecolor='k', linewidth=0.5)
    configure_ax_asia(ax)
    plt.savefig(outputs[0])
    plt.close('all')


@remake_required(depends_on=[configure_ax_asia])
def plot_mean_precip(inputs, outputs, dataset, hb_name):
    weighted_basin_mean_precip_filename = inputs['weighted']
    df_mean_precip = pd.read_hdf(weighted_basin_mean_precip_filename)
    mean_max_min_precip = pickle.loads(inputs['mean_precip_max_min'].read_bytes())
    max_mean_precip = mean_max_min_precip['max_mean_precip']
    # min_mean_precip = mean_max_min_precip['min_mean_precip']

    raster_hb_name = hb_name
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
    raster = raster_cube.data
    logger.debug(f'Plot maps - {hb_name}: {dataset}')

    mean_precip_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        mean_precip_map[raster == i] = df_mean_precip.values[i - 1]

    extent = get_extent_from_cube(raster_cube)
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    plt.title(f'{dataset} mean_precip')
    grey_fill = np.zeros((mean_precip_map.shape[0], mean_precip_map.shape[1], 3), dtype=int)
    grey_fill[raster_cube.data == 0] = (200, 200, 200)
    ax.imshow(grey_fill, extent=extent)

    masked_mean_precip_map = np.ma.masked_array(mean_precip_map, raster_cube.data == 0)
    im = ax.imshow(masked_mean_precip_map * 24,
                   cmap=cmap, norm=norm,
                   # vmin=1e-3, vmax=max_mean_precip,
                   origin='lower', extent=extent)
    plt.colorbar(im, label=f'precip. (mm day$^{{-1}}$)',
                 **cbar_kwargs, spacing='uniform',
                 orientation='horizontal')
    configure_ax_asia(ax, extent)

    mean_precip_filename = outputs[0]
    plt.savefig(mean_precip_filename)
    plt.close()


def _configure_hb_name_dataset_map_grid(axes, hb_names, datasets):
    for ax in axes.flatten():
        configure_ax_asia(ax, tight_layout=False)
    for ax, hb_name in zip(axes[0], hb_names):
        if hb_name == 'med':
            ax.set_title('medium')
        else:
            ax.set_title(hb_name)
    for ax in axes[:, 2].flatten():
        ax.get_yaxis().tick_right()
    for ax in axes[:, :2].flatten():
        ax.get_yaxis().set_ticks([])
    for ax in axes[:len(axes) - 1, :].flatten():
        ax.get_xaxis().set_ticks([])
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(0.01, 1.04, f'({c})', size=12, transform=ax.transAxes)


@remake_required(depends_on=[configure_ax_asia, _configure_hb_name_dataset_map_grid])
def plot_mean_precip_asia_combined(inputs, outputs, datasets, hb_names):
    imshow_data = {}
    for hb_name in hb_names:
        raster_hb_name = hb_name
        raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
        raster = raster_cube.data
        extent = get_extent_from_cube(raster_cube)

        cmorph_weighted_basin_mean_precip_filename = inputs[f'weighted_{hb_name}_cmorph']
        df_cmorph_mean_precip = pd.read_hdf(cmorph_weighted_basin_mean_precip_filename)

        cmorph_mean_precip_map = np.zeros_like(raster, dtype=float)
        for i in range(1, raster.max() + 1):
            cmorph_mean_precip_map[raster == i] = df_cmorph_mean_precip.values[i - 1]

        masked_cmorph_mean_precip_map = np.ma.masked_array(cmorph_mean_precip_map, raster_cube.data == 0)
        imshow_data[('cmorph', hb_name)] = masked_cmorph_mean_precip_map * 24

        for dataset in datasets[1:]:
            weighted_basin_mean_precip_filename = inputs[f'weighted_{hb_name}_{dataset}']
            df_mean_precip = pd.read_hdf(weighted_basin_mean_precip_filename)

            mean_precip_map = np.zeros_like(raster, dtype=float)
            for i in range(1, raster.max() + 1):
                mean_precip_map[raster == i] = df_mean_precip.values[i - 1]

            masked_mean_precip_map = np.ma.masked_array(mean_precip_map - cmorph_mean_precip_map,
                                                        raster_cube.data == 0)
            imshow_data[(dataset, hb_name)] = masked_mean_precip_map * 24

    figsize = (10, 5.5) if len(datasets) == 3 else (10, 8)
    fig, axes = plt.subplots(len(datasets), 3,
                             figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')

    # orig. -- continuous scale.
    # diff_cmap = mpl.cm.get_cmap('bwr')
    # diff_norm = MidPointNorm(0, -24, 72)
    bwr = mpl.cm.get_cmap('bwr')
    cmap_scale = [0., .5 / 3, 1 / 3, .5, .6, .7, .8, 1.]  # 8 -- colours from bwr to use.
    diff_bounds = [-27, -9, -3, -1, 1, 3, 9, 27, 81]  # 9 -- bounds to use.

    # https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
    top = mpl.cm.get_cmap('Oranges_r', 128)
    bottom = mpl.cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = colors.ListedColormap(newcolors, name='OrangeBlue')
    diff_cmap = colors.LinearSegmentedColormap.from_list('diff_cmap', [newcmp(x) for x in cmap_scale], newcmp.N)
    diff_norm = colors.BoundaryNorm(diff_bounds, diff_cmap.N)

    # Fills masked values.
    cmap.set_bad(color='k', alpha=0.1)
    diff_cmap.set_bad(color='k', alpha=0.1)

    for axrow, hb_name in zip(axes.T, hb_names):
        masked_cmorph_mean_precip_map = imshow_data[('cmorph', hb_name)]
        ax = axrow[0]

        cmorph_im = ax.imshow(masked_cmorph_mean_precip_map,
                              cmap=cmap, norm=norm,
                              # vmin=1e-3, vmax=max_mean_precip,
                              origin='lower', extent=extent)

        for ax, dataset in zip(axrow.T[1:], datasets[1:]):
            masked_mean_precip_map = imshow_data[(dataset, hb_name)]
            dataset_im = ax.imshow(masked_mean_precip_map,
                                   cmap=diff_cmap,
                                   norm=diff_norm,
                                   # vmin=-absmax, vmax=absmax,
                                   origin='lower', extent=extent)

    for ax, dataset in zip(axes[:, 0], datasets):
        if dataset == 'cmorph':
            ax.set_ylabel(STANDARD_NAMES[dataset])
        else:
            ax.set_ylabel(f'{STANDARD_NAMES[dataset]} $-$ {STANDARD_NAMES["cmorph"]}')
    _configure_hb_name_dataset_map_grid(axes, hb_names, datasets)

    cax = fig.add_axes([0.92, 0.66, 0.01, 0.3])
    # plt.colorbar(cmorph_im, cax=cax, orientation='vertical', label='precipitation (mm day$^{-1}$)', **cbar_kwargs)
    plt.colorbar(cmorph_im, cax=cax, orientation='vertical', **cbar_kwargs)
    cax.text(5.8, 1, 'precipitation (mm day$^{-1}$)', rotation=90)

    # cax2 = fig.add_axes([0.46, 0.07, 0.4, 0.01])
    cax2 = fig.add_axes([0.92, 0.02, 0.01, 0.6])
    # cb = plt.colorbar(dataset_im, cax=cax2, orientation='vertical', label='$\\Delta$ precipitation (mm hr$^{-1}$)')
    cb = plt.colorbar(dataset_im, cax=cax2, orientation='vertical')
    cax2.text(5.8, 0.8, '$\\Delta$ precipitation (mm day$^{-1}$)', rotation=90)
    # cb.set_label_coords(-0.2, 0.5)

    plt.subplots_adjust(left=0.06, right=0.86, top=0.96, bottom=0.04, wspace=0.1, hspace=0.15)

    mean_precip_filename = outputs[0]
    plt.savefig(mean_precip_filename)
    plt.close()


@remake_required(depends_on=[configure_ax_asia])
def plot_obs_mean_precip_diff(inputs, outputs, dataset, hb_name):
    weighted_basin_mean_precip_filename = inputs['dataset_weighted']
    obs_weighted_basin_mean_precip_filename = inputs['obs_weighted']

    df_mean_precip = pd.read_hdf(weighted_basin_mean_precip_filename)
    df_obs_mean_precip = pd.read_hdf(obs_weighted_basin_mean_precip_filename)

    # There can be NaNs in the datasets, as in e.g. APHRODITE at some fine-scale basins.
    # Use version of RMSE and MAE that mask these out.
    obs_rmse = rmse_mask_out_nan(df_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float),
                                 df_obs_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float))
    obs_mae = mae_mask_out_nan(df_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float),
                               df_obs_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float))

    raster_hb_name = hb_name
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
    raster = raster_cube.data

    mean_precip_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        mean_precip_map[raster == i] = df_mean_precip.values[i - 1]

    obs_mean_precip_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        obs_mean_precip_map[raster == i] = df_obs_mean_precip.values[i - 1]

    extent = get_extent_from_cube(raster_cube)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    plt.title(f'{dataset} mean_precip. RMSE: {obs_rmse:.3f} mm hr$^{{-1}}$; MAE: {obs_mae:.3f} mm hr$^{{-1}}$')
    grey_fill = np.zeros((mean_precip_map.shape[0], mean_precip_map.shape[1], 3), dtype=int)
    grey_fill[raster_cube.data == 0] = (200, 200, 200)
    ax.imshow(grey_fill, extent=extent)

    masked_mean_precip_map = np.ma.masked_array(mean_precip_map - obs_mean_precip_map, raster_cube.data == 0)

    im = ax.imshow(masked_mean_precip_map,
                   cmap='bwr',
                   norm=MidPointNorm(0, -1, 3),
                   # vmin=-absmax, vmax=absmax,
                   origin='lower', extent=extent)

    plt.colorbar(im, label=f'precip. (mm hr$^{{-1}}$)',
                 orientation='horizontal')
    configure_ax_asia(ax, extent)
    mean_precip_filename = outputs[0]
    plt.savefig(mean_precip_filename)
    plt.close()


def _plot_phase_alpha(ax, masked_phase_map, masked_mag_map, cmap, norm, extent):
    thresh_boundaries = [100 * 1 / 3, 100 * 2 / 3]
    # thresh_boundaries = [100 * 1 / 4, 100 * 1 / 3]
    med_thresh, strong_thresh = np.percentile(masked_mag_map.compressed(),
                                              thresh_boundaries)
    peak_strong = np.ma.masked_array(masked_phase_map,
                                     masked_mag_map < strong_thresh)
    peak_med = np.ma.masked_array(masked_phase_map,
                                  ((masked_mag_map >= strong_thresh) |
                                   (masked_mag_map < med_thresh)))
    peak_weak = np.ma.masked_array(masked_phase_map,
                                   masked_mag_map >= med_thresh)
    im0 = ax.imshow(peak_strong, origin='lower', extent=extent,
                    vmin=0, vmax=24, cmap=cmap, norm=norm)
    ax.imshow(peak_med, origin='lower', extent=extent, alpha=0.66,
              vmin=0, vmax=24, cmap=cmap, norm=norm)
    ax.imshow(peak_weak, origin='lower', extent=extent, alpha=0.33,
              vmin=0, vmax=24, cmap=cmap, norm=norm)
    # # plt.colorbar(im0, orientation='horizontal')
    # cax = fig.add_axes([0.05, 0.05, 0.9, 0.05])
    return im0


@remake_required(depends_on=[_plot_phase_alpha, configure_ax_asia, _configure_hb_name_dataset_map_grid])
def plot_phase_alpha_combined(inputs, outputs, datasets, hb_names, mode):
    imshow_data = {}
    for hb_name in hb_names:
        raster_hb_name = hb_name
        raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
        raster = raster_cube.data
        for dataset in datasets:
            weighted_basin_phase_mag_filename = inputs[f'weighted_{hb_name}_{dataset}']
            df_phase_mag = pd.read_hdf(weighted_basin_phase_mag_filename)

            phase_map, mag_map = gen_map_from_basin_values(df_phase_mag, raster)
            phase_map = iris.cube.Cube(phase_map, long_name='phase_map', units='hr',
                                       dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                            (raster_cube.coord('longitude'), 1)])
            mag_map = iris.cube.Cube(mag_map, long_name='magnitude_map', units='-',
                                     dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                          (raster_cube.coord('longitude'), 1)])
            masked_phase_map = np.ma.masked_array(phase_map.data, raster_cube.data == 0)
            masked_mag_map = np.ma.masked_array(mag_map.data, raster_cube.data == 0)
            imshow_data[(hb_name, dataset)] = (masked_phase_map, masked_mag_map)

    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
    # Fills masked values.
    cmap.set_bad(color='k', alpha=0.1)
    figsize = (10, 7) if len(datasets) == 3 else (10, 9)
    fig, axes = plt.subplots(len(datasets), 3, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    for axrow, dataset in zip(axes, datasets):
        for ax, hb_name in zip(axrow, hb_names):
            masked_phase_map, masked_mag_map = imshow_data[(hb_name, dataset)]
            extent = get_extent_from_cube(phase_map)
            _plot_phase_alpha(ax, masked_phase_map, masked_mag_map, cmap, norm, extent)

    for ax, dataset in zip(axes[:, 0], datasets):
        ax.set_ylabel(STANDARD_NAMES[dataset])
    _configure_hb_name_dataset_map_grid(axes, hb_names, datasets)

    cax = fig.add_axes([0.10, 0.07, 0.8, 0.05])
    v = np.linspace(0, 1, 24)
    d = cmap(v)[None, :, :4] * np.ones((3, 24, 4))
    d[1, :, 3] = 0.66
    d[0, :, 3] = 0.33
    cax.imshow(d, origin='lower', extent=(0, 24, 0, 2), aspect='auto')
    cax.set_yticks([0.3, 1.7])
    cax.set_yticklabels(['weak', 'strong'])
    cax.set_xticks(np.linspace(0, 24, 9))
    cax.set_xlabel('phase and strength of diurnal cycle')

    plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.16, wspace=0.1, hspace=0.15)

    plt.savefig(outputs[0])


@remake_required(depends_on=[_plot_phase_alpha, configure_ax_asia])
def plot_phase_mag(inputs, outputs, dataset, hb_name, mode):
    weighted_basin_phase_mag_filename = inputs['weighted']
    df_phase_mag = pd.read_hdf(weighted_basin_phase_mag_filename)

    raster_hb_name = hb_name
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
    raster = raster_cube.data
    phase_filename, alpha_phase_filename, mag_filename = outputs
    print(f'Plot maps - {hb_name}_{mode}: {dataset}')
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

    phase_map, mag_map = gen_map_from_basin_values(df_phase_mag, raster)
    phase_map = iris.cube.Cube(phase_map, long_name='phase_map', units='hr',
                               dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                    (raster_cube.coord('longitude'), 1)])
    mag_map = iris.cube.Cube(mag_map, long_name='magnitude_map', units='-',
                             dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                  (raster_cube.coord('longitude'), 1)])

    extent = get_extent_from_cube(phase_map)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_title(f'{dataset} {mode} phase')
    masked_phase_map = np.ma.masked_array(phase_map.data, raster_cube.data == 0)
    masked_mag_map = np.ma.masked_array(mag_map.data, raster_cube.data == 0)

    im = ax.imshow(masked_phase_map,
                   cmap=cmap, norm=norm,
                   origin='lower', extent=extent, vmin=0, vmax=24)
    plt.colorbar(im, orientation='horizontal')
    configure_ax_asia(ax, extent)
    # plt.tight_layout()
    plt.savefig(phase_filename)
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_title(f'{dataset} {mode} phase (alpha)')

    _plot_phase_alpha(ax, masked_phase_map, masked_mag_map, cmap, norm, extent)
    configure_ax_asia(ax, extent)

    cax, _ = cbar.make_axes_gridspec(ax, orientation='horizontal')
    v = np.linspace(0, 1, 24)
    d = cmap(v)[None, :, :4] * np.ones((3, 24, 4))
    d[1, :, 3] = 0.66
    d[0, :, 3] = 0.33
    cax.imshow(d, origin='lower', extent=(0, 24, 0, 2), aspect='auto')
    cax.set_yticks([])
    cax.set_xticks(np.linspace(0, 24, 9))

    # plt.tight_layout()
    plt.savefig(alpha_phase_filename)
    plt.close()

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_title(f'{dataset} {mode} strength')
    im = ax.imshow(masked_mag_map,
                   origin='lower', extent=extent, vmin=1e-2, norm=LogNorm())
    plt.colorbar(im, orientation='horizontal')
    configure_ax_asia(ax, extent)
    # plt.tight_layout()
    plt.savefig(mag_filename)
    plt.close()


def gen_map_from_basin_values(cmorph_phase_mag, raster):
    phase_mag = cmorph_phase_mag.values
    phase_map = np.zeros_like(raster, dtype=float)
    mag_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        phase_map[raster == i] = phase_mag[i - 1, 0]
        mag_map[raster == i] = phase_mag[i - 1, 1]
    return phase_map, mag_map


def plot_obs_vs_all_datasets_mean_precip(inputs, outputs, disp_mae=False):
    with inputs[0].open('rb') as f:
        all_rmses = pickle.load(f)

    all_rmse_filename, all_corr_filename = outputs

    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True, num=str(all_rmse_filename), figsize=(8, 5))

    # ax1.set_ylim((0, 5))
    for dataset, (rmses, maes, _) in list(all_rmses.items())[::-1]:
        # Skip N1280 Ensemble members.
        if dataset in ['u-aj399', 'u-az035']:
            continue
        p = ax.plot(np.array(rmses) * 24, label=STANDARD_NAMES[dataset])
        colour = p[0].get_color()
        if dataset == 'u-al508':
            colour_N1280 = colour
        if disp_mae:
            ax.plot(np.array(maes) * 24, linestyle='--', color=colour)

    # Shade all between parametrized N1280 sims.
    rmses_N1280 = []
    maes_N1280 = []
    for dataset in ['u-al508', 'u-aj399', 'u-az035']:
        rmses_N1280.append(all_rmses[dataset][0])
        maes_N1280.append(all_rmses[dataset][1])
    rmses_N1280 = np.array(rmses_N1280) * 24
    maes_N1280 = np.array(rmses_N1280) * 24
    ax.fill_between(range(len(rmses)), rmses_N1280.min(axis=0), rmses_N1280.max(axis=0), color=colour_N1280, alpha=0.5)
    if disp_mae:
        ax.fill_between(range(len(rmses)), maes_N1280.min(axis=0), maes_N1280.max(axis=0), color=colour_N1280, alpha=0.5)

    if len(rmses) == 3:
        ax.set_xticks([0, 1, 2])
    elif len(rmses) == 11:
        ax.set_xticks([0, 5, 10])
        ax.set_xticks(range(11), minor=True)
    # ax.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
    # ax.set_xticklabels(['small', 'medium', 'large'])
    ax.set_xticklabels(['small\n5040 km${^2}$', 'medium\n54600 km${^2}$', 'large\n55300 km${^2}$'])

    if disp_mae:
        ax.set_ylabel('mean precip.\nRMSE/MAE (mm day$^{-1}$)')
    else:
        ax.set_ylabel('mean precip.\nRMSE (mm day$^{-1}$)')
    ax.set_xlabel('basin size')
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(all_rmse_filename)

    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True, num=str(all_corr_filename), figsize=(12, 8))

    # ax1.set_ylim((0, 5))
    for dataset, (_, _, corrs) in list(all_rmses.items())[::-1]:

        r2 = [c.rvalue**2 for c in corrs]
        slope = [c.slope for c in corrs]

        p = ax.plot(r2, label=STANDARD_NAMES[dataset])
        colour = p[0].get_color()
        # ax.plot(slope, linestyle='--', color=colour)

    if len(rmses) == 3:
        ax.set_xticks([0, 1, 2])
    elif len(rmses) == 11:
        ax.set_xticks([0, 5, 10])
        ax.set_xticks(range(11), minor=True)
    # ax.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
    # ax.set_xticklabels(['small', 'medium', 'large'])
    ax.set_xticklabels(['small\n5040 km${^2}$', 'medium\n54600 km${^2}$', 'large\n55300 km${^2}$'])

    # ax.set_ylabel('correlations ($r^2$ - solid, slope - dashed)')
    ax.set_ylabel('correlations ($r^2$)')
    ax.set_xlabel('basin size')
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(all_corr_filename)


def plot_cmorph_vs_all_datasets_phase_mag(inputs, outputs):
    with inputs[0].open('rb') as f:
        all_rmses = pickle.load(f)

    all_rmse_filename = outputs[0]

    # N.B. VRMSE no longer used (Vector RMSE -- combination of phase/amplitude)
    # fig, axes = plt.subplots(3, 3, sharex=True, num=str(all_rmse_filename), figsize=(12, 8))
    fig, axes = plt.subplots(2, 3, sharex=True, num=str(all_rmse_filename), figsize=(10, 6))
    for i, mode in enumerate(PRECIP_MODES):
        ax1, ax2 = axes[:, i]
        # ax1.set_ylim((0, 5))
        rmses_for_mode = all_rmses[mode]
        for dataset, (phase_rmses, mag_rmses, vrmses) in list(rmses_for_mode.items())[::-1]:
            if dataset in ['u-aj399', 'u-az035']:
                continue
            p = ax1.plot(phase_rmses, label=STANDARD_NAMES[dataset])
            ax2.plot(mag_rmses, label=STANDARD_NAMES[dataset])
            # ax3.plot(vrmses, label=STANDARD_NAMES[dataset])
            colour = p[0].get_color()
            if dataset == 'u-al508':
                colour_N1280 = colour

        # Shade all between parametrized N1280 sims.
        phase_rmses_N1280 = []
        mag_rmses_N1280 = []
        # vrmses_N1280 = []
        for dataset in ['u-al508', 'u-aj399', 'u-az035']:
            phase_rmses_N1280.append(rmses_for_mode[dataset][0])
            mag_rmses_N1280.append(rmses_for_mode[dataset][1])
            # vrmses_N1280.append(rmses_for_mode[dataset][2])
        phase_rmses_N1280 = np.array(phase_rmses_N1280)
        mag_rmses_N1280 = np.array(mag_rmses_N1280)
        # vrmses_N1280 = np.array(vrmses_N1280)
        ax1.fill_between(range(len(phase_rmses)),
                         phase_rmses_N1280.min(axis=0),
                         phase_rmses_N1280.max(axis=0), color=colour_N1280, alpha=0.5)
        ax2.fill_between(range(len(phase_rmses)),
                         mag_rmses_N1280.min(axis=0),
                         mag_rmses_N1280.max(axis=0), color=colour_N1280, alpha=0.5)
        # ax3.fill_between(range(len(phase_rmses)),
        #                  vrmses_N1280.min(axis=0),
        #                  vrmses_N1280.max(axis=0), color=colour_N1280, alpha=0.5)

        if len(vrmses) == 3:
            ax2.set_xticks([0, 1, 2])
        elif len(vrmses) == 11:
            ax2.set_xticks([0, 5, 10])
            ax2.set_xticks(range(11), minor=True)
        # ax2.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
        # ax2.set_xticklabels(['small', 'medium', 'large'])
        ax2.set_xticklabels(['small\n5040 km${^2}$', 'medium\n54600 km${^2}$', 'large\n55300 km${^2}$'])

    axes[0, 0].set_title('amount')
    axes[0, 1].set_title('frequency')
    axes[0, 2].set_title('intensity')
    axes[0, 0].set_ylabel('phase\ncircular RMSE (hr)')
    axes[0, 0].get_yaxis().set_label_coords(-0.2, 0.5)
    axes[1, 0].set_ylabel('amplitude\nRMSE (-)')
    axes[1, 0].set_ylim((0.04, 0.145))
    axes[1, 0].get_yaxis().set_label_coords(-0.2, 0.5)
    # axes[2, 0].set_ylabel('combined\nVRMSE (-)')
    # axes[2, 0].get_yaxis().set_label_coords(-0.2, 0.5)
    axes[1, 1].set_xlabel('basin size')
    axes[1, 0].legend()

    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(0.01, 1.04, f'({c})', size=12, transform=ax.transAxes)

    plt.subplots_adjust(left=0.1, right=0.94, top=0.96, bottom=0.12, wspace=0.2, hspace=0.2)
    plt.savefig(all_rmse_filename)


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    for basin_scales in ['small_medium_large', 'sliding']:
        hb_raster_cubes_fn = PATHS['output_datadir'] / f'basin_weighted_analysis/hb_N1280_raster_{basin_scales}.nc'
        if basin_scales == 'small_medium_large':
            hb_names = HB_NAMES
        else:
            hb_names = [f'S{i}' for i in range(11)]
        shp_path_tpl = 'basin_weighted_analysis/{hb_name}/hb_{hb_name}.{ext}'
        for hb_name in hb_names:
            shp_inputs = {ext: PATHS['output_datadir'] / shp_path_tpl.format(hb_name=hb_name, ext=ext)
                          for ext in ['shp', 'dbf', 'prj', 'cpg', 'shx']}
            task_ctrl.add(Task(plot_hydrobasins_files,
                               shp_inputs,
                               [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' /
                                'hydrobasins_size' / f'map_{hb_name}.png'],
                               func_args=[hb_name],
                               ))

        weighted_mean_precip_tpl = 'basin_weighted_analysis/{hb_name}/' \
                                   '{dataset}.{hb_name}.area_weighted.mean_precip.hdf'

        weighted_mean_precip_filenames = defaultdict(list)
        for dataset, hb_name in itertools.product(DATASETS + ['aphrodite'], hb_names):
            fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name}
            max_min_path = PATHS['output_datadir'] / f'basin_weighted_analysis/{hb_name}/mean_precip_max_min.pkl'

            weighted_mean_precip_filename = PATHS['output_datadir'] / weighted_mean_precip_tpl.format(**fmt_kwargs)
            weighted_mean_precip_filenames[hb_name].append(weighted_mean_precip_filename)
            task_ctrl.add(Task(plot_mean_precip,
                               {'weighted': weighted_mean_precip_filename,
                                'raster_cubes': hb_raster_cubes_fn,
                                'mean_precip_max_min': max_min_path},
                               [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' /
                                'mean_precip' / f'map_{dataset}.{hb_name}.area_weighted.png'],
                               func_args=[dataset, hb_name]))

            if dataset != 'cmorph':
                fmt_kwargs = {'dataset': 'cmorph', 'hb_name': hb_name}
                cmorph_weighted_mean_precip_filename = (PATHS['output_datadir']
                                                        / weighted_mean_precip_tpl.format(**fmt_kwargs))
                task_ctrl.add(Task(plot_obs_mean_precip_diff,
                                   {'dataset_weighted': weighted_mean_precip_filename,
                                    'obs_weighted': cmorph_weighted_mean_precip_filename,
                                    'raster_cubes': hb_raster_cubes_fn,
                                    'mean_precip_max_min': max_min_path},
                                   [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' /
                                    'cmorph_mean_precip_diff' / f'map_{dataset}.{hb_name}.area_weighted.png'],
                                   func_args=[dataset, hb_name]))

            if dataset not in ['cmorph', 'aphrodite']:
                fmt_kwargs = {'dataset': 'aphrodite', 'hb_name': hb_name}
                obs_weighted_mean_precip_filename = (PATHS['output_datadir']
                                                     / weighted_mean_precip_tpl.format(**fmt_kwargs))
                task_ctrl.add(Task(plot_obs_mean_precip_diff,
                                   {'dataset_weighted': weighted_mean_precip_filename,
                                    'obs_weighted': obs_weighted_mean_precip_filename,
                                    'raster_cubes': hb_raster_cubes_fn,
                                    'mean_precip_max_min': max_min_path},
                                   [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' / 'cmorph_mean_precip_diff' /
                                    f'map_aphrodite_vs_{dataset}.{hb_name}.area_weighted.png'],
                                   func_args=[dataset, hb_name]))

        for obs in ['cmorph', 'aphrodite', 'u-al508', 'u-ak543']:
            mean_precip_rmse_data_filename = (PATHS['output_datadir'] /
                                              f'basin_weighted_analysis/{obs}.mean_precip_all_rmses.{basin_scales}.pkl')
            task_ctrl.add(Task(plot_obs_vs_all_datasets_mean_precip,
                               inputs=[mean_precip_rmse_data_filename],
                               outputs=[PATHS['figsdir'] / 'basin_weighted_analysis' / 'cmorph_vs' / 'mean_precip' /
                                        f'{obs}_vs_all_datasets.all_{f}.{basin_scales}.pdf' for f in ['rmse', 'corr']]
                               ))

        if basin_scales == 'small_medium_large':
            input_paths = {'raster_cubes': hb_raster_cubes_fn}
            for name, datasets in zip(['', 'full_'],
                                      (['cmorph', 'u-al508', 'u-ak543'], ['cmorph', 'u-al508', 'u-am754', 'u-ak543'])):
                paths = {f'weighted_{hb_name}_{dataset}': (PATHS['output_datadir'] /
                                                           weighted_mean_precip_tpl.format(hb_name=hb_name,
                                                                                           dataset=dataset))
                         for hb_name, dataset in itertools.product(hb_names, datasets)}
                input_paths.update(paths)
                task_ctrl.add(Task(plot_mean_precip_asia_combined,
                                   input_paths,
                                   [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' /
                                    'mean_precip_asia_combined' / f'{name}asia_combined_basin_scales.pdf'],
                                   func_args=[datasets, hb_names]))

        weighted_phase_mag_tpl = 'basin_weighted_analysis/{hb_name}/' \
                                 '{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.hdf'

        for dataset, hb_name, mode in itertools.product(DATASETS, hb_names, PRECIP_MODES):
            fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name, 'mode': mode}
            weighted_phase_mag_filename = PATHS['output_datadir'] / weighted_phase_mag_tpl.format(**fmt_kwargs)

            task_ctrl.add(Task(plot_phase_mag,
                               {'weighted': weighted_phase_mag_filename, 'raster_cubes': hb_raster_cubes_fn},
                               [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' /
                                mode / f'map_{dataset}.{hb_name}.{mode}.area_weighted.{v}.png'
                                for v in ['phase', 'alpha_phase', 'mag']],
                               func_args=[dataset, hb_name, mode]))

        if basin_scales == 'small_medium_large':
            for name, datasets in zip(['', 'full_'],
                                      (['cmorph', 'u-al508', 'u-ak543'], ['cmorph', 'u-al508', 'u-am754', 'u-ak543'])):
                for mode in PRECIP_MODES:
                    input_paths = {f'weighted_{hb_name}_{dataset}': (PATHS['output_datadir'] /
                                                                     weighted_phase_mag_tpl.format(hb_name=hb_name,
                                                                                                   dataset=dataset,
                                                                                                   mode=mode))
                                   for hb_name, dataset in itertools.product(hb_names, datasets)}
                    input_paths.update({'raster_cubes': hb_raster_cubes_fn})
                    task_ctrl.add(Task(plot_phase_alpha_combined,
                                       input_paths,
                                       [PATHS['figsdir'] / 'basin_weighted_analysis' / 'map' / 'phase_alpha_combined' /
                                        f'{name}{mode}_phase_alpha_combined_asia.pdf'],
                                       func_args=(datasets, hb_names, mode)))

        for area_weighted in [True, False]:
            weighted = 'area_weighted' if area_weighted else 'not_area_weighted'

            vrmse_data_filename = (PATHS['output_datadir'] /
                                   f'basin_weighted_analysis/all_rmses.{weighted}.{basin_scales}.pkl')

            task_ctrl.add(Task(plot_cmorph_vs_all_datasets_phase_mag,
                               [vrmse_data_filename],
                               [PATHS['figsdir'] / 'basin_weighted_analysis' / 'cmorph_vs' / 'phase_mag' /
                                f'cmorph_vs_all_datasets.all_rmse.{weighted}.{basin_scales}.pdf'],
                               )
                          )

    return task_ctrl
