from itertools import product

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (orog_path, surf_wind_path_tpl, precip_path_tpl,
                               raw_data_dc_fig_tpl, raw_data_fig_tpl, orog_mask_path_tpl, fmtp)

def get_region_box(region):
    if region == 'TP_southern_flank':
        return {'lon': [86, 100], 'lat': [20, 30]}
    elif region == 'sichuan_basin':
        return {'lon': [102, 112], 'lat': [26, 34]}


def get_region_constraint(region):
    region_box = get_region_box(region)
    lon = region_box['lon']
    lat = region_box['lat']
    return (iris.Constraint(coord_values={'latitude': lambda cell: lat[0] < cell < lat[1]})
            & iris.Constraint(coord_values={'longitude': lambda cell: lon[0] < cell < lon[1]}))


def dc(cube, num_per_day):
    # Superfast DC calc. Needs lots of mem though.
    assert cube.shape[0] % num_per_day == 0, 'Cube has wrong time dimension'
    num_days = cube.shape[0] // num_per_day

    dc = (cube.data
          .reshape(num_days, num_per_day, cube.shape[-2], cube.shape[-1])
          .mean(axis=0))
    return dc

def get_region_data(inputs, region):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    surf_wind = iris.load(str(inputs['surf_wind'])).concatenate()
    precip = iris.load_cube(str(inputs['precip']))

    # Needed so that has same dims as surf_wind after extract constraint.
    orog.coord('latitude').guess_bounds()
    orog.coord('longitude').guess_bounds()
    precip.coord('latitude').guess_bounds()
    precip.coord('longitude').guess_bounds()

    region_constraint = get_region_constraint(region)

    orog_region = orog.extract(region_constraint)
    surf_wind_region = surf_wind.extract(region_constraint)
    precip_region = precip.extract(region_constraint)
    u_region = surf_wind_region.extract_strict('x_wind')
    v_region = surf_wind_region.extract_strict('y_wind')

    lon = orog_region.coord('longitude').points
    lat = orog_region.coord('latitude').points
    extent = util.get_extent_from_cube(orog_region)
    mid_lon = (lon[0] + lon[-1]) / 2
    lst_offset = mid_lon / 360 * 24
    return (lon, lat, extent, lst_offset, orog_region, precip_region, u_region, v_region)


@remake_required(depends_on=[get_region_data, get_region_constraint, configure_ax_asia])
def plot_precip_wind_region(inputs, outputs, region, month):
    (lon, lat, extent, lst_offset,
     orog_region, precip_region, u_region, v_region) = get_region_data(inputs, region)

    region_constraint = get_region_constraint(region)
    orog_mask_cubes_region = iris.load(str(inputs['orog_mask']), region_constraint)
    dotprod = orog_mask_cubes_region.extract_strict('surf_wind x del orog')
    expanded_mask = orog_mask_cubes_region.extract_strict('expanded surf_wind x del orog > thresh')

    precip_vmax = precip_region.data.max()
    dotprod_vmin = dotprod.data.min()
    dotprod_vmax = dotprod.data.max()
    # dotprod_absmax = np.max([np.abs(dotprod_vmin), np.abs(dotprod_vmax)])
    dotprod_absmax = 0.05

    for i, output in enumerate(outputs):
        print(i)
        h = i % 24
        dom = i // 24 + 1
        fig, axes = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches(10, 8.5)
        lst = (h + lst_offset) % 24
        fig.suptitle(f'{month} {dom} LST: {lst:0.1f}')

        for ax in axes.flatten():
            configure_ax_asia(ax, extent=extent)
            ax.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000, 4000, 5000, 6000],
                       cmap='terrain')

        im = axes[0, 0].imshow(precip_region.data[i],
                               origin='lower', extent=extent, vmax=precip_vmax,
                               norm=mpl.colors.LogNorm())
        im = axes[0, 1].quiver(lon[::3], lat[::3],
                               u_region.data[i][::3, ::3], v_region.data[i][::3, ::3])
        im = axes[1, 0].imshow(dotprod.data[i],
                               origin='lower', extent=extent,
                               vmin=-dotprod_absmax, vmax=dotprod_absmax,
                               cmap='bwr')
        im = axes[1, 1].imshow(expanded_mask.data[i],
                               origin='lower', extent=extent)
        # plt.colorbar(im, orientation='horizontal', label='precip (??)', pad=0.1)
        plt.savefig(output)
        plt.close('all')


@remake_required(depends_on=[get_region_data, get_region_constraint, configure_ax_asia])
def plot_dc_region(inputs, outputs, region, num_per_day):
    (lon, lat, extent, lst_offset,
     orog_region, precip_region, u_region, v_region) = get_region_data(inputs, region)

    region_constraint = get_region_constraint(region)
    orog_mask_cubes_region = iris.load(str(inputs['orog_mask']), region_constraint)
    dotprod = orog_mask_cubes_region.extract_strict('surf_wind x del orog')
    expanded_mask = orog_mask_cubes_region.extract_strict('expanded surf_wind x del orog > thresh')

    dc_u = dc(u_region, num_per_day)
    dc_v = dc(v_region, num_per_day)
    dc_precip = dc(precip_region, num_per_day)
    dc_dotprod = dc(dotprod, num_per_day)
    dc_mask = dc(expanded_mask, num_per_day)

    precip_vmax = dc_precip.max()
    dotprod_absmax = 0.05

    for h, output in enumerate(outputs):
        print(h)
        fig, axes = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})

        fig.set_size_inches(10, 8.5)
        lst = (h + lst_offset) % 24
        fig.suptitle(f'LST: {lst:0.1f}')

        for ax in axes.flatten():
            configure_ax_asia(ax, extent=extent)
            ax.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000, 4000, 5000, 6000],
                       cmap='terrain')

        im = axes[0, 0].imshow(dc_precip[h],
                               origin='lower', extent=extent, vmax=precip_vmax,
                               norm=mpl.colors.LogNorm())
        im = axes[0, 1].quiver(lon[::3], lat[::3], dc_u[h][::3, ::3], dc_v[h][::3, ::3])
        im = axes[1, 0].imshow(dc_dotprod[h],
                               origin='lower', extent=extent,
                               vmin=-dotprod_absmax, vmax=dotprod_absmax,
                               cmap='bwr')
        im = axes[1, 1].imshow(dc_mask[h],
                               origin='lower', extent=extent)
        # plt.colorbar(im, orientation='horizontal', label='precip (??)', pad=0.1)
        plt.savefig(output)
        plt.close('all')


@remake_task_control
def gen_task_ctrl(test=False):
    tc = TaskControl(__file__)
    models = ['al508', 'ak543']
    months = [6, 7, 8]
    year = 2006
    regions = ['TP_southern_flank', 'sichuan_basin']
    days_in_month = 30

    if test:
        models = models[:1]
        months = months[:1]
        days_in_month = 2
        # regions = regions[:1]

    for model, month, region in product(models, months, regions):
        surf_wind_path = fmtp(surf_wind_path_tpl, model=model, year=year, month=month)
        precip_path = fmtp(precip_path_tpl, model=model, year=year, month=month)
        orog_mask_path = fmtp(orog_mask_path_tpl, model=model, year=year, month=month,
                              dotprod_thresh=0.05, dist_thresh=100)


        raw_data_fig_paths = [fmtp(raw_data_fig_tpl, model=model, year=year, month=month,
                                      day=day, hour=h, region=region)
                              for day in range(1, days_in_month + 1)
                              for h in range(0, 24)]
        tc.add(Task(plot_precip_wind_region,
                    {
                        'orog': orog_path,
                        'surf_wind': surf_wind_path,
                        'precip': precip_path,
                        'orog_mask': orog_mask_path,
                    },
                    raw_data_fig_paths,
                    func_args=(region, month)))

        raw_data_dc_fig_paths = [fmtp(raw_data_dc_fig_tpl, model=model, year=year, month=month,
                                      hour=h, region=region)
                              for h in range(24)]
        tc.add(Task(plot_dc_region,
                    {
                        'orog': orog_path,
                        'surf_wind': surf_wind_path,
                        'precip': precip_path,
                        'orog_mask': orog_mask_path,
                    },
                    raw_data_dc_fig_paths,
                    func_args=(region, 24)))

    return tc

