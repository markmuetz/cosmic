from itertools import product

import headless_matplotlib  # uses 'agg' backend if HEADLESS env var set.
import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cosmic import util
from cosmic.plotting_util import configure_ax_asia
from remake import Task, TaskControl, remake_task_control, remake_required
from orog_precip_paths import (orog_path, surf_wind_path_tpl, precip_path_tpl,
                               raw_data_dc_fig_tpl, raw_data_fig_tpl, fmtp)


def get_region_constraint(region):
    if region == 'TP_southern_flank':
        return (iris.Constraint(coord_values={'latitude': lambda cell: 20 < cell < 30})
                & iris.Constraint(coord_values={'longitude': lambda cell: 86 < cell < 100}))


def dc(cube, num_per_day):
    # Superfast DC calc. Needs lots of mem though.
    assert cube.shape[0] % num_per_day == 0, 'Cube has wrong time dimension'
    num_days = cube.shape[0] // num_per_day

    dc = (cube.data
          .reshape(num_days, num_per_day, cube.shape[-2], cube.shape[-1])
          .mean(axis=0))
    return dc


@remake_required(depends_on=[get_region_constraint, configure_ax_asia])
def plot_precip_wind_region(inputs, outputs, region):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    surf_wind = iris.load(str(inputs['surf_wind'])).concatenate()
    precip = iris.load_cube(str(inputs['precip']))

    region_constraint = get_region_constraint(region)

    orog_region = orog.extract(region_constraint)
    surf_wind_region = surf_wind.extract(region_constraint)
    precip_region = precip.extract(region_constraint)
    u_region = surf_wind_region.extract_strict('x_wind')
    v_region = surf_wind_region.extract_strict('y_wind')
    lon = orog_region.coord('longitude').points
    lat = orog_region.coord('latitude').points

    extent = util.get_extent_from_cube(precip_region)
    extent_uv = util.get_extent_from_cube(u_region)
    lon_uv = u_region.coord('longitude').points
    lat_uv = u_region.coord('latitude').points

    precip_vmax = precip_region.data.max()

    for i, output in enumerate(outputs):
        print(i)
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches(10, 7.5)

        configure_ax_asia(ax1, extent=extent)
        configure_ax_asia(ax2, extent=extent_uv)

        im = ax1.imshow(precip_region.data[i],
                        origin='lower', extent=extent, vmax=precip_vmax)
        ax1.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000])
        ax2.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000])
        im = ax2.quiver(lon_uv[::3], lat_uv[::3], u_region.data[i][::3, ::3], v_region.data[i][::3, ::3])
        # plt.colorbar(im, orientation='horizontal', label='precip (??)', pad=0.1)
        plt.savefig(output)
        plt.close('all')


@remake_required(depends_on=[get_region_constraint, configure_ax_asia])
def plot_dc_region(inputs, outputs, region, num_per_day):
    orog = iris.load_cube(str(inputs['orog']), 'surface_altitude')
    surf_wind = iris.load(str(inputs['surf_wind'])).concatenate()
    precip = iris.load_cube(str(inputs['precip']))

    region_constraint = get_region_constraint(region)

    orog_region = orog.extract(region_constraint)
    surf_wind_region = surf_wind.extract(region_constraint)
    precip_region = precip.extract(region_constraint)
    u_region = surf_wind_region.extract_strict('x_wind')
    v_region = surf_wind_region.extract_strict('y_wind')
    lon = orog_region.coord('longitude').points
    lat = orog_region.coord('latitude').points

    extent = util.get_extent_from_cube(precip_region)
    extent_uv = util.get_extent_from_cube(u_region)
    lon_uv = u_region.coord('longitude').points
    lat_uv = u_region.coord('latitude').points

    dc_u = dc(u_region, num_per_day)
    dc_v = dc(v_region, num_per_day)
    dc_precip = dc(precip_region, num_per_day)

    precip_vmax = dc_precip.max()

    for i, output in enumerate(outputs):
        print(i)
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches(10, 7.5)

        configure_ax_asia(ax1, extent=extent)
        configure_ax_asia(ax2, extent=extent_uv)

        im = ax1.imshow(dc_precip[i],
                        origin='lower', extent=extent, vmax=precip_vmax)
        ax1.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000])
        ax2.contour(lon, lat, orog_region.data, [500, 1000, 2000, 3000])
        im = ax2.quiver(lon_uv[::3], lat_uv[::3], dc_u[i][::3, ::3], dc_v[i][::3, ::3])
        # plt.colorbar(im, orientation='horizontal', label='precip (??)', pad=0.1)
        plt.savefig(output)
        plt.close('all')


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    models = ['al508', 'ak543']
    months = [6, 7, 8]
    year = 2006
    region = 'TP_southern_flank'

    if True:
        model = models[0]
        month = months[0]
    # for model, month in product(models, months):
        surf_wind_path = fmtp(surf_wind_path_tpl, model=model, year=year, month=month)
        precip_path = fmtp(precip_path_tpl, model=model, year=year, month=month)

        raw_data_fig_paths = [fmtp(raw_data_fig_tpl, model=model, year=year, month=month,
                                      day=day, hour=h, region=region)
                              for day in range(1, 31)
                              for h in range(0, 24)]
        tc.add(Task(plot_precip_wind_region,
                    {'orog': orog_path, 'surf_wind': surf_wind_path, 'precip': precip_path},
                    raw_data_fig_paths,
                    func_args=(region, )))

        raw_data_dc_fig_paths = [fmtp(raw_data_dc_fig_tpl, model=model, year=year, month=month,
                                      hour=h, region=region)
                              for h in range(24)]
        tc.add(Task(plot_dc_region,
                    {'orog': orog_path, 'surf_wind': surf_wind_path, 'precip': precip_path},
                    raw_data_dc_fig_paths,
                    func_args=(region, 24)))

    return tc

