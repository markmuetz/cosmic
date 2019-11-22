import iris
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import linregress

from cosmic.util import build_raster_from_cube

from basmati.hydrosheds import load_hydrobasins_geodataframe


def compare_mean_precip(hydrosheds_dir, dataset1, dataset2, land_only=False, check_calcs=False):
    cube1 = iris.load_cube(f'data/{dataset1}_china_jja_2009_amount.nc')
    cube2 = iris.load_cube(f'data/{dataset2}_china_jja_2009_amount.nc')

    min_lat1, max_lat1 = cube1.coord('latitude').points[[0, -1]]
    min_lon1, max_lon1 = cube1.coord('longitude').points[[0, -1]]

    min_lat2, max_lat2 = cube2.coord('latitude').points[[0, -1]]
    min_lon2, max_lon2 = cube2.coord('longitude').points[[0, -1]]

    # I'm not going to pretend this isn't confusing.
    # Want the maximum min_lat so that I can extract cubes with same dims, because
    # e.g. CMORPH N1280 has one fewer coord in some dims that native N1280 runs.
    min_lat = max(min_lat1, min_lat2)
    max_lat = min(max_lat1, max_lat2)

    min_lon = max(min_lon1, min_lon2)
    max_lon = min(max_lon1, max_lon2)

    lat_fn = lambda cell: min_lat <= cell <= max_lat
    lon_fn = lambda cell: min_lon < cell < max_lon

    constraint = (iris.Constraint(coord_values={'latitude': lat_fn})
                  & iris.Constraint(coord_values={'longitude':lon_fn}))

    cube1 = cube1.extract(constraint)
    cube2 = cube2.extract(constraint)

    assert np.all(cube1.coord('latitude').points == cube2.coord('latitude').points)
    assert np.all(cube1.coord('longitude').points == cube2.coord('longitude').points)

    raw_data1 = cube1.data
    raw_data2 = cube2.data

    # Convert these from mm hr-1 to mm day-1
    if dataset1[:6] == 'cmorph' or dataset1[:2] == 'u-':
        raw_data1 *= 24
    if dataset2[:6] == 'cmorph' or dataset2[:2] == 'u-':
        raw_data2 *= 24

    full_mask = raw_data1.mask | raw_data2.mask
    full_mask |= np.isnan(raw_data1)
    full_mask |= np.isnan(raw_data2)
    if land_only:
        hb = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', [1])
        raster = build_raster_from_cube(cube1, hb)
        full_mask |= raster == 0

    title = f'{dataset1} vs {dataset2} JJA 2009'
    if land_only:
        title += ' land only'
    # compressed removes masked values (identically for each array as mask the same).
    data1 = np.ma.masked_array(raw_data1, full_mask).flatten().compressed()
    data2 = np.ma.masked_array(raw_data2, full_mask).flatten().compressed()

    max_precip = max(data1.max(), data2.max())

    if check_calcs:
        extent = [min_lon, max_lon, min_lat, max_lat]
        fig = plt.figure('check-calc: ' + title, figsize=(10, 5))

        ax1 = fig.add_subplot(121, aspect='equal')
        ax2 = fig.add_subplot(122, aspect='equal')
        ax1.set_title(dataset1)
        ax2.set_title(dataset2)
        ax1.imshow(np.ma.masked_array(raw_data1, full_mask), origin='lower',
                   extent=extent, vmax=max_precip, norm=LogNorm())
        im = ax2.imshow(np.ma.masked_array(raw_data2, full_mask), origin='lower',
                        extent=extent, vmax=max_precip, norm=LogNorm())
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.15, 0.11, 0.7, 0.03])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='precip (mm day$^{-1}$)')
        if land_only:
            plt.savefig(f'figs/compare_check_calcs_jja_2009_{dataset1}_vs_{dataset2}.land_only.png')
        else:
            plt.savefig(f'figs/compare_check_calcs_jja_2009_{dataset1}_vs_{dataset2}.png')

    fig = plt.figure(title, figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title(title)
    ax.scatter(data1, data2)
    ax.plot([0, max_precip], [0, max_precip], 'k--')

    ax.set_xlim((0, max_precip))
    ax.set_ylim((0, max_precip))
    ax.set_xlabel(f'{dataset1} (mm day$^{{-1}}$)')
    ax.set_ylabel(f'{dataset2} (mm day$^{{-1}}$)')

    res = linregress(data1, data2)
    x = np.linspace(0, max_precip, 2)
    y = res.slope * x + res.intercept
    ax.plot(x, y, 'r--', label=(f'm = {res.slope:.2f}\n'
                                f'c = {res.intercept:.2f}\n'
                                f'r$^2$ = {res.rvalue**2:.2f}\n'
                                f'p = {res.pvalue:.2f}'))
    ax.legend(loc=2)
    if land_only:
        plt.savefig(f'figs/compare_jja_2009_{dataset1}_vs_{dataset2}.land_only.png')
    else:
        plt.savefig(f'figs/compare_jja_2009_{dataset1}_vs_{dataset2}.png')
    plt.close(fig)
