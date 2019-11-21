from argparse import ArgumentParser
import itertools
from pathlib import Path

import iris
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from cosmic.util import build_raster_from_cube

from basmati.hydrosheds import load_hydrobasins_geodataframe


def compare_mean_precip(dataset1, dataset2, land_only=False):
    plt.close('all')
    cube1 = iris.load_cube(f'data/{dataset1}_china_jja_2009_amount.nc')
    cube2 = iris.load_cube(f'data/{dataset2}_china_jja_2009_amount.nc')

    min_lat1, max_lat1 = cube1.coord('latitude').points[[0, -1]]
    min_lon1, max_lon1 = cube1.coord('longitude').points[[0, -1]]

    min_lat2, max_lat2 = cube2.coord('latitude').points[[0, -1]]
    min_lon2, max_lon2 = cube2.coord('longitude').points[[0, -1]]

    # I'm not going to pretend this isn't confusing.
    # Want the maximum min_lat so that I can extract cubes with same dims.
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

    data1 = cube1.data
    data2 = cube2.data


    if dataset1[:6] == 'cmorph' or dataset1[:2] == 'u-':
        data1 *= 24
    if dataset2[:6] == 'cmorph' or dataset2[:2] == 'u-':
        data2 *= 24

    full_mask = data1.mask | data2.mask
    full_mask |= np.isnan(data1)
    full_mask |= np.isnan(data2)
    if land_only:
        hb = load_hydrobasins_geodataframe('/home/markmuetz/HydroSHEDS', 'as', [1])
        raster = build_raster_from_cube(cube1, hb)
        full_mask |= raster == 0

    data1 = np.ma.masked_array(data1, full_mask).flatten()
    data2 = np.ma.masked_array(data2, full_mask).flatten()
    max_precip = max(data1.max(), data2.max())
    title = f'{dataset1} vs {dataset2} JJA 2009'
    if land_only:
        title += ' land only'
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

    res = linregress(data1.compressed(), data2.compressed())
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dataset1')
    parser.add_argument('--dataset2')
    parser.add_argument('--land-only', action='store_true')
    args = parser.parse_args()

    lowres_datasets = ['gauge_china_2419', 'aphrodite', 'cmorph_0p25']
    hires_datasets = ['cmorph_8km_N1280', 'u-ak543_native', 'u-al508_native', 'u-am754_native']

    if args.all:
        for d1, d2 in itertools.combinations(lowres_datasets, 2):
            print(f'  comparing {d1} vs {d2}')
            compare_mean_precip(d1, d2)
        for d1, d2 in itertools.combinations(hires_datasets, 2):
            print(f'  comparing {d1} vs {d2}')
            compare_mean_precip(d1, d2, land_only=False)
            compare_mean_precip(d1, d2, land_only=True)
    else:
        compare_mean_precip(args.dataset1, args.dataset2, args.land_only)

