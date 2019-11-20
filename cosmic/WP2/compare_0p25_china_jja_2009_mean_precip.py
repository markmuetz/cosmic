from argparse import ArgumentParser
from pathlib import Path

import iris
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def compare_mean_precip(dataset1, dataset2):
    plt.close('all')
    cube1 = iris.load_cube(f'data/{dataset1}_china_jja_2009_amount.nc')
    cube2 = iris.load_cube(f'data/{dataset2}_china_jja_2009_amount.nc')

    data1 = cube1.data
    data2 = cube2.data

    if dataset1 == 'cmorph_0p25':
        data1 *= 24
    if dataset2 == 'cmorph_0p25':
        data2 *= 24

    full_mask = data1.mask | data2.mask
    full_mask |= np.isnan(data1)
    full_mask |= np.isnan(data2)
    data1 = np.ma.masked_array(data1, full_mask).flatten()
    data2 = np.ma.masked_array(data2, full_mask).flatten()
    title = f'{dataset1} vs {dataset2} JJA 2009'
    fig = plt.figure(title, figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title(title)
    ax.scatter(data1, data2)
    ax.plot([0, 20], [0, 20], 'k--')

    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))
    ax.set_xlabel(f'{dataset1} (mm day$^{{-1}}$)')
    ax.set_ylabel(f'{dataset2} (mm day$^{{-1}}$)')

    res = linregress(data1.compressed(), data2.compressed())
    x = np.linspace(0, 20, 2)
    y = res.slope * x + res.intercept
    ax.plot(x, y, 'r--', label=(f'm = {res.slope:.2f}\n'
                                f'c = {res.intercept:.2f}\n'
                                f'r$^2$ = {res.rvalue**2:.2f}\n'
                                f'p = {res.pvalue:.2f}'))
    ax.legend(loc=2)
    plt.savefig(f'figs/compare_jja_2009_{dataset1}_vs_{dataset2}.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dataset1')
    parser.add_argument('--dataset2')
    args = parser.parse_args()

    if args.all:
        compare_mean_precip('gauge_china_2419', 'aphrodite')
        compare_mean_precip('gauge_china_2419', 'cmorph_0p25')
        compare_mean_precip('aphrodite', 'cmorph_0p25')
    else:
        compare_mean_precip(args.dataset1, args.dataset2)

