"""
Experimental code to test ideas around detecting multiple maxima (multipeak)

Useful to try but probably won't persue much further.
"""
import itertools

import iris
import matplotlib.pyplot as plt

from cosmic.WP2.multipeak import calc_data_maxima, plot_data_maxima_windowed

from paths import PATHS


def multipeak_fig(data_maxima, title, extent, nsteps, nx, ny, china_only):
    plot_data_maxima_windowed(data_maxima, title, extent, nsteps, nx, ny, china_only)
    (PATHS['figsdir'] / 'multipeak').mkdir(exist_ok=True)
    plt.savefig(PATHS['figsdir'] / 'multipeak' / f'{title}.png')


def multipeak_all_figs_gen():
    cmorph_path = (PATHS['datadir'] /
                   'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
    cmorph_amount = iris.load_cube(str(cmorph_path), 'amount_of_precip_jja')
    lat = cmorph_amount.coord('latitude').points
    lon = cmorph_amount.coord('longitude').points
    extent = list(lon[[0, -1]]) + list(lat[[0, -1]])

    data = cmorph_amount.data
    data_maxima_2hr_exclusion = calc_data_maxima(data, 4)
    data_maxima_23hr_exclusion = calc_data_maxima(data, 47)

    exclusions = ['2hr', '23hr']
    regions = ['asia', 'china']
    windows = ['3hr', '1hr']

    for exclusion, region, window in itertools.product(exclusions, regions, windows):
        if exclusion == '2hr':
            data = data_maxima_2hr_exclusion
        elif exclusion == '23hr':
            data = data_maxima_23hr_exclusion

        if window == '3hr':
            nx = 4
            ny = 2
        elif window == '1hr':
            nx = 6
            ny = 4

        title = f'{region}_{exclusion}_exclusion_{window}'
        yield multipeak_fig, (data, title, extent, 48, nx, ny), {'china_only': region == 'china'}
