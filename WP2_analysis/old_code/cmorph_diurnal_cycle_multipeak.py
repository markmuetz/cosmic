"""
Experimental code to test ideas around detecting multiple maxima (multipeak)

Useful to try but probably won't persue much further.
"""
import itertools

import iris
import matplotlib.pyplot as plt

from cosmic.WP2.multipeak import calc_data_maxima, plot_data_maxima_windowed

from paths import PATHS


def multipeak_fig(title, exclusion, window, nsteps,china_only):
    cmorph_path = (PATHS['datadir'] /
                   'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
    cmorph_amount = iris.load_cube(str(cmorph_path), 'amount_of_precip_jja')
    lat = cmorph_amount.coord('latitude').points
    lon = cmorph_amount.coord('longitude').points
    extent = list(lon[[0, -1]]) + list(lat[[0, -1]])

    data = cmorph_amount.data
    if exclusion == '2hr':
        data = calc_data_maxima(data, 4)
    elif exclusion == '23hr':
        data = calc_data_maxima(data, 47)

    if window == '3hr':
        nx = 4
        ny = 2
    elif window == '1hr':
        nx = 6
        ny = 4

    plot_data_maxima_windowed(data, title, extent, nsteps, nx, ny, china_only)
    (PATHS['figsdir'] / 'multipeak').mkdir(exist_ok=True)
    plt.savefig(PATHS['figsdir'] / 'multipeak' / f'{title}.png')


def multipeak_all_figs_gen():
    exclusions = ['2hr', '23hr']
    regions = ['asia', 'china']
    windows = ['3hr', '1hr']

    for exclusion, region, window in itertools.product(exclusions, regions, windows):
        title = f'{region}_{exclusion}_exclusion_{window}'
        yield multipeak_fig, (title, exclusion, window, 48), {'china_only': region == 'china'}
