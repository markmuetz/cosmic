from argparse import ArgumentParser
from pathlib import Path

from cosmic.WP2.plot_gauge_data import plot_li2018_fig2a_reproduction

from paths import PATHS


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--stretch-lat', action='store_true')
    parser.add_argument('--search-rad', type=float, default=0.48)
    parser.add_argument('--grid-spacing', type=float, default=0.2)
    return parser.parse_args()


def all_plot_gauge_data_gen():
    kwargs = {'stretch_lat': True, 'search_rad': 0.48, 'grid_spacing': 0.2}
    yield plot_li2018_fig2a_reproduction, (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir']), kwargs


if __name__ == '__main__':
    args = get_args()
    kwargs = vars(args)
    basedir = Path(kwargs.pop('basedir'))
    hydrosheds_dir = Path(kwargs.pop('hydrosheds_dir'))

    plot_li2018_fig2a_reproduction(basedir, hydrosheds_dir, **kwargs)

