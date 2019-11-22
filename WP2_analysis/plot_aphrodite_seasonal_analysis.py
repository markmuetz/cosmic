from cosmic.WP2.plot_aphrodite_seasonal_analysis import plot_aphrodite_seasonal_analysis

from paths import PATHS


def all_aphrodite_gen():
    yield plot_aphrodite_seasonal_analysis, (PATHS['datadir'], ), {}
