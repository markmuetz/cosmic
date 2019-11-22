from argparse import ArgumentParser
import itertools

from cosmic.WP2.compare_china_jja_2009_mean_precip import compare_mean_precip
from paths import PATHS


def all_compares_gen(check_calcs=True):
    lowres_datasets = ['gauge_china_2419', 'aphrodite', 'cmorph_0p25']
    hires_datasets = ['cmorph_8km_N1280', 'u-ak543_native', 'u-al508_native', 'u-am754_native']

    for d1, d2 in itertools.combinations(lowres_datasets, 2):
        yield compare_mean_precip, (PATHS['hydrosheds_dir'], d1, d2), {'check_calcs': check_calcs}
    for d1, d2 in itertools.combinations(hires_datasets, 2):
        yield compare_mean_precip, (PATHS['hydrosheds_dir'], d1, d2), {'check_calcs': check_calcs, 'land_only': False}
        yield compare_mean_precip, (PATHS['hydrosheds_dir'], d1, d2), {'check_calcs': check_calcs, 'land_only': True}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dataset1')
    parser.add_argument('--dataset2')
    parser.add_argument('--land-only', action='store_true')
    parser.add_argument('--check-calcs', action='store_true')
    args = parser.parse_args()

    if args.all:
        for fn, args, kwargs in all_compares_gen(args.check_calcs):
            fn(*args, **kwargs)
    else:
        compare_mean_precip(args.dataset1, args.dataset2, args.land_only, check_calcs=args.check_calcs)
