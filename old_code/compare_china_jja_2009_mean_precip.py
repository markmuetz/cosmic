from argparse import ArgumentParser
import itertools

from cosmic.WP2.compare_china_jja_2009_mean_precip import compare_mean_precip
from config import PATHS


def all_compares_gen(check_calcs=True):
    lowres_datasets = ['gauge_china_2419', 'aphrodite', 'cmorph_0p25']
    hires_datasets = ['cmorph_8km_N1280', 'u-ak543_native', 'u-al508_native', 'u-am754_native']

    for d1, d2 in itertools.combinations(lowres_datasets, 2):
        d1daterange = '200906-200908'
        d2daterange = '200906-200908'
        yield compare_mean_precip, (PATHS['hydrosheds_dir'], PATHS['figsdir'], d1, d2,
                                    d1daterange, d2daterange), {'check_calcs': check_calcs}
        yield compare_mean_precip, (PATHS['hydrosheds_dir'], PATHS['figsdir'], d1, d2,
                                    d1daterange, d2daterange), {'check_calcs': check_calcs, 'plot_type': 'heatmap'}
    for d1, d2 in itertools.combinations(hires_datasets, 2):
        d2daterange = '200906-200908'
        if d1[:2] == 'u-':
            d1dateranges = ['200806-200808']
        else:
            d1dateranges = [f'{y}06-{y}08' for y in range(1998, 2019)]

        if d2[:2] == 'u-':
            d2daterange = '200806-200808'
        for d1daterange in d1dateranges:
            yield compare_mean_precip, (PATHS['hydrosheds_dir'], PATHS['figsdir'], d1, d2,
                                        d1daterange, d2daterange), {'check_calcs': check_calcs, 'land_only': False}
            yield compare_mean_precip, (PATHS['hydrosheds_dir'], PATHS['figsdir'], d1, d2,
                                        d1daterange, d2daterange), {'check_calcs': check_calcs, 'land_only': True}
            yield compare_mean_precip, (PATHS['hydrosheds_dir'], PATHS['figsdir'], d1, d2,
                                        d1daterange, d2daterange), {'check_calcs': check_calcs,
                                                                    'land_only': True,
                                                                    'plot_type': 'heatmap'}


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
