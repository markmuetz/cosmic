from cosmic.WP2.extract_china_jja_2009_mean_precip import extract_dataset, DATASETS

from paths import PATHS


def extract_all_dataset_gen():
    for dataset in DATASETS:
        if dataset[:2] == 'u-':
            dataset += '_native'
        yield extract_dataset, (PATHS['datadir'], dataset), {}
