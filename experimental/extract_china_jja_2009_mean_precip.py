from cosmic.WP2.extract_china_jja_2009_mean_precip import extract_dataset, DATASETS

from cosmic.config import PATHS


def extract_all_dataset_gen():
    for dataset, dateranges in DATASETS.items():
        if dataset[:2] == 'u-':
            dataset += '_native'
        for daterange in dateranges:
            yield extract_dataset, (PATHS['datadir'], dataset, daterange), {}
