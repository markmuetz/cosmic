from cosmic.WP2.extract_china_jja_2009_mean_precip import extract_dataset, DATASETS

from paths import PATHS


def extract_all_dataset_gen():
    for dataset in DATASETS:
        yield extract_dataset, (PATHS['datadir'], dataset), {}
