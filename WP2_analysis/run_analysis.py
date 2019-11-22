from argparse import ArgumentParser
from pathlib import Path
import hashlib
import warnings

from afi_figs_all import afi_all_figs_gen
from extract_china_jja_2009_mean_precip import extract_all_dataset_gen
from compare_china_jja_2009_mean_precip import all_compares_gen
from plot_gauge_data import all_plot_gauge_data_gen
from plot_seasonal_analysis import all_seasonal_analysis_gen

fn_generators = [
    extract_all_dataset_gen,
    afi_all_figs_gen,
    all_compares_gen,
    all_plot_gauge_data_gen,
    all_seasonal_analysis_gen,
]


def task_hash(fn, args, kwargs):
    # return hashlib.sha1(repr((fn.__name__, fn.__code__.co_code, args, kwargs)).encode()).hexdigest()
    return hashlib.sha1(repr((fn.__name__, args, kwargs)).encode()).hexdigest()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--figures', default='ALL', choices=['ALL'] + [gen.__name__ for gen in fn_generators])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    cli_args = parser.parse_args()

    task_cache = Path('.run_analysis/task_cache')
    task_cache.mkdir(exist_ok=True, parents=True)
    if cli_args.clear_cache:
        for filepath in task_cache.glob('*'):
            filepath.unlink()

    tasks = []
    for gen in fn_generators:
        if 'ALL' in cli_args.figures or gen.__name__ in cli_args.figures:
            tasks.extend(list(gen()))

    hashes = set()
    cached = []
    success = []
    fail = []
    for i, (fn, args, kwargs) in enumerate(tasks):
        print(f'{i + 1}/{len(tasks)}: {fn.__name__}, {args}, {kwargs}')
        task_hash_key = task_hash(fn, args, kwargs)
        assert task_hash_key not in hashes
        hashes.add(task_hash_key)
        task_cache_file = task_cache / task_hash_key
        if task_cache_file.exists() and not cli_args.force:
            cached.append((fn, args, kwargs))
            continue
        try:
            fn(*args, **kwargs)
            success.append((fn, args, kwargs))
            task_cache_file.touch()
        except Exception as e:
            warnings.warn(f'Could not run {fn.__name__}, {args}, {kwargs}:')
            fail.append((fn, args, kwargs))
            print(e)

    print()
    print(f'Cached: {len(cached)}')
    print(f'Succeeded: {len(success)}')
    print(f'Failed: {len(fail)}')
