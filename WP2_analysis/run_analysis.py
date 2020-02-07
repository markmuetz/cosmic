import sys
from argparse import ArgumentParser
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt

from cosmic.task import TaskControl, Task

from extract_china_jja_2009_mean_precip import extract_all_dataset_gen
from afi_figs_all import afi_all_figs_gen
from compare_china_jja_2009_mean_precip import all_compares_gen
from plot_gauge_data import all_plot_gauge_data_gen
from plot_seasonal_analysis import all_seasonal_analysis_gen
from plot_aphrodite_seasonal_analysis import all_aphrodite_gen
from cmorph_diurnal_cycle_multipeak import multipeak_all_figs_gen
from basin_diurnal_cycle_analysis import basin_analysis_all

from paths import hostname

fn_generators = [
    extract_all_dataset_gen,
    afi_all_figs_gen,
    all_compares_gen,
    all_plot_gauge_data_gen,
    all_seasonal_analysis_gen,
    all_aphrodite_gen,
    multipeak_all_figs_gen,
    basin_analysis_all,
]


def task_hash(hostname, fn, args, kwargs):
    # return hashlib.sha1(repr((fn.__name__, fn.__code__.co_code, args, kwargs)).encode()).hexdigest()
    return hashlib.sha1(repr((hostname, fn.__name__, args, kwargs)).encode()).hexdigest()


def gen_task_fn(fn):
    """Function factory: uses closure to capture fn and call it.
    """
    def task_fn(inputs, outputs, *args, **kwargs):
        print(f'  Inner fn: {fn}(*{args}, **{kwargs})')
        res = fn(*args, **kwargs)
        outputs[0].touch()
        plt.close('all')
        return res
    return task_fn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--analysis', default='ALL', choices=['ALL'] + [gen.__name__ for gen in fn_generators])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--raise-exceptions', '-X', action='store_true')
    cli_args = parser.parse_args()

    task_dir = Path('data/tasks')
    task_dir.mkdir(exist_ok=True, parents=True)
    if cli_args.clear_cache:
        for filepath in task_dir.glob('*'):
            filepath.unlink()
            sys.exit()

    task_ctrl = TaskControl()
    tasks = []
    for gen in fn_generators:
        if 'ALL' in cli_args.analysis or gen.__name__ in cli_args.analysis:
            for fn, args, kwargs in gen():
                task_hash_key = task_hash(hostname, fn, args, kwargs)
                task_ctrl.add(Task(gen_task_fn(fn), [], [task_dir / task_hash_key],
                                   fn_args=args, fn_kwargs=kwargs))
    task_ctrl.finilize().run(cli_args.force)
