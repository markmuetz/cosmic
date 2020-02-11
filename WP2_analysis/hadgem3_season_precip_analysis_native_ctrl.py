import itertools
from pathlib import Path

from cosmic.task import Task

import iris
from iris.experimental.equalise_cubes import equalise_attributes

import cosmic.WP2.seasonal_precip_analysis as spa

SCRIPT_PATH = 'hadgem3_season_precip_analysis_native.py'

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/')
MODELS = {
    'HadGEM3-GC31-HM': {
        'input_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-HM',
    },
    'HadGEM3-GC31-MM': {
        'input_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-MM',
    },
    'HadGEM3-GC31-LM': {
        'input_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-LM',
    },
}
PRECIP_THRESHES = [0.05, 0.1, 0.2]
# SEASONS = ['jja', 'son', 'djf', 'mam']
SEASONS = ['JJA']

START_YEAR = 2005
END_YEAR = 2009

BSUB_KWARGS = {
    'job_name': 'hg_spa',
    'queue': 'short-serial',
    'max_runtime': '02:00',
    'mem': '64000',
}


def task_fn(input_filenames, output_filenames, precip_thresh, season):
    output_filename, done_filename = output_filenames

    season_cubes = iris.load([str(fn) for fn in input_filenames])
    equalise_attributes(season_cubes)
    season_cube = season_cubes.concatenate_cube()
    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh)

    iris.save(analysis_cubes, str(output_filename))
    done_filename.touch()


CONFIG_KEYS = []
SCRIPT_ARGS = {}
for model, precip_thresh, season in itertools.product(MODELS.keys(), PRECIP_THRESHES, SEASONS):
    key = f'{model}_{season}_{precip_thresh}'
    input_dir = MODELS[model]['input_dir']
    input_filenames = sorted([fn for fn in
                              input_dir.glob(f'{model}.highresSST-present.r1i1p1f1.????.{season}.asia_precip.nc')
                              if START_YEAR <= int(fn.stem.split('.')[3]) <= END_YEAR])
    thresh_text = str(precip_thresh).replace('.', 'p')
    output_filename = input_dir / f'{model}.highresSST-present.r1i1p1f1.{START_YEAR}-{END_YEAR}.' \
                                  f'{season}.asia_precip.ppt_thresh_{thresh_text}.nc'
    done_filename = (output_filename.parent / (output_filename.name + '.done'))
    task = Task(task_fn, input_filenames, [output_filename, done_filename], fn_args=[precip_thresh, season])

    if task.requires_rerun():
        CONFIG_KEYS.append(key)
        SCRIPT_ARGS[key] = task
