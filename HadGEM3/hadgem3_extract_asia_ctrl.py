import itertools
from pathlib import Path

SCRIPT_PATH = 'hadgem3_extract_asia.py'

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/')

MODELS = {
    'HadGEM3-GC31-HM': {
        'datadir':
            BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831',
        'output_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-HM',
    },
    'HadGEM3-GC31-MM': {
        'datadir':
            BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-MM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170818',
        'output_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-MM',
    },
    'HadGEM3-GC31-LM': {
        'datadir':
            BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-LM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170906',
        'output_dir': BASEDIR / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-LM',
    },
}

years = range(2005, 2010)
seasons = ['JJA']
models_years_seasons = list(itertools.product(MODELS.keys(), years, seasons))

CONFIG_KEYS = [f'{model}{y}{s}' for model, y, s in models_years_seasons]

BSUB_KWARGS = {
    'job_name': 'hg3_ea',
    'queue': 'short-serial',
    'max_runtime': '01:30',
    'mem': 32000,
}

SCRIPT_ARGS = {}
for key, (model, year, season) in zip(CONFIG_KEYS, models_years_seasons):
    SCRIPT_ARGS[key] = (model, year, season)
