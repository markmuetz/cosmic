import itertools
from pathlib import Path

SCRIPT_PATH = 'hadgem3_regrid.py'

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/')
TARGET_FILENAME = Path('/gws/nopw/j04/cosmic/mmuetz/data/u-ak543/ap9.pp/'
                       'ak543a.p9jja.200502-200901.asia_precip.ppt_thresh_0p1.nc')

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
