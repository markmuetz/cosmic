import itertools
from pathlib import Path

SCRIPT_PATH = 'hadgem3_season_precip_analysis.py'

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

CONFIG_KEYS = []
SCRIPT_ARGS = {}
for model, precip_thresh, season in itertools.product(MODELS.keys(), PRECIP_THRESHES, SEASONS):
    key = f'{model}_{season}_{precip_thresh}'
    CONFIG_KEYS.append(key)
    SCRIPT_ARGS[key] = (model, precip_thresh, season)
