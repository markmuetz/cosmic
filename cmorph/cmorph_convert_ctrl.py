from pathlib import Path

SCRIPT_PATH = 'cmorph_convert.py'

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')

years = range(1998, 2020)

CONFIG_KEYS = years

BSUB_KWARGS = {
    'job_name': 'cmorph_cv',
    'queue': 'new_users',
    'max_runtime': '00:30',
}

SCRIPT_ARGS = {}
for year in CONFIG_KEYS:
    SCRIPT_ARGS[str(year)] = year
