from pathlib import Path

RUNID = 'u-ak543'
STREAM = 'ap9'

SCRIPT_PATH = '/home/users/mmuetz/projects/cosmic/cosmic/processing/extract_region.py'
BASE_PATH = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{RUNID}/{STREAM}.pp')

# Only do JJA.
paths = sorted(list(BASE_PATH.glob('precip_????06')) +
               list(BASE_PATH.glob('precip_????07')) +
               list(BASE_PATH.glob('precip_????08')))
CONFIG_KEYS = [p.stem for p in paths]

BSUB_KWARGS = {
    'job_name': 'ak543exeu',
    'queue': 'short-serial',
    'max_runtime': '03:30',
}

STRATIFORM = True
SCRIPT_ARGS = {}
for k, path in zip(CONFIG_KEYS, paths):
    SCRIPT_ARGS[k] = ('europe', path)
