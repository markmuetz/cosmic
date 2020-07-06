from pathlib import Path

RUNID = 'u-aj399'
STREAM = 'ap9'

SCRIPT_PATH = '/home/users/mmuetz/projects/cosmic/cosmic/processing/extract_region.py'
BASE_PATH = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{RUNID}/{STREAM}.pp')

# Only do JJA.
paths = sorted(list(BASE_PATH.glob('precip_????06')) +
               list(BASE_PATH.glob('precip_????07')) +
               list(BASE_PATH.glob('precip_????08')))
CONFIG_KEYS = [p.stem for p in paths]

BSUB_KWARGS = {
    'job_name': 'aj399Xas',
    'queue': 'short-serial',
    'max_runtime': '05:00',
}

STRATIFORM = False
SCRIPT_ARGS = {}
for k, path in zip(CONFIG_KEYS, paths):
    SCRIPT_ARGS[k] = ('asia', path)
