from pathlib import Path

RUNID = 'u-ak543'
STREAM = 'ap9'

SCRIPT_PATH = '/gws/nopw/j04/cosmic/mmuetz/projects/cosmic/cosmic/processing/extract_asia.py'
BASE_PATH = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{RUNID}/{STREAM}.pp')

paths = sorted(BASE_PATH.glob('precip_??????'))
CONFIG_KEYS = [p.stem for p in paths]

BSUB_KWARGS = {
    'job_name': 'ex_asia',
    'queue': 'new_users',
    'max_runtime': '00:30',
}

STRATIFORM = True
SCRIPT_ARGS = {}
for k, path in zip(CONFIG_KEYS, paths):
    SCRIPT_ARGS[k] = path
