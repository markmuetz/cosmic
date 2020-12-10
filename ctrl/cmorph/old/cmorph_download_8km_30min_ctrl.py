from pathlib import Path

SCRIPT_PATH = 'cmorph_download_8km_30min.py'

years = range(2002, 2003)

CONFIG_KEYS = years

BSUB_KWARGS = {
    'job_name': 'cmorph_dl',
    'queue': 'short-serial',
    'max_runtime': '01:30',
}

SCRIPT_ARGS = {}
for year in CONFIG_KEYS:
    SCRIPT_ARGS[str(year)] = year
