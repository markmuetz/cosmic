from pathlib import Path

SCRIPT_PATH = 'cmorph_download.py'

years = [2014, 2018, 2019]

CONFIG_KEYS = years

BSUB_KWARGS = {
    'job_name': 'cmorph_dl',
    'queue': 'new_users',
    'max_runtime': '00:30',
}

SCRIPT_ARGS = {}
for year in CONFIG_KEYS:
    SCRIPT_ARGS[year] = year
