from pathlib import Path

SCRIPT_PATH = 'cmorph_seasonal_precip_analysis.py'

SEASONS = ['jja', 'son', 'djf', 'mam']
PRECIP_THRESHES = [0.2, 0.4, 0.8, 1.6]

START_YEAR_MONTH = (1998, 1)
END_YEAR_MONTH = (2018, 12)

BSUB_KWARGS = {
    'job_name': 'cmorph_spa',
    'queue': 'new_users',
    'max_runtime': '02:00',
    'mem': '64000',
}

CONFIG_KEYS = []
SCRIPT_ARGS = {}
for season in SEASONS:
    for precip_thresh in PRECIP_THRESHES:
        key = f'{season}_{precip_thresh}'
        CONFIG_KEYS.append(key)
        SCRIPT_ARGS[key] = (season, precip_thresh)
