from pathlib import Path

SCRIPT_PATH = 'cmorph_seasonal_precip_analysis.py'

SEASONS = ['jja', 'son', 'djf', 'mam']
PRECIP_THRESHES = [0.1]

CMORPH_DATASET = '8km-30min'
# CMORPH_DATASET = '0.25deg-3HRLY'
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
