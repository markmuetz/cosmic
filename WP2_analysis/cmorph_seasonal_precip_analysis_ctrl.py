from pathlib import Path

SCRIPT_PATH = 'cmorph_seasonal_precip_analysis.py'

# SEASONS = ['jja', 'son', 'djf', 'mam']
SEASONS = ['jja']
PRECIP_THRESHES = [0.05, 0.1, 0.2]

# CMORPH_DATASET = '8km-30min'
# RESOLUTION = 'N1280'
CMORPH_DATASET = '0.25deg-3HLY'
RESOLUTION = None
# START_YEAR_MONTH = (1998, 1)
# END_YEAR_MONTH = (2018, 12)
START_YEAR_MONTH = (2009, 6)
END_YEAR_MONTH = (2009, 8)

BSUB_KWARGS = {
    'job_name': 'cmorph_spa',
    'queue': 'short-serial',
    'max_runtime': '20:00',
    'mem': '32000',
}

CONFIG_KEYS = []
SCRIPT_ARGS = {}
for season in SEASONS:
    for precip_thresh in PRECIP_THRESHES:
        key = f'{season}_{precip_thresh}'
        CONFIG_KEYS.append(key)
        SCRIPT_ARGS[key] = (season, precip_thresh)
