SCRIPT_PATH = 'UM_seasonal_precip_analysis.py'
SPLIT_STREAM = 'a.p9'
LOC = 'asia'

RUNIDS = ['ak543', 'am754', 'al508']
PRECIP_THRESHES = [0.05, 0.1, 0.2]
# SEASONS = ['jja', 'son', 'djf', 'mam']
SEASONS = ['jja']

START_YEAR_MONTH = (2005, 2)
END_YEAR_MONTH = (2009, 1)

BSUB_KWARGS = {
    'job_name': 'UM_spa',
    'queue': 'short-serial',
    'max_runtime': '02:00',
    'mem': '64000',
}

CONFIG_KEYS = []
SCRIPT_ARGS = {}
for runid in RUNIDS:
    for precip_thresh in PRECIP_THRESHES:
        for season in SEASONS:
            key = f'{runid}_{season}_{precip_thresh}'
            CONFIG_KEYS.append(key)
            SCRIPT_ARGS[key] = (runid, precip_thresh, season)
