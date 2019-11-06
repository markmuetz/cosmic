SCRIPT_PATH = 'seasonal_precip_analysis.py'
RUNID = 'ak543'
SPLIT_STREAM = 'a.p9'
LOC = 'asia'
PRECIP_THRESH = 0.1

CONFIG_KEYS = ['jja', 'son', 'djf', 'mam']

START_YEAR_MONTH = (2005, 2)
END_YEAR_MONTH = (2009, 1)

BSUB_KWARGS = {
    'job_name': 'ak543_spa',
    'queue': 'new_users',
    'max_runtime': '01:00',
    'mem': '64000',
}

SCRIPT_ARGS = {}
for season in CONFIG_KEYS:
    SCRIPT_ARGS[season] = season
