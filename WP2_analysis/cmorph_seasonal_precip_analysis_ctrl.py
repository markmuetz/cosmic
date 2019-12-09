import itertools
from pathlib import Path

from cmorph_seasonal_precip_analysis import fmt_year_month


SCRIPT_PATH = 'cmorph_seasonal_precip_analysis.py'

# SEASONS = ['jja', 'son', 'djf', 'mam']
# SEASONS = ['jja']
# PRECIP_THRESHES = [0.05, 0.1, 0.2]
SEASONS = ['jja']
PRECIP_THRESHES = [0.1]

CMORPH_DATASET = '8km-30min'
RESOLUTION = 'N1280'
# CMORPH_DATASET = '0.25deg-3HLY'
# RESOLUTION = None
# START_YEAR_MONTH = (1998, 1)
# END_YEAR_MONTH = (2018, 12)
# START_YEAR_MONTH = (2009, 6)
# END_YEAR_MONTH = (2009, 8)
DATERANGES = [((year, 6), (year, 8)) for year in range(1998, 2019)]

BSUB_KWARGS = {
    'job_name': 'cmorph_spa',
    'queue': 'short-serial',
    'max_runtime': '20:00',
    'mem': '32000',
}

CONFIG_KEYS = []
SCRIPT_ARGS = {}

all_options = itertools.product(DATERANGES, SEASONS, PRECIP_THRESHES)
for (start_year_month, end_year_month), season, precip_thresh in all_options:
    key = (fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
           + f'_{season}_{precip_thresh}')
    CONFIG_KEYS.append(key)
    SCRIPT_ARGS[key] = (start_year_month, end_year_month, season, precip_thresh)

