import sys
sys.path.insert(0, '.')
from common import AP9_LS_RAIN_SNOW_ONLY, BASE_OUTPUT_DIRPATH

AP9_LS_RAIN_SNOW_ONLY['start_year_month'] = (2005, 1)
AP9_LS_RAIN_SNOW_ONLY['end_year_month'] = (2009, 1)

ACTIVE_RUNIDS = ['u-ak543']

MASS_INFO = {}
MASS_INFO['u-ak543'] = {
    'stream': {
        'ap9': AP9_LS_RAIN_SNOW_ONLY,
    },
}
