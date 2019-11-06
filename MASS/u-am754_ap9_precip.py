import sys
sys.path.insert(0, '.')
from common import AP9_TOTAL_RAIN_SNOW_ONLY, BASE_OUTPUT_DIRPATH

AP9_TOTAL_RAIN_SNOW_ONLY['start_year_month'] = (2008, 1)
AP9_TOTAL_RAIN_SNOW_ONLY['end_year_month'] = (2009, 6)

ACTIVE_RUNIDS = ['u-am754']

MASS_INFO = {}
MASS_INFO['u-am754'] = {
    'stream': {
        'ap9': AP9_TOTAL_RAIN_SNOW_ONLY,
    },
}
