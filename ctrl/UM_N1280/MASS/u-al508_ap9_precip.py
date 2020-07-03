import sys
sys.path.insert(0, '.')
from common import AP9_TOTAL_RAIN_SNOW_ONLY, BASE_OUTPUT_DIRPATH

AP9_TOTAL_RAIN_SNOW_ONLY['start_year_month'] = (2008, 4)
AP9_TOTAL_RAIN_SNOW_ONLY['end_year_month'] = (2009, 12)

ACTIVE_RUNIDS = ['u-al508']

MASS_INFO = {}
MASS_INFO['u-al508'] = {
    'stream': {
        'ap9': AP9_TOTAL_RAIN_SNOW_ONLY,
    },
}
