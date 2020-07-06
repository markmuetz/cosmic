import sys
from itertools import product

sys.path.insert(0, '.')
from common import AP9_TOTAL_RAIN_SNOW_ONLY, BASE_OUTPUT_DIRPATH

AP9_TOTAL_RAIN_SNOW_ONLY['years_months'] = list(product([2006, 2007, 2008], [6, 7, 8]))

ACTIVE_RUNIDS = ['u-aj981']

MASS_INFO = {}
MASS_INFO['u-aj981'] = {
    'stream': {
        'ap9': AP9_TOTAL_RAIN_SNOW_ONLY,
    },
}
