import sys
from itertools import product

sys.path.insert(0, '.')
from common import AP9_TOTAL_RAIN_SNOW_ONLY, BASE_OUTPUT_DIRPATH

AP9_TOTAL_RAIN_SNOW_ONLY['years_months'] = list(product([2005], [6, 7, 8]))

ACTIVE_RUNIDS = ['u-aj399']

MASS_INFO = {}
MASS_INFO['u-aj399'] = {
    'stream': {
        'ap9': AP9_TOTAL_RAIN_SNOW_ONLY,
    },
}
