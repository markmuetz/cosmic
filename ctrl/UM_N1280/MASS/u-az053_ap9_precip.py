import sys
from itertools import product

sys.path.insert(0, '.')
from common import AP9_TOTAL_PPT_ONLY, BASE_OUTPUT_DIRPATH

AP9_TOTAL_PPT_ONLY['years_months'] = list(product([2005, 2006, 2007, 2008], [6, 7, 8]))

ACTIVE_RUNIDS = ['u-az035']

MASS_INFO = {}
MASS_INFO['u-az035'] = {
    'stream': {
        'ap9': AP9_TOTAL_PPT_ONLY,
    },
}
