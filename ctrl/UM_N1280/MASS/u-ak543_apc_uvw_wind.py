import sys
from itertools import product
sys.path.insert(0, '.')
from common import APC_UVW_WIND, BASE_OUTPUT_DIRPATH

APC_UVW_WIND['years_months'] = list(product([2005, 2006, 2007, 2008], [6, 7, 8]))
# APC_UVW_WIND['years_months'] = list(product([2006], [7]))

ACTIVE_RUNIDS = ['u-ak543']

MASS_INFO = {}
MASS_INFO['u-ak543'] = {
    'stream': {
        'apc': APC_UVW_WIND
    },
}
