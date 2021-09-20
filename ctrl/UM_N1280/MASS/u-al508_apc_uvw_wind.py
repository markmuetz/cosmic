import sys
from itertools import product
sys.path.insert(0, '.')
from common import APC_UVW_WIND, BASE_OUTPUT_DIRPATH

APC_UVW_WIND['years_months'] = list(product([2009], [6, 7, 8]))
# APC_UVW_WIND['years_months'] = list(product([2006], [7]))

ACTIVE_RUNIDS = ['u-al508']

MASS_INFO = {}
MASS_INFO['u-al508'] = {
    'stream': {
        'apc': APC_UVW_WIND
    },
}
