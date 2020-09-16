import sys
sys.path.insert(0, '.')
from common import AP9_SURF_WIND, BASE_OUTPUT_DIRPATH

# AP9_SURF_WIND['years_months'] = product([2006, 2007, 2008, 2009], [6, 7, 8])
AP9_SURF_WIND['start_year_month'] = (2006, 6)
AP9_SURF_WIND['end_year_month'] = (2006, 8)

ACTIVE_RUNIDS = ['u-ak543']

MASS_INFO = {}
MASS_INFO['u-ak543'] = {
    'stream': {
        'ap9': AP9_SURF_WIND,
    },
}
