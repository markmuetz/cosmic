import sys
sys.path.insert(0, '.')
from common import AP8_LOWLEVEL_WIND, BASE_OUTPUT_DIRPATH

AP8_LOWLEVEL_WIND['start_year_month'] = (2006, 6)
AP8_LOWLEVEL_WIND['end_year_month'] = (2006, 8)

ACTIVE_RUNIDS = ['u-al508']

MASS_INFO = {}
MASS_INFO['u-al508'] = {
    'stream': {
        'ap8': AP8_LOWLEVEL_WIND,
    },
}
