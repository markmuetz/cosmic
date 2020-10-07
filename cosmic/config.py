from pathlib import Path
import socket
import warnings

import iris

ALL_PATHS = {
    'mistakenot': {
        'datadir': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data'),
        'gcosmic': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic'),
        'output_datadir': Path('/home/markmuetz/cosmic_WP2_analysis/data'),
        'hydrosheds_dir': Path('/home/markmuetz/HydroSHEDS'),
        'figsdir': Path('/home/markmuetz/cosmic_WP2_analysis/figs/mistakenot'),
    },
    'zerogravitas': {
        'datadir': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data'),
        'gcosmic': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic'),
        'output_datadir': Path('/home/markmuetz/cosmic_WP2_analysis/data'),
        'hydrosheds_dir': Path('/home/markmuetz/HydroSHEDS'),
        'figsdir': Path('/home/markmuetz/cosmic_WP2_analysis/figs/zerogravitas'),
    },
    'jasmin': {
        'datadir': Path('/gws/nopw/j04/cosmic/mmuetz/data'),
        'gcosmic': Path('/gws/nopw/j04/cosmic'),
        'output_datadir': Path('/gws/nopw/j04/cosmic/mmuetz/data/cosmic_WP2_analysis/new_data'),
        'hydrosheds_dir': Path('/gws/nopw/j04/cosmic/mmuetz/HydroSHEDS'),
        'figsdir': Path('/gws/nopw/j04/cosmic/mmuetz/data/cosmic_WP2_analysis/new_figs'),
    }
}


def _short_hostname():
    hostname = socket.gethostname()
    if hostname.split('.')[1] == 'jasmin':
        return 'jasmin'
    return hostname


hostname = _short_hostname()
if hostname[:4] == 'host':
    hostname = 'jasmin'

if hostname not in ALL_PATHS:
    raise Exception(f'Unknown hostname: {hostname}')

PATHS = ALL_PATHS[hostname]
for k, path in PATHS.items():
    if not path.exists():
        warnings.warn(f'Warning: path missing {k}: {path}')

STANDARD_NAMES = {
    'cmorph': 'CMORPH',
    'u-al508': 'N1280-PC',
    'u-aj399': 'N1280-PCem2',
    'u-az035': 'N1280-PCem3',
    'u-am754': 'N1280-HC',
    'u-ak543': 'N1280-EC',
    'HadGEM3-GC31-HM': 'N512',
    'HadGEM3-GC31-MM': 'N216',
    'HadGEM3-GC31-LM': 'N96',
}

CONSTRAINT_ASIA = (iris.Constraint(coord_values={'latitude': lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude': lambda cell: 56.9 < cell < 151.1}))

CONSTRAINT_CHINA = (iris.Constraint(coord_values={'latitude': lambda cell: 18 < cell < 41})
                    & iris.Constraint(coord_values={'longitude': lambda cell: 97.5 < cell < 125}))

# Based on Malcolm Roberts' request and expanded by 2deg.
CONSTRAINT_EU = (iris.Constraint(coord_values={'latitude': lambda cell: 28 < cell < 67}))
# CANNOT constrain across the GMT boundary (0).
# Use intersection instead.
# & iris.Constraint(coord_values={'longitude': lambda cell: -22 < cell < 37}))


