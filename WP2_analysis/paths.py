from pathlib import Path
import socket
import warnings

ALL_PATHS = {
    'mistakenot': {
        'datadir': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data'),
        'hydrosheds_dir': Path('/home/markmuetz/HydroSHEDS'),
        'figsdir': Path('figs/mistakenot'),
    },
    'zerogravitas': {
        'datadir': Path('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data'),
        'hydrosheds_dir': Path('/home/markmuetz/HydroSHEDS'),
        'figsdir': Path('figs/zerogravitas'),
    },
    'jasmin': {
        'datadir': Path('/gws/nopw/j04/cosmic/mmuetz/data'),
        'hydrosheds_dir': Path('/gws/nopw/j04/cosmic/mmuetz/HydroSHEDS'),
        'figsdir': Path('figs/jasmin'),
    }
}


def _short_hostname():
    hostname = socket.gethostname()
    if hostname[:6] == 'jasmin':
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
