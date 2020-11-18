import itertools
from pathlib import Path

SCRIPT_PATH = 'cmorph_extract_asia_8km_30min.py'

BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data/8km-30min')

# years = [1999]
# months = [4]
# years = range(1998, 2019)
# months = range(1, 13)
# years_months = list(itertools.product(years, months))
years_months = [(2001, 7),
                (2003, 7),
                (2004, 3),
                (2004, 12),
                (2006, 10),
                (2009, 8),
                (2014, 5),
                (2015, 10)]

CONFIG_KEYS = [f'{y}{m:02}' for y, m in years_months]

BSUB_KWARGS = {
    'job_name': 'cmorph_ea',
    'queue': 'short-serial',
    'max_runtime': '01:30',
    'mem': 32000,
}

SCRIPT_ARGS = {}
for key, (year, month) in zip(CONFIG_KEYS, years_months):
    SCRIPT_ARGS[key] = (year, month)
