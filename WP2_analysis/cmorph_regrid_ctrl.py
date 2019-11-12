import itertools
SCRIPT_PATH = 'cmorph_regrid.py'

# years_months = list(itertools.product(range(1998, 2019), range(1, 13)))
# Taking ages; restart.
years_months = [(2011, 10)]

CONFIG_KEYS = [f'{y}{m:02}' for y, m in years_months]

BSUB_KWARGS = {
    'job_name': 'cmorph_rg',
    'queue': 'short-serial',
    'max_runtime': '02:00',
    'mem': '32000',
}

SCRIPT_ARGS = {}
for key, (year, month) in zip(CONFIG_KEYS, years_months):
    SCRIPT_ARGS[key] = (year, month)
