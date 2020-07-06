from pathlib import Path

RUNID = 'u-az035'
STREAM = 'ap9'

SCRIPT_PATH = '/home/users/mmuetz/projects/cosmic/cosmic/processing/convert_pp_to_nc.py'
BASE_PATH = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/{RUNID}/{STREAM}.pp')

paths = sorted(BASE_PATH.glob('precip_??????/*.pp'))
# paths = sorted(BASE_PATH.glob('precip_200506/*.pp'))
CONFIG_KEYS = [p.stem for p in paths]

BSUB_KWARGS = {
    'job_name': 'conv',
    'queue': 'short-serial',
    'max_runtime': '05:00',
}

IRIS_CUBE_ATTRS = {
    'grid': 'N1280',
    'institution': 'Met Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK',
    'institution_id': 'MOHC',
    'source_type': 'AGCM',
    'model': 'u-az035',
    'experiment_details': 'convection parametrization scheme',
}

DIAGTYPE = 'precip'
DELETE_PP = True

SCRIPT_ARGS = {}
for k, path in zip(CONFIG_KEYS, paths):
    SCRIPT_ARGS[k] = path
