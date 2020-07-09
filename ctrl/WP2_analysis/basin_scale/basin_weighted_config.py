import numpy as np

from cosmic.config import PATHS

HADGEM_FILENAME_TPL = 'PRIMAVERA_HighResMIP_MOHC/{model}/' \
                      'highresSST-present/r1i1p1f1/E1hr/pr/gn/{timestamp}/' \
                      'pr_E1hr_{model}_highresSST-present_r1i1p1f1_gn_{daterange}.nc'

HADGEM_MODELS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
]

HADGEM_TIMESTAMPS = ['v20170906', 'v20170818', 'v20170831']
HADGEM_DATERANGES = ['201401010030-201412302330', '201401010030-201412302330', '201404010030-201406302330']

HADGEM_FILENAMES = {
    model: PATHS['datadir'] / HADGEM_FILENAME_TPL.format(model=model, timestamp=timestamp, daterange=daterange)
    for model, timestamp, daterange in zip(HADGEM_MODELS, HADGEM_TIMESTAMPS, HADGEM_DATERANGES)
}

DATASETS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
    'u-ak543',
    'u-am754',
    'u-al508',
    'u-aj399',
    'u-az035',
    'cmorph',
]

DATASET_RESOLUTION = {
    'HadGEM3-GC31-LM': 'N96',
    'HadGEM3-GC31-MM': 'N216',
    'HadGEM3-GC31-HM': 'N512',
    'u-ak543': 'N1280',
    'u-am754': 'N1280',
    'u-al508': 'N1280',
    'u-aj399': 'N1280',
    'u-az035': 'N1280',
    'cmorph': 'N1280',
    'aphrodite': '0.25deg'
}
HB_NAMES = ['small', 'med', 'large']
PRECIP_MODES = ['amount', 'freq', 'intensity']
SCALES = {
    'small': (2_000, 20_000),
    'med': (20_000, 200_000),
    'large': (200_000, 2_000_000),
}
N_SLIDING_SCALES = 11
SLIDING_LOWER = np.exp(np.linspace(np.log(2_000), np.log(200_000), N_SLIDING_SCALES))
SLIDING_UPPER = np.exp(np.linspace(np.log(20_000), np.log(2_000_000), N_SLIDING_SCALES))

SLIDING_SCALES = dict([(f'S{i}', (SLIDING_LOWER[i], SLIDING_UPPER[i])) for i in range(N_SLIDING_SCALES)])



