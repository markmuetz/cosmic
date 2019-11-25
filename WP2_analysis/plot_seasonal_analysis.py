import itertools

from cosmic.WP2.plot_seasonal_analysis import main

from paths import PATHS

RUNIDS = [
    'ak543',
    'al508',
    'am754',
]
UM_DATERANGE = [
    '200806-200808',
    '200502-200901',
]

CMORPHS = [
    'cmorph_0p25',
    'cmorph_8km',
]
CMORPH_DATERANGE = [
    '200906-200908',
    '199801-201812',
]
PRECIP_THRESHES = [
    0.05,
    0.1,
    0.2,
]


def all_seasonal_analysis_gen():
    for runid, daterange in itertools.product(RUNIDS, UM_DATERANGE):
        seasons = ['jja']
        resolution = None
        args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                runid, daterange, seasons, resolution, PRECIP_THRESHES)
        yield main, args, {}

    for cmorph, daterange in itertools.product(CMORPHS, CMORPH_DATERANGE):
        seasons = ['jja']
        resolution = None
        if cmorph == 'cmorph_8km':
            if daterange == '199801-201812':
                args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                        cmorph, daterange, seasons, resolution, [0.1])
                yield main, args, {}

            resolution = 'N1280'
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield main, args, {}
        else:
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield main, args, {}
