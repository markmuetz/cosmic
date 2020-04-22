import itertools

import headless_matplotlib
from remake import TaskControl, Task, remake_task_control
from cosmic.WP2 import plot_seasonal_analysis

from config import PATHS
from util import task_unique_name_from_fn_args_kwargs

RUNIDS = [
    'ak543',
    'al508',
    # 'am754',
]
UM_DATERANGE = [
    # '200806-200808',
    '200506-200808',
]

CMORPHS = [
    # 'cmorph_0p25',
    'cmorph_8km',
]
CMORPH_DATERANGE = [
    # '200906-200908',
    '199801-201812',
]
PRECIP_THRESHES = [
    # 0.05,
    0.1,
    # 0.2,
]

HADGEMS = [
    'HadGEM3-GC31-HM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-LM',
]


def do_plot(inputs, outputs, *args):
    plot_seasonal_analysis.main(*args)
    outputs[0].touch()


def all_seasonal_analysis_gen():
    for runid, daterange in itertools.product(RUNIDS, UM_DATERANGE):
        seasons = ['jja']
        resolution = None
        args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                runid, daterange, seasons, resolution, PRECIP_THRESHES)
        yield do_plot, args, {}

    for runid in HADGEMS:
        seasons = ['JJA']
        resolution = 'N1280'
        args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                runid, '2005-2009', seasons, resolution, PRECIP_THRESHES)
        yield do_plot, args, {}

    for cmorph, daterange in itertools.product(CMORPHS, CMORPH_DATERANGE):
        seasons = ['jja']
        resolution = None
        if cmorph == 'cmorph_8km':
            if daterange == '199801-201812':
                args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                        cmorph, daterange, seasons, resolution, [0.1])
                yield do_plot, args, {}

            resolution = 'N1280'
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield do_plot, args, {}
        else:
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield do_plot, args, {}


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    for fn, args, kwargs in all_seasonal_analysis_gen():
        task_unique_name = task_unique_name_from_fn_args_kwargs(fn, args, kwargs)
        task = Task(fn,
                    [plot_seasonal_analysis.__file__],
                    [PATHS['figsdir'] / 'seasonal_analysis' / task_unique_name],
                    func_args=args,
                    func_kwargs=kwargs)
        task_ctrl.add(task)

    return task_ctrl
