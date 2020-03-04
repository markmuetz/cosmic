import itertools

from remake import TaskControl, Task
from cosmic.WP2 import plot_seasonal_analysis

from paths import PATHS

REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'

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


def all_seasonal_analysis_gen():
    for runid, daterange in itertools.product(RUNIDS, UM_DATERANGE):
        seasons = ['jja']
        resolution = None
        args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                runid, daterange, seasons, resolution, PRECIP_THRESHES)
        yield plot_seasonal_analysis.main, args, {}

    for runid in HADGEMS:
        seasons = ['JJA']
        resolution = 'N1280'
        args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                runid, '2005-2009', seasons, resolution, PRECIP_THRESHES)
        yield plot_seasonal_analysis.main, args, {}

    for cmorph, daterange in itertools.product(CMORPHS, CMORPH_DATERANGE):
        seasons = ['jja']
        resolution = None
        if cmorph == 'cmorph_8km':
            if daterange == '199801-201812':
                args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                        cmorph, daterange, seasons, resolution, [0.1])
                yield plot_seasonal_analysis.main, args, {}

            resolution = 'N1280'
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield plot_seasonal_analysis.main, args, {}
        else:
            args = (PATHS['datadir'], PATHS['hydrosheds_dir'], PATHS['figsdir'],
                    cmorph, daterange, seasons, resolution, PRECIP_THRESHES)
            yield plot_seasonal_analysis.main, args, {}


def gen_task_ctrl():
    task_ctrl = TaskControl(enable_file_task_content_checks=True, dotremake_dir='.remake.plot_seasonal_analysis')
    # This depends on the file cosmic/plot_seasonal_analysis.py -- any changes to this will require rerun.
    task = Task(main, [plot_seasonal_analysis.__file__], [PATHS['figsdir'] / 'plot_seasonal_analysis' / 'task.out'])
    task_ctrl.add(task)
    return task_ctrl


def main(inputs, outputs):
    for fn, args, kwargs in all_seasonal_analysis_gen():
        fn(*args, **kwargs)
    outputs[0].touch()


if __name__ == '__main__':
    main(None, None)
