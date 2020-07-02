from pathlib import Path

from remake import TaskControl, Task, remake_task_control
from cosmic.WP2.plot_aphrodite_seasonal_analysis import plot_aphrodite_seasonal_analysis

from cosmic.config import PATHS


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    aphrodite_dir = Path('aphrodite_data/025deg')
    inputs = {
        ' daily precipitation analysis interpolated onto 0.25deg grids': (PATHS['datadir'] /
                                                                          aphrodite_dir /
                                                                          'APHRO_MA_025deg_V1901.2009.nc')}
    outputs = {
        'asia': PATHS['figsdir'] / 'aphrodite' / 'asia_aphrodite_2009_jja.png',
        'china': PATHS['figsdir'] / 'aphrodite' / 'china_aphrodite_2009_jja.png',
    }

    task_ctrl.add(Task(plot_aphrodite_seasonal_analysis, inputs, outputs, func_args=(True, )))

    outputs = {
        'asia': PATHS['figsdir'] / 'aphrodite' / 'asia_aphrodite_2009_jja.lognorm.png',
        'china': PATHS['figsdir'] / 'aphrodite' / 'china_aphrodite_2009_jja.lognorm.png',
    }

    task_ctrl.add(Task(plot_aphrodite_seasonal_analysis, inputs, outputs, func_args=(False, )))
    return task_ctrl
