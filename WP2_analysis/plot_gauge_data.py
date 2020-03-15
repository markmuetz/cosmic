import headless_matplotlib
import matplotlib.pyplot as plt

from remake import TaskControl, Task, remake_required
from cosmic.WP2 import plot_gauge_data

from paths import PATHS


REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'


@remake_required(depends_on=[plot_gauge_data.plot_li2018_fig2a_reproduction])
def do_plot(inputs, outputs, *args, **kwargs):
    plot_gauge_data.plot_li2018_fig2a_reproduction(*args, **kwargs)
    plt.savefig(outputs[0])


def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    for fn, args, kwargs in all_plot_gauge_data_gen():
        kwargs_str = '.'.join([f'{k}-{i}' for k, i in kwargs.items()])
        task = Task(fn,
                    [],
                    [PATHS['figsdir'] / 'gauge_data' / f'precip_station_jja_cressman.{kwargs_str}.png'],
                    func_args=args,
                    func_kwargs=kwargs)
        task_ctrl.add(task)
    return task_ctrl


def all_plot_gauge_data_gen():
    kwargs = {'stretch_lat': True, 'search_rad': 0.48, 'grid_spacing': 0.2}
    yield (do_plot,
           (PATHS['datadir'], PATHS['hydrosheds_dir']),
           kwargs)

