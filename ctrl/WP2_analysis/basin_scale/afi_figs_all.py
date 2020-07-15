import itertools

import iris

from cosmic.WP2.afi_base import AFI_basePlotter
from cosmic.WP2.afi_mean_plot import AFI_meanPlotter
from cosmic.WP2.afi_diurnal_cycle_plot import AFI_diurnalCyclePlotter

from remake import TaskControl, Task, remake_required, remake_task_control

from cosmic.config import PATHS
from seasonal_precip_analysis import fmt_thresh_text


MODES = ['amount', 'freq', 'intensity']


@remake_required(depends_on=[AFI_meanPlotter, AFI_basePlotter])
def fig_afi_mean(inputs, outputs, season, region, method, runids):
    afi_mean = AFI_meanPlotter(runids, season, region, method)
    cubes = {}
    for (runid, cube_name), cube_path in inputs.items():
        cubes[(runid, cube_name)] = iris.load_cube(str(cube_path), cube_name)
    afi_mean.set_cubes(cubes)
    afi_mean.plot()
    afi_mean.save(outputs[0])


@remake_required(depends_on=[AFI_diurnalCyclePlotter, AFI_basePlotter])
def fig_afi_diurnal_cycle(inputs, outputs, season, region, method, runids):
    afi_mean = AFI_diurnalCyclePlotter(runids, season, region, method)
    cubes = {}
    for (runid, cube_name), cube_path in inputs.items():
        cubes[(runid, cube_name)] = iris.load_cube(str(cube_path), cube_name)
    afi_mean.set_cubes(cubes)
    afi_mean.plot()
    afi_mean.save(outputs[0])


class AfiTask(Task):
    def __init__(self, func, runids, datadir, figsdir, duration, precip_thresh, season, region, method=None):
        thresh_text = fmt_thresh_text(precip_thresh)

        # N.B. applies to nc file that is read in.
        domain = region
        if domain == 'china':
            domain = 'asia'
        inputs = {}

        for runid in runids:
            if runid == 'cmorph':
                if duration == 'short':
                    daterange = '200906-200908'
                elif duration == 'long':
                    daterange = '199801-201812'
                else:
                    daterange = duration
                rel_path = 'cmorph_data/8km-30min'
                filename = f'cmorph_8km_N1280.{daterange}.{season}.{domain}_precip_afi.ppt_thresh_{thresh_text}.nc'
                # filename = f'cmorph_ppt_{season}.{daterange}.{region}_precip.ppt_thresh_{thresh_text}.N1280.nc'
            else:
                if duration == 'short':
                    daterange = '200806-200808'
                elif duration == 'long':
                    daterange = '200506-200808'
                else:
                    daterange = '200506-200808'

                rel_path = f'{runid}/ap9.pp'
                # filename = f'{runid}a.p9{season}.{daterange}.{region}_precip.ppt_thresh_{thresh_text}.nc'
                filename = f'{runid[2:]}.{daterange}.{season}.{domain}_precip_afi.ppt_thresh_{thresh_text}.nc'

            for mode in MODES:
                inputs[(runid, f'{mode}_of_precip_{season}')] = datadir / rel_path / filename

        output_path = (figsdir / 'AFI' / '_'.join(runids) /
                       f'{func.__name__}.{duration}.{season}.{region}.{method}.ppt_thresh_{thresh_text}.pdf')
        super().__init__(func, inputs, [output_path], func_args=(season, region, method, runids))


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    # Can only do 1 or 3 runids ATM.
    all_runids = [['cmorph', 'u-al508', 'u-ak543'],
                  ['cmorph'],
                  ['cmorph', 'u-am754', 'u-ak543'],
                  ['u-al508', 'u-am754', 'u-ak543'],
                  ['u-al508', 'u-aj399', 'u-az035']]
    season = 'jja'
    durations = ['long']
    for start_year in range(1998, 2016):
        durations.append(f'{start_year}06-{start_year + 3}08')
    precip_threshes = [0.1]
    methods = ['peak', 'harmonic']
    # regions = ['china', 'asia', 'europe']
    regions = ['china', 'asia']

    # Run all durations for first runid.
    task_data = list(itertools.product([all_runids[0]], durations, precip_threshes, regions))
    # Run first duration for all other runids.
    task_data.extend(itertools.product(all_runids[1:], [durations[0]], precip_threshes, regions))

    for runids, duration, precip_thresh, region in task_data:
        task = AfiTask(fig_afi_mean, runids, PATHS['datadir'], PATHS['figsdir'],
                       duration, precip_thresh, season, region)
        task_ctrl.add(task)
        for method in methods:
            task = AfiTask(fig_afi_diurnal_cycle, runids, PATHS['datadir'], PATHS['figsdir'],
                           duration, precip_thresh, season, region, method)
            task_ctrl.add(task)

    return task_ctrl
