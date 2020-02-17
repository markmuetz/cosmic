import sys
import itertools

import geopandas as gpd
import iris
import numpy as np
import pandas as pd

from cosmic.task import TaskControl, Task
from paths import PATHS


def gen_vector_basin_stats(inputs, outputs, hb_names):
    columns = ['basin_scale', 'num_basins', 'total_area', 'avg_area', ]
    output = []
    for hb_name in hb_names:
        basin_vector_filename = inputs[f'basin_vector_{hb_name}']
        hb = gpd.read_file(str(basin_vector_filename))

        output.append([hb_name, len(hb), hb.SUB_AREA.sum(), hb.SUB_AREA.mean()])
    df = pd.DataFrame(output, columns=columns)
    df.to_csv(outputs[0])


def gen_weights_basin_stats(inputs, outputs, resolutions, hb_names):
    columns = ['resolution', 'basin_scale', 'avg_full_cells', 'avg_sum_cells']
    output = []
    for res, hb_name in itertools.product(resolutions, hb_names):
        print(f'  {res} - {hb_name}')
        basin_weights_filename = inputs[f'basin_weights_{res}_{hb_name}']
        weights_cube = iris.load_cube(str(basin_weights_filename))
        full_cells = []
        sum_cells = []
        # Stream data; uses a lot less mem.
        for i in range(weights_cube.shape[0]):
            basin_weight = weights_cube[i].data
            full_cells.append((basin_weight == 1).sum())
            sum_cells.append(basin_weight.sum())
        output.append([res, hb_name, np.mean(full_cells), np.mean(sum_cells)])

    df = pd.DataFrame(output, columns=columns)
    df.to_csv(outputs[0])


def gen_task_ctrl():
    task_ctrl = TaskControl()
    basin_scales = 'sliding'

    if basin_scales == 'small_medium_large':
        pass
        # hb_names = HB_NAMES
    else:
        hb_names = [f'S{i}' for i in range(11)]

    inputs = {f'basin_vector_{hb_name}': f'data/basin_weighted_diurnal_cycle/hb_{hb_name}.shp'
              for hb_name in hb_names}

    task_ctrl.add(Task(gen_vector_basin_stats,
                       inputs,
                       [PATHS['figsdir'] / 'basin_stats' / 'basin_vector_stats.csv'],
                       fn_args=(hb_names,),
                       ))

    resolutions = ['N1280', 'N512', 'N216', 'N96']
    inputs = {f'basin_weights_{res}_{hb_name}': f'data/basin_weighted_diurnal_cycle/weights_{res}_{hb_name}.nc'
              for res in resolutions
              for hb_name in hb_names}

    task_ctrl.add(Task(gen_weights_basin_stats,
                       inputs,
                       [PATHS['figsdir'] / 'basin_stats' / 'basin_weights_stats.csv'],
                       fn_args=(resolutions, hb_names),
                       ))

    return task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl()
    task_ctrl.finalize()
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.run()
