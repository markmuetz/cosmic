# coding: utf-8
from collections import Counter
from pathlib import Path

import geopandas as gpd
import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basmati.hydrosheds import load_hydrobasins_geodataframe
from remake import Task, TaskControl
from cosmic import util
# from cosmic.task import Task, TaskControl
from paths import PATHS

REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'

FILENAME_TPL = '/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/PRIMAVERA_HighResMIP_MOHC/{model}/' \
               'highresSST-present/r1i1p1f1/E1hr/pr/gn/{timestamp}/' \
               'pr_E1hr_{model}_highresSST-present_r1i1p1f1_gn_{daterange}.nc'

MODELS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
]
TIMESTAMPS = ['v20170906', 'v20170818', 'v20170831']
DATERANGES = ['201401010030-201412302330', '201401010030-201412302330', '201404010030-201406302330']

FILENAMES = {
    model: FILENAME_TPL.format(model=model, timestamp=timestamp, daterange=daterange)
    for model, timestamp, daterange in zip(MODELS, TIMESTAMPS, DATERANGES)
}

CONSTRAINT_ASIA = (iris.Constraint(coord_values={'latitude': lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude': lambda cell: 56.9 < cell < 151.1}))


def gen_raster_cube(inputs, outputs, hb_name, method):
    hb = gpd.read_file(str(inputs['hydrobasins']))
    cube = iris.load_cube(str(inputs['cube']))
    if method == 'global':
        # Build global raster, then extract Asia.
        raster_cube = util.build_raster_cube_from_cube(cube, hb, hb_name)
        raster_cube_asia = raster_cube.extract(CONSTRAINT_ASIA)
    elif method == 'local':
        # extract asia from cube, then build local raster (uses affine_tx under the hood).
        cube_asia = cube.extract(CONSTRAINT_ASIA)
        raster_cube_asia = util.build_raster_cube_from_cube(cube_asia, hb, hb_name)
    iris.save(raster_cube_asia, str(outputs[0]))


def plot_raster_cube(inputs, outputs, hb_name, method):
    hb = gpd.read_file(str(inputs['hydrobasins']))
    raster_cube_asia = iris.load_cube(str(inputs['raster_cube_asia']))

    lat_min = raster_cube_asia.coord('latitude').bounds[0][0]
    lat_max = raster_cube_asia.coord('latitude').bounds[-1][1]
    lon_min = raster_cube_asia.coord('longitude').bounds[0][0]
    lon_max = raster_cube_asia.coord('longitude').bounds[-1][1]

    extent = (lon_min, lon_max, lat_min, lat_max)

    plt.figure(str(outputs[0]), figsize=(10, 8))
    plt.imshow(raster_cube_asia.data, origin='lower', extent=extent)
    ax = plt.gca()
    hb.geometry.boundary.plot(ax=ax, color=None, edgecolor='r')
    plt.xlim((50, 160))
    plt.ylim((0, 60))
    plt.savefig(outputs[0])
    plt.close()


def gen_hydrobasins_files(inputs, outputs, hb_name):
    hb = load_hydrobasins_geodataframe(PATHS['hydrosheds_dir'], 'as', range(1, 9))
    if hb_name == 'small':
        hb_size = hb.area_select(2000, 20000)
    elif hb_name == 'medium':
        hb_size = hb.area_select(20000, 200000)
    elif hb_name == 'large':
        hb_size = hb.area_select(200000, 2000000)
    hb_size.to_file(outputs[0])


def gen_raster_stats(inputs_dict, outputs):
    hbs = {}
    for hb_name in ['small', 'medium', 'large']:
        hb_size = gpd.read_file(str(inputs_dict[('hydrobasins', hb_name)]))
        hbs[hb_name] = hb_size

    output_text = []
    for model in MODELS:
        for hb_name in ['small', 'medium', 'large']:
            hb_size = hbs[hb_name]
            global_raster_filename = inputs_dict[(model, hb_name, 'global')]
            local_raster_filename = inputs_dict[(model, hb_name, 'local')]
            global_raster = iris.load_cube(str(global_raster_filename))
            local_raster = iris.load_cube(str(local_raster_filename))

            cells_per_basin = np.array([(local_raster.data == i + 1).sum() for i in range(len(hb_size))])
            number_with_count = Counter(cells_per_basin)

            output_text.append(f'{model} - {hb_name}')
            output_text.append(f'  - global vs local diff: {(global_raster.data != local_raster.data).sum()}')
            output_text.append(f'  - number_with_counts (first 10): {number_with_count.most_common()[:10]}')
            output_text.append(f'  - mean cells_per_basin: {cells_per_basin.mean()}')
    with outputs[0].open('w') as f:
        f.write('\n'.join(output_text) + '\n')


def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    scales = ['small', 'medium', 'large']
    for hb_name in scales:
        task = Task(gen_hydrobasins_files, [], [PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.{ext}'
                                                for ext in ['shp', 'dbf', 'prj', 'cpg', 'shx']],
                    func_args=[hb_name])
        task_ctrl.add(task)

    index_data = []

    for model in MODELS:
        for hb_name in scales:
            cube_path = FILENAMES[model]
            for method in ['global', 'local']:
                raster_path = (PATHS['output_datadir'] /
                               'raster_vs_hydrobasins' /
                               f'raster_{model}_{hb_name}_{method}.nc')
                fig_path = (PATHS['figsdir'] /
                            'raster_vs_hydrobasins' / model /
                            f'raster_{model}_{hb_name}_{method}.png')
                inputs = {'hydrobasins': PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.shp',
                          'cube': cube_path}
                task = Task(gen_raster_cube, inputs, [raster_path],
                            func_kwargs={'hb_name': hb_name, 'method': method})
                task_ctrl.add(task)
                index_data.append({'model': model, 'hb_name': hb_name, 'method': method, 'task': task})

                inputs = {'hydrobasins': PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.shp',
                          'raster_cube_asia': raster_path}
                task_ctrl.add(Task(plot_raster_cube, inputs, [fig_path],
                                   func_kwargs={'hb_name': hb_name, 'method': method}))

    input_filenames = {}
    index = pd.DataFrame(index_data)

    for model in MODELS:
        for hb_name in scales:
            diff_methods = index[(index.model == model) & (index.hb_name == hb_name)]
            global_raster_task = diff_methods[diff_methods.method == 'global'].task.values[0]
            local_raster_task = diff_methods[diff_methods.method == 'local'].task.values[0]
            input_filenames[(model, hb_name, 'global')] = global_raster_task.outputs[0]
            input_filenames[(model, hb_name, 'local')] = local_raster_task.outputs[0]

    for hb_name in scales:
        input_filenames[('hydrobasins', hb_name)] = PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.shp'

    output_path = (PATHS['figsdir'] / 'raster_vs_hydrobasins' / f'raster_stats.txt')
    task = Task(gen_raster_stats, input_filenames, [output_path])
    task_ctrl.add(task)
    return task_ctrl


if __name__ == '__main__':
    pass
