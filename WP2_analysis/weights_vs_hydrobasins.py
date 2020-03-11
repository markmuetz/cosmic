import sys
import itertools
import iris
import headless_matplotlib
import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd

from remake import Task, TaskControl
import cosmic.util as util
from paths import PATHS

REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'

FILENAME_TPL = 'PRIMAVERA_HighResMIP_MOHC/{model}/' \
               'highresSST-present/r1i1p1f1/E1hr/pr/gn/{timestamp}/' \
               'pr_E1hr_{model}_highresSST-present_r1i1p1f1_gn_{daterange}.nc'

HADGEM_MODELS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
]
MODELS = HADGEM_MODELS + ['u-ak543']

TIMESTAMPS = ['v20170906', 'v20170818', 'v20170831']
DATERANGES = ['201401010030-201412302330', '201401010030-201412302330', '201404010030-201406302330']

FILENAMES = {
    model: PATHS['datadir'] / FILENAME_TPL.format(model=model, timestamp=timestamp, daterange=daterange)
    for model, timestamp, daterange in zip(HADGEM_MODELS, TIMESTAMPS, DATERANGES)
}
FILENAMES['u-ak543'] = PATHS['datadir'] / 'u-ak543/ap9.pp/precip_200601/ak543a.p9200601.asia_precip.nc'


HB_NAMES = ['large', 'medium', 'small']

CONSTRAINT_ASIA = (iris.Constraint(coord_values={'latitude': lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude': lambda cell: 56.9 < cell < 151.1}))


def gen_weights_cube(inputs, outputs, hb_name):
    cube = iris.load_cube(str(inputs['model']), constraint=CONSTRAINT_ASIA)
    hb = gpd.read_file(str(inputs['hb_name_shp']))
    weights_cube = util.build_weights_cube_from_cube(cube, hb, f'weights_{hb_name}')
    # Cubes are very sparse. You can get a 800x improvement in file size using zlib!
    # BUT I think it takes a lot longer to read them. Leave uncompressed.
    # iris.save(weights_cube, str(outputs[0]), zlib=True)
    iris.save(weights_cube, str(outputs[0]))


def plot_weights_cube(inputs, outputs):
    model, hb_name = inputs.keys()
    hb = gpd.read_file(str(inputs[hb_name]))
    weights_cube = iris.load_cube(str(inputs[model]))
    lat_max, lat_min, lon_max, lon_min, nlat, nlon = util.get_latlon_from_cube(weights_cube)
    dlat = (lat_max - lat_min) / nlat
    dlon = (lon_max - lon_min) / nlon

    for i, (w, basin, output_path) in enumerate(zip(weights_cube.slices_over('basin_index'),
                                                    [r for i, r in hb.iterrows()],
                                                    outputs.values())):
        plt.figure()
        if (w.data == 0).all():
            plt.savefig(output_path)
            output_path.touch()
            continue

        (min_lat_index, max_lat_index), (min_lon_index, max_lon_index) = [(min(v), max(v))
                                                                          for v in np.where(w.data != 0)]
        min_lat, max_lat = weights_cube.coord('latitude').points[[min_lat_index, max_lat_index]]
        min_lon, max_lon = weights_cube.coord('longitude').points[[min_lon_index, max_lon_index]]
        plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max), vmin=0, vmax=1)
        ax = plt.gca()
        hb[hb.PFAF_ID == basin.PFAF_ID].geometry.boundary.plot(ax=ax, color=None, edgecolor='r')
        plt.xlim(min_lon - dlon, max_lon + dlon)
        plt.ylim(min_lat - dlat, max_lat + dlat)
        plt.savefig(output_path)
        plt.close('all')


def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    for model, hb_name in itertools.product(MODELS, HB_NAMES):
        hb_names = {f'hb_name_{ext}': PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.{ext}'
                    for ext in ['shp', 'dbf', 'prj', 'cpg', 'shx']}
        input_filenames = {'model': FILENAMES[model]}
        input_filenames.update(hb_names)
        weights_filename = PATHS['output_datadir'] / f'weights_vs_hydrobasins/weights_{model}_{hb_name}.nc'
        task_ctrl.add(Task(gen_weights_cube, input_filenames, [weights_filename], func_args=(hb_name,)))
        input_filenames = {'model': weights_filename,
                           'hb_name': PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.shp'}

        hb = gpd.read_file(str(input_filenames['hb_name']))
        output_filenames = {i: PATHS['figsdir'] / 'weights_vs_hydrobasins' / f'{model}_{hb_name}' / f'basin_{i}.png'
                            for i in range(len(hb))}
        task_ctrl.add(Task(plot_weights_cube, input_filenames, output_filenames))
    return task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl()
    task_ctrl.finalize()
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.run()

