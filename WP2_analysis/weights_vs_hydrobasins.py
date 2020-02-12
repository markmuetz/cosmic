import itertools
import geopandas as gpd
import iris
import matplotlib.pyplot as plt


from cosmic.task import Task, TaskControl
import cosmic.util as util
from paths import PATHS

FILENAME_TPL = 'PRIMAVERA_HighResMIP_MOHC/{model}/' \
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
    model: PATHS['datadir'] / FILENAME_TPL.format(model=model, timestamp=timestamp, daterange=daterange)
    for model, timestamp, daterange in zip(MODELS, TIMESTAMPS, DATERANGES)
}

HB_NAMES = ['large', 'med', 'small']

CONSTRAINT_ASIA = (iris.Constraint(coord_values={'latitude': lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude': lambda cell: 56.9 < cell < 151.1}))


def gen_weights_cube(inputs, outputs):
    model, hb_name = inputs.keys()
    cube = iris.load_cube(str(inputs[model]), constraint=CONSTRAINT_ASIA)
    hb = gpd.read_file(str(inputs[hb_name]))
    weights_cube = util.build_weights_cube_from_cube(cube, hb, f'weights_{hb_name}')
    iris.save(weights_cube, str(outputs[0]))


def plot_weights_cube(inputs, outputs):
    model, hb_name = inputs.keys()
    hb = gpd.read_file(str(inputs[hb_name]))
    weights_cube = iris.load_cube(str(inputs[model]))
    lat_max, lat_min, lon_max, lon_min, nlat, nlon = util.get_latlon_from_cube(weights_cube)

    for i, (w, basin) in enumerate(zip(weights_cube.slices_over('basin_index'), [r for i, r in hb.iterrows()])):
        plt.figure()
        plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max))
        ax = plt.gca()
        hb[hb.PFAF_ID == basin.PFAF_ID].geometry.boundary.plot(ax=ax, color=None, edgecolor='r')
        plt.xlim(basin.geometry.bounds[0], basin.geometry.bounds[2])
        plt.ylim(basin.geometry.bounds[1], basin.geometry.bounds[3])
        plt.savefig(outputs[i])
        plt.close()


def gen_task_ctrl():
    task_ctrl = TaskControl()
    for model, hb_name in itertools.product(MODELS, HB_NAMES):
        input_filenames = {'model': FILENAMES[model], 'hb_name': f'data/raster_vs_hydrobasins/hb_{hb_name}.shp'}
        weights_filename = f'data/weights_vs_hydrobasins/weights_{model}_{hb_name}.nc'
        task_ctrl.add(Task(gen_weights_cube, input_filenames, [weights_filename]))
        input_filenames = {'model': weights_filename, 'hb_name': f'data/raster_vs_hydrobasins/hb_{hb_name}.shp'}

        hb = gpd.read_file(str(input_filenames['hb_name']))
        output_filenames = {i: PATHS['figsdir'] / 'weights_vs_hydrobasins' / f'{model}_{hb_name}' / f'basin_{i}.png'
                            for i in range(len(hb))}
        task_ctrl.add(Task(plot_weights_cube, input_filenames, output_filenames))
    return task_ctrl


task_ctrl = gen_task_ctrl()


if __name__ == '__main__':
    task_ctrl.finalize().run()

