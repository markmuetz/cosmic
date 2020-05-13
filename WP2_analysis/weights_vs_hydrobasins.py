import sys
import string
import itertools
import iris
import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd

from remake import Task, TaskControl, remake_task_control
import cosmic.util as util
from cosmic.config import PATHS, CONSTRAINT_ASIA

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


def plot_weights_cube_table(inputs, outputs, rows, cols):
    fig, axes = plt.subplots(3, 4, sharex='row', figsize=(10, 6),
                             gridspec_kw={'height_ratios': [2, 2, 2.7]})
    extents = {
        'small': [107.5, 111.5, 17.8, 20.2],
        'medium': [133, 141, 49.5, 54.5],
        'large': [119, 133, 41, 53],
    }
    ticks = {
        'small': (
            [108, 111], [18, 20]
        ),
        'medium': (
            [134, 140], [50, 54]
        ),
        'large': (
            [120, 132], [42, 52]
        ),
    }
    for axrow, row in zip(axes, rows):
        hb_name, basin_index = row
        hb = gpd.read_file(str(inputs[hb_name]))
        for ax, model in zip(axrow, cols):
            weights_cube = iris.load_cube(str(inputs[(model, hb_name)]))
            basin = hb.loc[basin_index]
            w = weights_cube[basin_index]

            lat_max, lat_min, lon_max, lon_min, nlat, nlon = util.get_latlon_from_cube(weights_cube)
            extent = (lon_min, lon_max, lat_min, lat_max)

            im = ax.imshow(w.data, origin='lower', extent=extent, vmin=0, vmax=1)
            hb[hb.PFAF_ID == basin.PFAF_ID].geometry.boundary.plot(ax=ax, color=None, edgecolor='r')

            extent = extents[hb_name]
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])

            xticks, yticks = ticks[hb_name]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'${t}\\degree$ E' for t in xticks])
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'${t}\\degree$ N' for t in yticks])

    for axrow in axes:
        for i in range(3):
            axrow[i].get_yaxis().set_ticks([])
            axrow[i].get_yaxis().tick_right()

        axrow[-1].get_yaxis().tick_right()

    cax = fig.add_axes([0.12, 0.07, 0.74, 0.02])
    plt.colorbar(im, cax=cax, orientation='horizontal', label='cell weight')

    axes[0, 0].set_title('N96')
    axes[0, 1].set_title('N216')
    axes[0, 2].set_title('N512')
    axes[0, 3].set_title('N1280')

    axes[0, 0].set_ylabel('small')
    axes[0, 0].get_yaxis().set_label_coords(-0.1, 0.5)
    axes[1, 0].set_ylabel('medium')
    axes[1, 0].get_yaxis().set_label_coords(-0.1, 0.5)
    axes[2, 0].set_ylabel('large')
    axes[2, 0].get_yaxis().set_label_coords(-0.1, 0.5)
    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(0.01, 1.04, f'({c})', size=12, transform=ax.transAxes)

    plt.subplots_adjust(left=0.04, right=0.94, top=0.96, bottom=0.15, hspace=0.32)

    plt.savefig(outputs[0])


@remake_task_control
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

        # TODO: Has not necessarily been created (if e.g. just querying task_ctrl).
        # TODO: Also, creating loads of output files creates loads of metadata, which takes up space.
        # TODO: Perhaps zip all the pngs together?
        # hb = gpd.read_file(str(input_filenames['hb_name']))
        # output_filenames = {i: PATHS['figsdir'] / 'weights_vs_hydrobasins' / f'{model}_{hb_name}' / f'basin_{i}.png'
        #                     for i in range(len(hb))}
        # task_ctrl.add(Task(plot_weights_cube, input_filenames, output_filenames))

    input_filenames = {(model, hb_name): (PATHS['output_datadir'] /
                                          f'weights_vs_hydrobasins/weights_{model}_{hb_name}.nc')
                       for model, hb_name in itertools.product(MODELS, HB_NAMES)}
    for hb_name in HB_NAMES:
        input_filenames[hb_name] = PATHS['output_datadir'] / f'raster_vs_hydrobasins/hb_{hb_name}.shp'
    output_filenames = [PATHS['figsdir'] / 'weights_vs_hydrobasins' / 'weights_cube_table' / f'basins_table.pdf']

    task_ctrl.add(Task(plot_weights_cube_table, input_filenames, output_filenames,
                       func_kwargs={'rows': list(zip(HB_NAMES[::-1], [770, 201, 23])),
                                    'cols': MODELS}
                       ))
    return task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl()
    task_ctrl.finalize()
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.run()

