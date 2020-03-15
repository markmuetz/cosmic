import headless_matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import iris

from remake import TaskControl, Task, remake_required
from cosmic.util import load_cmap_data
from config import PATHS, STANDARD_NAMES
from basin_weighted_analysis import _configure_ax_asia, get_dataset_path

REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'

DATASETS = [
    'cmorph',
    'u-al508',
    'u-ak543',
]


@remake_required(depends_on=[_configure_ax_asia])
def plot_gridpoint_mean_precip_asia(inputs, outputs):
    # TODO: Saturated colour scale.
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')
    ppt_cubes = []
    for dataset, path in inputs.items():
        ppt_cube = iris.load_cube(str(path), 'precip_flux_mean')
        assert ppt_cube.units == 'mm hr-1'
        ppt_cubes.append(ppt_cube)

    # TODO: DRY (utils).
    lon = ppt_cube.coord('longitude')
    lat = ppt_cube.coord('latitude')
    if not lat.has_bounds():
        lat.guess_bounds()
    if not lon.has_bounds():
        lon.guess_bounds()
    lon_min, lon_max = lon.bounds[0, 0], lon.bounds[-1, 1]
    lat_min, lat_max = lat.bounds[0, 0], lat.bounds[-1, 1]
    extent = (lon_min, lon_max, lat_min, lat_max)

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 3),
                             subplot_kw=dict(projection=ccrs.PlateCarree()))
    for ax, cube, dataset in zip(axes, ppt_cubes, inputs.keys()):
        ax.set_title(STANDARD_NAMES[dataset])
        # Convert from mm hr-1 to mm day-1
        im = ax.imshow(cube.data * 24, extent=extent, norm=norm, cmap=cmap)
        _configure_ax_asia(ax, tight_layout=False)
    axes[1].get_yaxis().set_ticklabels([])
    axes[2].get_yaxis().set_ticklabels([])

    cax = fig.add_axes([0.12, 0.15, 0.74, 0.02])
    plt.colorbar(im, cax=cax, orientation='horizontal', label='precipitation (mm day$^{-1}$)', **cbar_kwargs)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.98, bottom=0.17)
    plt.savefig(outputs[0])


def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    input_paths = {dataset: get_dataset_path(dataset) for dataset in DATASETS}
    task = Task(plot_gridpoint_mean_precip_asia,
                input_paths,
                [PATHS['figsdir'] / 'gridpoint_analysis' / f'gridpoint_mean_precip_asia.pdf'])
    task_ctrl.add(task)
    return task_ctrl

