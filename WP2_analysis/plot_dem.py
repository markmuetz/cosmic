import headless_matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from basmati.hydrosheds import load_hydrosheds_dem, load_hydrobasins_geodataframe
from basmati.utils import build_raster_from_geometries
from remake import TaskControl, Task

from paths import PATHS
from basin_weighted_analysis import _configure_ax_asia


REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'


def plot_dem(inputs, outputs, title):
    hydrosheds_dir = PATHS['hydrosheds_dir']
    bounds, tx, dem, mask = load_hydrosheds_dem(hydrosheds_dir, 'as')
    hb_gdf = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', [1])
    raster = build_raster_from_geometries(hb_gdf.geometry, dem.shape, tx)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    ma_dem = np.ma.masked_array(dem, (mask == -1) | (raster == 0))
    plt.figure(figsize=(10, 7.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    grey_fill = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=int)
    grey_fill[raster == 0] = (200, 200, 200)
    ax.imshow(grey_fill[::-1], extent=extent)
    im = ax.imshow(ma_dem[::-1, :], extent=extent, origin='lower', cmap='terrain')

    _configure_ax_asia(ax, tight_layout=False)
    rect = Rectangle((97.5, 18), 125 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    plt.colorbar(im, orientation='horizontal', label='elevation (m)', pad=0.1)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.0)

    plt.savefig(outputs[0])


def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    title = 'DEM Asia at 30 s resolution (1 / 120 deg)'
    task = Task(plot_dem,
                [],
                [PATHS['figsdir'] / 'dem' / f'dem_asia.pdf'], func_args=(title,))
    task_ctrl.add(task)
    return task_ctrl
