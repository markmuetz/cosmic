import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from basmati.hydrosheds import load_hydrosheds_dem, load_hydrobasins_geodataframe
from basmati.utils import build_raster_from_geometries
from remake import TaskControl, Task, remake_required, remake_task_control

from cosmic.config import PATHS
from cosmic.plotting_util import configure_ax_asia


@remake_required(depends_on=[configure_ax_asia])
def plot_dem(inputs, outputs):
    hydrosheds_dir = PATHS['hydrosheds_dir']
    bounds, tx, dem, mask = load_hydrosheds_dem(hydrosheds_dir, 'as')
    hb_gdf = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', [1])
    raster = build_raster_from_geometries(hb_gdf.geometry, dem.shape, tx)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    ma_dem = np.ma.masked_array(dem, (mask == -1) | (raster == 0))
    plt.figure(figsize=(5, 3.6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cmap = plt.cm.get_cmap('terrain')
    # Fills masked values.
    cmap.set_bad(color='k', alpha=0.1)
    im = ax.imshow(ma_dem[::-1, :], extent=extent, origin='lower', cmap=cmap)

    configure_ax_asia(ax, tight_layout=False)
    rect = Rectangle((97.5, 18), 125 - 97.5, 41 - 18, linewidth=1.5, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    plt.colorbar(im, orientation='horizontal', label='elevation (m)', pad=0.1)
    plt.subplots_adjust(left=0.11, right=0.94, top=0.95, bottom=0.04)

    plt.savefig(outputs[0])


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    task_ctrl.add(Task(plot_dem, [], [PATHS['figsdir'] / 'dem' / f'dem_asia.pdf']))
    return task_ctrl
