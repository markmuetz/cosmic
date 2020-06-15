import sys

import cartopy.crs as ccrs
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np

from cosmic import util
from cosmic.util import configure_ax_asia
from cosmic.config import CONSTRAINT_ASIA, PATHS


def plot_orog(orog, grad_orog):
    extent = util.get_extent_from_cube(orog)
    plt.figure('orog')
    plt.imshow(orog.data[1:-1], origin='lower', extent=extent)
    plt.colorbar()
    plt.figure('d orog dx')
    plt.imshow(grad_orog[0].data, origin='lower', extent=extent)
    plt.colorbar()
    plt.figure('d orog dy')
    plt.imshow(grad_orog[1].data, origin='lower', extent=extent)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # Meant to be used in ipython with `run -i ...`
    try:
        print(orog)
    except NameError:
        # Gen setup data.

        orog = iris.load_cube(str(PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.orog'),
                              'surface_altitude')
        grad_orog = util.calc_uniform_lat_lon_grad(orog)

        uplus = orog.copy()
        uplus.data = np.ones((1920, 2560)) * 5
        uplus.units = 'm s-1'
        uplus.rename('u')
        vplus = uplus.copy()
        vplus.rename('v')

        uminus = orog.copy()
        uminus.data = -1 * np.ones((1920, 2560)) * 5
        uminus.units = 'm s-1'
        uminus.rename('u')
        vminus = uminus.copy()
        vminus.rename('v')

        windplus = iris.cube.CubeList([uplus[1:-1, :], vplus[1:-1, :]])
        windminus = iris.cube.CubeList([uminus[1:-1, :], vminus[1:-1, :]])

        dotplus = util.calc_2d_dot_product(windplus, grad_orog)
        dotplus_thresh = dotplus.copy()
        dotplus_thresh.data = dotplus.data > 0.1
        dotplus_thresh_asia = dotplus_thresh.extract(CONSTRAINT_ASIA)

        dotminus = util.calc_2d_dot_product(windminus, grad_orog)
        dotminus_thresh = dotminus.copy()
        dotminus_thresh.data = dotminus.data > 0.1
        dotminus_thresh_asia = dotminus_thresh.extract(CONSTRAINT_ASIA)

        lat = dotplus_thresh.coord('latitude').points
        lon = dotplus_thresh.coord('longitude').points
        Lon, Lat = np.meshgrid(lon, lat)

        lat_asia = dotplus_thresh_asia.coord('latitude').points
        lon_asia = dotplus_thresh_asia.coord('longitude').points
        Lon_asia, Lat_asia = np.meshgrid(lon_asia, lat_asia)

    if sys.argv[1] == 'plot_orog':
        plot_orog(orog, grad_orog)

    if sys.argv[1] == 'test_close_methods':
        plt.ion()

        try:
            print(mask_asia)
        except NameError:
            dist_asia = util.CalcLatLonDistanceMask(Lat_asia, Lon_asia, 100, False)
            mask_asia = dist_asia.calc_close_to_mask(dotplus_thresh_asia)
            mask_asia_acc = util.calc_close_to_mask(dotplus_thresh_asia, 100, 'accurate')

        plt.figure('acc vs new method')
        plt.imshow(mask_asia_acc.data.astype(int) - mask_asia.data.astype(int), origin='lower')

    if sys.argv[1] == 'plot_masks':
        fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
        for axrow, dot, dot_thresh_asia in zip(axes,
                                               [dotplus, dotminus],
                                               [dotplus_thresh_asia, dotminus_thresh_asia]):

            extent = util.get_extent_from_cube(dot_thresh_asia)
            for ax in axrow:
                configure_ax_asia(ax, tight_layout=False)
            ax0, ax1, ax2, ax3 = axrow
            ax0.imshow(dot.extract(CONSTRAINT_ASIA).data, origin='lower', extent=extent)
            ax1.imshow(dot_thresh_asia.data, origin='lower', extent=extent)
            for ax, dist in zip([ax2, ax3], [50, 100]):
                dist_asia = util.CalcLatLonDistanceMask(Lat_asia, Lon_asia, dist, False)
                mask_asia = dist_asia.calc_close_to_mask(dot_thresh_asia)
                ax.imshow(mask_asia.data, origin='lower', extent=extent)

        axes[0, 0].set_title('(u, v) x $\\nabla$z')
        axes[0, 1].set_title('mask on > 0.1 (m s$^{-1}$)')
        axes[0, 2].set_title('expand by 50 km')
        axes[0, 3].set_title('expand by 100 km')

        axes[0, 0].set_ylabel('u, v = 5 m s$^{-1}$')
        axes[1, 0].set_ylabel('u, v = -5 m s$^{-1}$')
        for ax in axes[0, :].flatten():
            ax.get_xaxis().set_ticks([])
        for ax in axes[:, 1:].flatten():
            ax.get_yaxis().set_ticks([])

