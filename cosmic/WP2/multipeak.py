"""
Experimental code to calculate local maxima, and display the locations as a function of time in a grid.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def calc_data_maxima(data, order):
    """Calculate the local maxima using an exclusion of order to avoid multiple close maxima.

    :param data: 3D numpy array (axis 0: time)
    :param order: number of points to consider for maxima
    :return: 3D numpy bool array with same shape as data, True if maximum
    """
    argrelmax = signal.argrelmax(data, order=order, mode='wrap')
    dmean = data.mean(axis=0)
    data_maxima = np.zeros(data.shape, dtype=bool)
    for i, j, k in zip(*argrelmax):
        if data[i, j, k] > dmean[j, k]:
            data_maxima[i, j, k] = True
    return data_maxima


def plot_data_maxima_windowed(data_maxima, title, extent, nsteps=48, nx=4, ny=2, china_only=False):
    """Plot the maxima, with suitable defaults for CMORPH.

    :param data_maxima: as returned by calc_data_maxima
    :param title: title to use
    :param extent: extent of data
    :param nsteps: nsteps per day
    :param nx: number of axes in x-dir
    :param ny: number of axes in y-dir
    :param china_only: limit to China (rough) coords
    :return:
    """
    assert nsteps // (nx * ny) == nsteps / (nx * ny)
    data_maxima_window = data_maxima.reshape(-1, nsteps // (nx * ny), 588, 669).sum(axis=1)
    timestep = 24 // (nx * ny)
    # toffset of central China, GMT+7.
    toffset = 7

    fig, axes = plt.subplots(ny, nx, num=title, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches(12, 8)
    for i in range(nx * ny):
        ax = axes.flatten()[i]
        ax.set_title(f'{(i * timestep + toffset) % 24} - {((i + 1) * timestep + toffset) % 24}')
        ax.imshow(data_maxima_window[i], origin='lower', extent=extent, vmax=1)
        if china_only:
            ax.set_xlim((98, 124))
            ax.set_ylim((18, 42))
        ax.coastlines()
