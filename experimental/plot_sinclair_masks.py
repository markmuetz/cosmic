# coding: utf-8
import iris
from iris import plot as iplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from remake import Task, TaskControl, remake_task_control
from cosmic.config import PATHS

BASEDIR = '/home/markmuetz/mirrors/jasmin/gw_primavera/cache/bvanniere/' \
          'primavera/Orographic_precipitation/ERA_interim_orographic_precipitation_daily'


def plot_masks(inputs, outputs):
    r_clim = iris.load_cube(str(inputs[0]))
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(15, 8))

    for i, ax in enumerate(axes.flatten()):
        ax.set_title(i + 1)
        ax.imshow(r_clim[i].data > 0.5, extent=(0, 360, -90, 90))

    plt.savefig(outputs[0])


def distance(lon1, lon2, lat1, lat2):
    """Taken from B. Vanniere."""
    R = 6371.
    dlon = (lon2 - lon1)
    dlon = (dlon + 180.) % 360. - 180.
    x = dlon * np.cos(0.5 * (lat2 + lat1) * np.pi / 180.)
    y = lat2 - lat1
    d = np.pi / 180. * R * np.sqrt(x * x + y * y)
    return d


def distance_orography(orog, KK, HH):
    """ This function calculate if a given point is less than KK km away from orography
    of height HH

    Taken from B. Vanniere."""
    close_to_orog = orog.copy()
    far_to_orog = orog.copy()

    close_to_orog.data = np.zeros(close_to_orog.shape)

    orog_sup = orog.data >= HH

    shape = orog.shape
    latitude = orog.coord('latitude').points
    longitude = orog.coord('longitude').points
    XX, YY = np.meshgrid(longitude, latitude)

    for i in range(shape[0]):
        for j in range(shape[1]):
            lat1 = latitude[i]
            lon1 = longitude[j]
            if orog_sup[i, j] == 1.:
                yn = distance(lon1, XX, lat1, YY) < KK
                close_to_orog.data += yn

    close_to_orog.data = close_to_orog.data > 0.
    far_to_orog.data = 1. - close_to_orog.data
    return close_to_orog, far_to_orog


def calc_extended_orog_mask(inputs, outputs):
    r_clim = iris.load_cube(str(inputs[0]))
    r_orog_mask = r_clim.copy()
    r_orog_mask.data = (r_clim.data > 0.5).astype(float)
    r_extended_orog_mask = r_orog_mask.copy()
    for i in range(12):
        print(i)
        r_extended_orog_mask[i].data = distance_orography(r_orog_mask[i], 100, 1)[0]
    iris.save(r_extended_orog_mask, str(outputs[0]))


def plot_masks_combined(inputs, outputs):
    """Borrow code from B. Vanniere to get plots looking very similar."""
    r_clim = iris.load_cube(str(inputs[0]))
    clevs = list(np.array([0.0, 1., 2., 4., 6., 8., 10., 15., 20., 30.]) / 20.)
    cmap = plt.cm.gist_stern_r
    cmaplist = [cmap(i) for i in range(0, cmap.N, cmap.N // 10)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    iplt.contourf(r_clim.collapsed('t', iris.analysis.MEAN), clevs,
                  cmap=cmap,
                  extent=(0, 360, -90, 90))
    cb = plt.colorbar(orientation='vertical')
    cb.set_ticks(clevs)
    plt.gca().coastlines()
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(8, 3.5)

    plt.savefig(outputs[0])
    r_extended_orog_mask = iris.load_cube(str(inputs[1]))
    plt.clf()
    cmap = plt.cm.jet
    clevs = np.arange(0.0, 1.1, 0.1)
    cmaplist = [cmap(i) for i in range(0, cmap.N, cmap.N // 10)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    iplt.contourf(r_extended_orog_mask.collapsed('t', iris.analysis.MEAN),
                  clevs,
                  cmap=cmap,
                  extent=(0, 360, -90, 90))
    cb = plt.colorbar(orientation='vertical')
    cb.set_ticks(clevs)

    plt.gca().coastlines()
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(8, 3.5)
    plt.savefig(outputs[1])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    tc.add(Task(plot_masks,
                [f'{BASEDIR}/R_clim.nc'],
                [PATHS['figsdir'] / 'experimental' / 'sinclair_orog_masks.png']))
    tc.add(Task(calc_extended_orog_mask,
                [f'{BASEDIR}/R_clim.nc'],
                [PATHS['figsdir'] / 'experimental' / 'extended_orog_mask.nc']))

    tc.add(Task(plot_masks_combined,
                [f'{BASEDIR}/R_clim.nc',
                 PATHS['figsdir'] / 'experimental' / 'extended_orog_mask.nc'],
                [PATHS['figsdir'] / 'experimental' / 'sinclair_orog_data_combined.png',
                 PATHS['figsdir'] / 'experimental' / 'sinclair_orog_masks_combined.png']))
    return tc


if __name__ == '__main__':
    tc = gen_task_ctrl()
    tc.finalize().run()
