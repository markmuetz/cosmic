import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import iris
import cartopy.crs as ccrs

MODES = ['amount', 'freq', 'intensity']

TITLE_MODE_MAP = {
    'amount': 'Amount',
    'freq': 'Frequency',
    'intensity': 'Intensity',
}
TITLE_RUNID_MAP = {
    'cmorph_8km': 'CMORPH 8 km',
    'ak543': 'UM explicit conv.',
    'al508': 'UM parametrized conv.',
}


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


class AFI_mean:
    def __init__(self, datadir, duration, precip_thresh):
        self.datadir = Path(datadir)
        self.duration = duration
        self.precip_thresh = precip_thresh
        self.thresh_text = str(precip_thresh).replace('.', 'p')

        self.runids = ['cmorph_8km', 'ak543', 'al508']
        self.season = 'jja'

        self.cubes = {}
        self.load_cubes()
        self.axes = self.gen_axes()
        self.fig_axes = self.axes[:-1]


    def gen_axes(self):
        self.fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[1, 1, 1, 0.2])
        # colorbar_ax = fig.add_axes([0.42, 0.06, 0.4, 0.01])
        axes = []
        for i in range(4):
            ax_row = []
            for j in range(3):
                if i < 3:
                    ax_row.append(plt.subplot(gs[i, j], projection=ccrs.PlateCarree()))
                else:
                    ax_row.append(plt.subplot(gs[i, j]))
            axes.append(ax_row)
        return np.array(axes)

    def load_cubes(self):
        for runid in self.runids:
            self.cubes[runid] = {}
            if runid == 'cmorph_8km':
                rel_path = 'cmorph_data/8km-30min'
                if self.duration == 'short':
                    daterange = '200906-200908'
                elif self.duration == 'long':
                    daterange = '199801-201812'
                filename = f'cmorph_ppt_{self.season}.{daterange}.asia_precip.ppt_thresh_{self.thresh_text}.N1280.nc'
            else:
                if self.duration == 'short':
                    daterange = '200806-200808'
                elif self.duration == 'long':
                    daterange = '200502-200901'

                rel_path = f'u-{runid}/ap9.pp'
                filename = f'{runid}a.p9{self.season}.{daterange}.asia_precip.ppt_thresh_{self.thresh_text}.nc'

            for mode in MODES:
                cube = iris.load_cube(str(self.datadir / rel_path / filename), f'{mode}_of_precip_{self.season}')
                self.cubes[runid][f'{mode}_{self.season}'] = cube

    def plot(self):
        ims = []
        for i, runid in enumerate(self.runids):
            for j, mode in enumerate(MODES):
                ax = self.fig_axes[i, j]
                if i == 2:
                    ims.append(self.plot_ax(ax, self.cubes[runid][f'{mode}_{self.season}'], mode))
                else:
                    self.plot_ax(ax, self.cubes[runid][f'{mode}_{self.season}'], mode)
                if i != 2:
                    ax.get_xaxis().set_ticks([])

                if j == 0:
                    ax.set_ylabel(TITLE_RUNID_MAP[runid])
                else:
                    ax.get_yaxis().set_ticks([])

        for j, (mode, im) in enumerate(zip(MODES, ims)):
            title_ax = self.axes[0, j]
            title_ax.set_title(TITLE_MODE_MAP[mode])

            cax = self.axes[-1, j]
            cax.axis('off')
            if mode == 'amount':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')
                cbar_kwargs['norm'] = norm
                units = 'mm day$^{-1}$'
            elif mode == 'freq':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb2.pkl')
                cbar_kwargs['norm'] = norm
                units = '%'
            elif mode == 'intensity':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb3.pkl')
                cbar_kwargs['norm'] = norm
                units = 'mm hr$^{-1}$'

            plt.colorbar(im, ax=cax, label=f'{mode} precip. ({units})',
                         **cbar_kwargs, spacing='uniform', 
                         orientation='horizontal', fraction=0.9)
        self.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, wspace=0.1, hspace=0.1) 

    def plot_ax(self, ax, cube, mode):
        lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
        lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

        extent = (lon_min, lon_max, lat_min, lat_max)
        if mode == 'amount':
            cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')
            data = cube.data.mean(axis=0) * 24  # mm/hr -> mm/day
            kwargs = {'vmin': 1e-3, 'vmax': 12, 'cmap': cmap}
        elif mode == 'freq':
            data = cube.data.mean(axis=0) * 100
            cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb2.pkl')
            kwargs = {'vmin': 0, 'cmap': cmap}
            kwargs['vmin'] = 3
            kwargs['vmax'] = 40
        elif mode == 'intensity':
            cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb3.pkl')
            data = cube.data.mean(axis=0)
            kwargs = {'vmin': 1e-2, 'vmax': 4, 'cmap': cmap}
        kwargs['norm'] = norm

        im = ax.imshow(data, origin='lower', extent=extent, **kwargs)
        ax.set_xlim((97.5, 125))
        ax.set_ylim((18, 41))
        ax.set_xticks([100, 110, 120])
        ax.set_yticks([20, 30, 40])
        ax.coastlines(resolution='50m')
        return im
    
    def save(self):
        plt.savefig(f'figs/afi_mean.{self.duration}.{self.season}.ppt_thresh_{self.precip_thresh}.png')


if __name__ == '__main__':
    datadir = sys.argv[1]
    duration = sys.argv[2]
    precip_thresh = sys.argv[3]
    afi_mean = AFI_mean(datadir, duration, precip_thresh)
    afi_mean.plot()
    afi_mean.save()
