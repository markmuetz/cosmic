import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from afi_base import AFI_base, load_cmap_data, MODES, TITLE_MODE_MAP, TITLE_RUNID_MAP


class AFI_mean(AFI_base):
    def __init__(self, datadir, duration, precip_thresh):
        super().__init__(datadir, duration, precip_thresh)
        self.name = 'afi_mean'

    def gen_axes(self):
        self.fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[1, 1, 1, 0.2])
        fig_axes = []
        cb_axes = []
        for i in range(3):
            ax_row = []
            for j in range(3):
                ax_row.append(plt.subplot(gs[i, j], projection=ccrs.PlateCarree()))
            fig_axes.append(ax_row)
        for j in range(3):
            cb_axes.append(plt.subplot(gs[-1, j]))
        return np.array(fig_axes), np.array(cb_axes)

    def add_titles_colourbars(self):
        for j, mode in enumerate(MODES):
            title_ax = self.fig_axes[0, j]
            title_ax.set_title(TITLE_MODE_MAP[mode])
            im = self.image_grid[-1][j]

            cax = self.cb_axes[j]
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

    def plot_ax(self, ax, cube, runid, mode):
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
        return im


if __name__ == '__main__':
    datadir = sys.argv[1]
    duration = sys.argv[2]
    precip_thresh = sys.argv[3]
    afi_mean = AFI_mean(datadir, duration, precip_thresh)
    afi_mean.plot()
    afi_mean.save()
