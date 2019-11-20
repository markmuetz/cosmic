from argparse import ArgumentParser
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from .afi_base import AFI_base, load_cmap_data, MODES, TITLE_MODE_MAP, TITLE_RUNID_MAP


class AFI_diurnal_cycle(AFI_base):
    def __init__(self, datadir, duration, precip_thresh):
        super().__init__(datadir, duration, precip_thresh)
        self.name = 'afi_diurnal_cycle'

    def gen_axes(self):
        self.fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[1, 1, 1, 0.2])
        fig_axes = []
        cb_axes = []
        for i in range(3):
            ax_row = []
            for j in range(3):
                ax_row.append(self.fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree()))
            fig_axes.append(ax_row)
        cb_axes.append(plt.subplot(gs[-1, :]))
        return np.array(fig_axes), np.array(cb_axes)

    def add_titles_colourbars(self):
        for j, mode in enumerate(MODES):
            title_ax = self.fig_axes[0, j]
            title_ax.set_title(TITLE_MODE_MAP[mode])
            im = self.image_grid[-1][j]

        cax = self.cb_axes[0]
        cax.axis('off')
        im = self.image_grid[-1][1]

        cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
        plt.colorbar(im, ax=cax, label=f'diurnal cycle (hr)', 
                     orientation='horizontal', norm=norm, **cbar_kwargs,
                     fraction=0.6, aspect=35)

    def plot_ax(self, ax, cube, runid, mode):
        lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
        lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

        extent = (lon_min, lon_max, lat_min, lat_max)
        t_offset = cube.coord('longitude').points / 180 * 12

        if runid == 'cmorph_0p25':
            # CMORPH 0.25deg data is 3-hourly.
            season_peak_time_GMT = cube.data.argmax(axis=0) * 3
            season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 1.5) % 24
        elif runid == 'cmorph_8km':
            # CMORPH 8km data is 30-min-ly
            season_peak_time_GMT = cube.data.argmax(axis=0) / 2
            season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 0.25) % 24
        else:
            # model data is 1-hourly.
            season_peak_time_GMT = cube.data.argmax(axis=0)
            season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 0.5) % 24

        cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

        im = ax.imshow(season_peak_time_LST, origin='lower', extent=extent, cmap=cmap, norm=norm,
                       vmin=0, vmax=24)

        return im


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('duration')
    parser.add_argument('precip_thresh')
    args = parser.parse_args()

    afi_diurnal_cycle = AFI_diurnal_cycle(args.datadir, args.duration, args.precip_thresh)
    afi_diurnal_cycle.plot()
    afi_diurnal_cycle.save()
