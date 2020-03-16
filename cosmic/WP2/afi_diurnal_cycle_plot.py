import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from .afi_base import AFI_basePlotter, load_cmap_data
from cosmic.WP2.diurnal_cycle_analysis import calc_diurnal_cycle_phase_amp_peak, calc_diurnal_cycle_phase_amp_harmonic


class AFI_diurnalCyclePlotter(AFI_basePlotter):
    name = 'afi_diurnal_cycle'

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
        for j, mode in enumerate(self.MODES):
            title_ax = self.fig_axes[0, j]
            title_ax.set_title(self.TITLE_MODE_MAP[mode])
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

        if self.method == 'peak':
            season_phase_LST = calc_diurnal_cycle_phase_amp_peak(cube)[0]
        elif self.method == 'harmonic':
            season_phase_LST = calc_diurnal_cycle_phase_amp_harmonic(cube)[0]
        cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

        im = ax.imshow(season_phase_LST, origin='lower', extent=extent, cmap=cmap, norm=norm,
                       vmin=0, vmax=24)

        return im
