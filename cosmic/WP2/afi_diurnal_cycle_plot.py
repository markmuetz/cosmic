import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from .afi_base import AFI_basePlotter
from cosmic.util import load_cmap_data
from cosmic.WP2.diurnal_cycle_analysis import calc_diurnal_cycle_phase_amp_peak, calc_diurnal_cycle_phase_amp_harmonic


class AFI_diurnalCyclePlotter(AFI_basePlotter):
    def gen_axes(self):
        if self.domain == 'china':
            gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[1, 1, 1, 0.3])
        elif self.domain in ['asia', 'europe']:
            gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[1, 1, 1, 0.3])
        fig_axes = []
        cb_axes = []
        for i in range(3):
            ax_row = []
            for j in range(3):
                ax_row.append(self.fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree()))
            fig_axes.append(ax_row)
        # cb_axes.append(plt.subplot(gs[-1, :]))
        cb_axes.append(self.fig.add_axes([0.1, 0.06, 0.85, 0.06]))
        return np.array(fig_axes), np.array(cb_axes)

    def add_titles_colourbars(self):
        for j, mode in enumerate(self.MODES):
            title_ax = self.fig_axes[0, j]
            title_ax.set_title(self.TITLE_MODE_MAP[mode])
            im = self.image_grid[-1][j]

        cax = self.cb_axes[0]
        # cax.axis('off')
        im = self.image_grid[-1][1]

        cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
        if True:
            v = np.linspace(0, 1, 24)
            d = cmap(v)[None, :, :4] * np.ones((3, 24, 4))
            d[1, :, 3] = 0.66
            d[0, :, 3] = 0.33
            cax.imshow(d, origin='lower', extent=(0, 24, 0, 2), aspect='auto')
            cax.set_yticks([0.3, 1.7])
            cax.set_yticklabels(['weak', 'strong'])
            cax.set_xticks(np.linspace(0, 24, 9))
            cax.set_xlabel('phase and strength of diurnal cycle')
        else:
            plt.colorbar(im, ax=cax, label=f'diurnal cycle (hr)',
                         orientation='horizontal', norm=norm, **cbar_kwargs,
                         fraction=0.6, aspect=35)

    def plot_ax(self, ax, cube, runid, mode):
        lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
        lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

        extent = (lon_min, lon_max, lat_min, lat_max)

        if self.method == 'peak':
            season_phase_LST, amp = calc_diurnal_cycle_phase_amp_peak(cube)
        elif self.method == 'harmonic':
            season_phase_LST, amp = calc_diurnal_cycle_phase_amp_harmonic(cube)
        cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
        if runid == 'cmorph_8km':
            Lat, Lon = np.meshgrid(cube.coord('latitude').points, cube.coord('longitude').points, indexing='ij')
            season_phase_LST = np.ma.masked_array(season_phase_LST, Lat > 59)
            amp = np.ma.masked_array(amp, Lat > 59)

        if True:
            thresh_boundaries = [100 * 1 / 3, 100 * 2 / 3]
            if runid == 'cmorph_8km':
                med_thresh, strong_thresh = np.percentile(amp.compressed(),
                                                          thresh_boundaries)
            else:
                med_thresh, strong_thresh = np.percentile(amp,
                                                          thresh_boundaries)
            peak_strong = np.ma.masked_array(season_phase_LST,
                                             amp < strong_thresh)
            peak_med = np.ma.masked_array(season_phase_LST,
                                          ((amp >= strong_thresh) |
                                           (amp < med_thresh)))
            peak_weak = np.ma.masked_array(season_phase_LST,
                                           amp >= med_thresh)
            im0 = ax.imshow(peak_strong, origin='lower', extent=extent,
                            vmin=0, vmax=24, cmap=cmap, norm=norm)
            ax.imshow(peak_med, origin='lower', extent=extent, alpha=0.66,
                      vmin=0, vmax=24, cmap=cmap, norm=norm)
            ax.imshow(peak_weak, origin='lower', extent=extent, alpha=0.33,
                      vmin=0, vmax=24, cmap=cmap, norm=norm)

            return im0
        else:
            im = ax.imshow(season_phase_LST, origin='lower', extent=extent, cmap=cmap, norm=norm,
                           vmin=0, vmax=24)
            return im
