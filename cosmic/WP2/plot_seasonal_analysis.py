# coding: utf-8
import sys
from pathlib import Path
import pickle

import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
import iris
import cartopy.crs as ccrs
from scipy.ndimage.filters import gaussian_filter

from cosmic.util import sysrun

SEASONS = ['jja', 'son', 'djf', 'mam']
MODES = ['amount', 'freq', 'intensity']


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


class SeasonAnalysisPlotter:
    def __init__(self, datadir, runid, daterange, seasons, precip_thresh, resolution):
        self.datadir = datadir
        self.runid = runid
        self.daterange = daterange
        if seasons == 'all':
            self.seasons = SEASONS
        else:
            self.seasons = seasons.split(',')
        self.resolution = resolution
        self.precip_thresh = precip_thresh
        self.thresh_text = str(precip_thresh).replace('.', 'p')
        self.cubes = {}
        self.figdir = Path(f'figs/{runid}/{daterange}')
        self.load_cubes()

    def savefig(self, filename):
        filepath = self.figdir / filename
        dirname = filepath.absolute().parent
        dirname.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(filepath))

    def load_cubes(self):
        for i, season in enumerate(self.seasons):
            if self.runid[:6] == 'cmorph':
                if self.resolution:
                    filename = f'cmorph_ppt_{season}.{self.daterange}.asia_precip.ppt_thresh_{self.thresh_text}.{self.resolution}.nc'
                else:
                    filename = f'cmorph_ppt_{season}.{self.daterange}.asia_precip.ppt_thresh_{self.thresh_text}.nc'
            else:
                filename = f'{self.runid}a.p9{season}.{self.daterange}.asia_precip.ppt_thresh_{self.thresh_text}.nc'

            season_mean = iris.load_cube(f'{self.datadir}/{filename}',
                                         'precip_flux_mean')
            season_std = iris.load_cube(f'{self.datadir}/{filename}',
                                        'precip_flux_std')
            self.cubes[f'season_mean_{season}'] = season_mean
            self.cubes[f'season_std_{season}'] = season_std
            for mode in MODES:
                cube = iris.load_cube(f'{self.datadir}/{filename}',
                                      f'{mode}_of_precip_{season}')
                self.cubes[f'{mode}_{season}'] = cube

    def plot_season_mean(self):
        fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 12))

        season_mean_min = 1e10
        season_mean_max = 0
        season_std_min = 1e10
        season_std_max = 0

        for i, season in enumerate(self.seasons):
            season_mean = self.cubes[f'season_mean_{season}']
            season_std = self.cubes[f'season_std_{season}']

            season_mean_min = min(season_mean_min, season_mean.data.min())
            season_mean_max = max(season_mean_max, season_mean.data.max())
            season_std_min = min(season_std_min, season_std.data.min())
            season_std_max = max(season_std_max, season_std.data.max())
        season_mean_min = 1e-3
        season_mean_max = 3
        season_std_min = 1e-3

        lon_min, lon_max = season_mean.coord('longitude').points[[0, -1]]
        lat_min, lat_max = season_mean.coord('latitude').points[[0, -1]]

        scale_lon = 360 / 2560
        scale_lat = 180 / 1920

        extent = (lon_min, lon_max, lat_min, lat_max)

        plt.subplots_adjust(wspace=0.08, hspace=0.08, bottom=0.15, top=0.99)

        for i, season in enumerate(self.seasons):
            season_mean = self.cubes[f'season_mean_{season}']
            season_std = self.cubes[f'season_std_{season}']
            ax0 = axes[i, 0]
            ax1 = axes[i, 1]
            ax0.set_ylabel(f'{season}'.upper())

            # ax0.set_title(f'{season}'.upper())
            # ax1.set_title(f'{season}'.upper())
            ax0.set_xticks([60, 150])
            ax1.set_xticks([60, 150])
            ax0.set_yticks([10, 50])
            ax1.set_yticks([10, 50])
            # ax0.set_xlim((97.5, 125))
            # ax0.set_ylim((18, 41))
            # ax1.set_xlim((97.5, 125))
            # ax1.set_ylim((18, 41))

            im0 = ax0.imshow(season_mean.data, origin='lower', norm=LogNorm(), extent=extent,
                             vmin=season_mean_min, vmax=season_mean_max)
            im1 = ax1.imshow(season_std.data, origin='lower', norm=LogNorm(), extent=extent,
                             vmin=season_std_min, vmax=season_std_max)

            cax0 = fig.add_axes([0.13, 0.1, 0.35, 0.01])
            cax1 = fig.add_axes([0.54, 0.1, 0.35, 0.01])
            if i == 3:
                cb0 = plt.colorbar(im0, cax=cax0, 
                                   label='$\\mu$ precip. (mm hr$^{-1}$)',
                                   orientation='horizontal', extend='both')
                cb1 = plt.colorbar(im1, cax=cax1, 
                                   label='$\\sigma$ precip. (mm hr$^{-1}$)',
                                   orientation='horizontal', extend='min')
        self.savefig(f'season_mean/asia_{mode}_mean.png')


    def plot_season_afi_gmt(self, mode):
        for i, season in enumerate(self.seasons):
            cube = self.cubes[f'{mode}_{season}']
            lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
            lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

            extent = (lon_min, lon_max, lat_min, lat_max)
            for j, hour in enumerate(cube.coord('time')):
                plt.clf()
                plt.figure(figsize=(12.8, 9.6))
                title = f'{self.runid} {mode} {season} {j:02d} GMT ppt_thresh={self.precip_thresh} mm hr$^{{-1}}$'
                print(title)
                plt.title(title)
                if mode == 'freq':
                    kwargs = {'vmin': 0}
                    data = cube[j].data * 100
                    cbar_kwargs = {}
                    units = '%'
                elif mode == 'amount':
                    data = cube[j].data
                    kwargs = {'vmin': 1e-3, 'vmax': 3, 'norm': LogNorm()}
                    cbar_kwargs = {'extend': 'max'}
                    units = 'mm hr$^{-1}$'
                elif mode == 'intensity':
                    data = cube[j].data
                    kwargs = {'vmin': 1e-2, 'vmax': 3, 'norm': LogNorm()}
                    cbar_kwargs = {'extend': 'max'}
                    units = 'mm hr$^{-1}$'
                im0 = plt.imshow(data, origin='lower', extent=extent,
                                 **kwargs)
                plt.colorbar(im0, label=f'{mode} precip. ({units})',
                             orientation='horizontal', **cbar_kwargs)
                fig.set_size_inches(12, 8)
                self.savefig(f'ppt_thresh_{self.thresh_text}/hourly/GMT/asia_{mode}_{season}_hr{j:02d}_GMT.png')
                plt.close('all')

    def plot_season_afi_mean(self, mode):
        for i, season in enumerate(self.seasons):
            cube = self.cubes[f'{mode}_{season}']
            lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
            lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

            extent = (lon_min, lon_max, lat_min, lat_max)
            plt.clf()
            ax = plt.axes(projection=ccrs.PlateCarree())
            fig = plt.gcf()
            ax.coastlines()

            # plt.figure(figsize=(12.8, 9.6))
            title = f'{self.runid} {mode} {season} mean ppt_thresh={self.precip_thresh} mm hr$^{{-1}}$'
            print(title)
            plt.title(title)
            if mode == 'freq':
                data = cube.data.mean(axis=0) * 100
                units = '%'
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb2.pkl')
                kwargs = {'vmin': 0, 'cmap': cmap}
                cbar_kwargs['norm'] = norm
                kwargs['vmin'] = 3
                kwargs['vmax'] = 40

                # if self.precip_thresh == 0.1:
                #     # Consistent with Li 2018.
                #     kwargs['vmin'] = 3
                #     kwargs['vmax'] = 40
                #     ticks = [3, 5, 8, 15, 25, 40]
                #     kwargs['norm'] = VrangeNorm(ticks)
                #     cbar_kwargs['extend'] = 'both'
            elif mode == 'amount':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')
                cbar_kwargs['norm'] = norm
                data = cube.data.mean(axis=0) * 24  # mm/hr -> mm/day
                kwargs = {'vmin': 1e-3, 'vmax': 12, 'cmap': cmap}
                units = 'mm day$^{-1}$'
            elif mode == 'intensity':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb3.pkl')
                cbar_kwargs['norm'] = norm
                data = cube.data.mean(axis=0)
                kwargs = {'vmin': 1e-2, 'vmax': 4, 'cmap': cmap}
                # cbar_kwargs = {'extend': 'max'}
                units = 'mm hr$^{-1}$'
            im0 = ax.imshow(data, origin='lower', norm=norm, extent=extent, **kwargs)
            rect = Rectangle((97.5, 18), 125 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

            # cax = fig.add_axes([0.12, 0.1, 0.8, 0.03])
            # cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
            #                                spacing='uniform',
            #                                orientation='horizontal',
            #                                label=f'{mode} precip. ({units})',
            #                                **cbar_kwargs)

            plt.colorbar(im0, label=f'{mode} precip. ({units})',
                         orientation='horizontal', **cbar_kwargs, spacing='uniform')
            if mode == 'amount':
                index_argmax = np.unravel_index(data.argmax(), data.shape)
                print(f'max. amount: {data.max()} mm hr-1')
                lat = cube[:, index_argmax[0], index_argmax[1]].coord('latitude').points[0]
                lon = cube[:, index_argmax[0], index_argmax[1]].coord('longitude').points[0]
                print(f'lat, lon: {lat}, {lon}')
                ax.plot(lon, lat, 'kx')
            fig.set_size_inches(12, 8)
            print(extent)
            ax.set_xticks(np.linspace(60, 150, 10))
            ax.set_yticks(np.linspace(10, 50, 5))
            self.savefig(f'ppt_thresh_{self.thresh_text}/hourly/mean/asia_{mode}_{season}_mean.png')

            # Same as Li 2018.
            ax.set_xlim((97.5, 125))
            ax.set_ylim((18, 41))
            ax.set_xticks([100, 110, 120])
            ax.set_yticks([20, 30, 40])
            fig.set_size_inches(6, 8)
            self.savefig(f'ppt_thresh_{self.thresh_text}/hourly/mean/china_{mode}_{season}_mean.png')
            plt.close('all')


    def plot_season_afi_lst(self, mode):
        for i, season in enumerate(self.seasons):
            cube = self.cubes[f'{mode}_{season}']
            lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
            lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

            extent = (lon_min, lon_max, lat_min, lat_max)
            hours = np.arange(25)
            offset_fns = []
            print('Gen interp fns')
            for lon_index, lon in enumerate(cube.coord('longitude').points):
                # Need to handle all values between 0 and 24 in interp; repeat first time at end of interp_data.
                interp_data = np.zeros((cube.shape[0] + 1, cube.shape[1]))
                interp_data[:-1] = cube[:, :, lon_index].data
                interp_data[-1] = interp_data[0]

                offset_fn = interp1d(hours, interp_data, axis=0)
                offset_fns.append(offset_fn)

            for j, hour in enumerate(cube.coord('time')):
                plt.clf()
                plt.figure(figsize=(12.8, 9.6))
                title = f'{self.runid} {mode} {season} {j:02d} LST ppt_thresh={self.precip_thresh} mm hr$^{{-1}}$'
                print(title)
                plt.title(title)

                data_lst = np.zeros_like(cube[j].data)
                for lon_index, lon in enumerate(cube.coord('longitude').points):
                    t_offset = lon / 180 * 12
                    offset_fn = offset_fns[lon_index]
                    data_lst[:, lon_index] = offset_fn((j - t_offset) % 24)

                if mode == 'freq':
                    kwargs = {'vmin': 0}
                    data_lst = data_lst * 100
                    units = '%'
                elif mode == 'amount':
                    data_lst = data_lst
                    kwargs = {'vmin': 1e-3, 'vmax': 30, 'norm': LogNorm()}
                    units = 'mm hr$^{-1}$'
                elif mode == 'intensity':
                    data_lst = data_lst
                    kwargs = {'vmin': 1e-2, 'vmax': 30, 'norm': LogNorm()}
                    units = 'mm hr$^{-1}$'

                im0 = plt.imshow(data_lst, origin='lower', extent=extent,
                                 **kwargs)
                plt.colorbar(im0,
                             label=f'{mode} precip. ({units})',
                             orientation='horizontal')
                fig.set_size_inches(12, 8)
                self.savefig(f'ppt_thresh_{self.thresh_text}/hourly/LST/asia_{mode}_{season}_hr{j:02d}_LST.png')
                plt.close('all')

    def plot_afi_diurnal_cycle(self, mode, cmap_name='li2018fig3', overlay_style=None):
        for season in self.seasons:
            cube = self.cubes[f'{mode}_{season}']
            lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
            lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

            extent = (lon_min, lon_max, lat_min, lat_max)
            t_offset = cube.coord('longitude').points / 180 * 12
            if self.runid == 'cmorph_0p25':
                # CMORPH 0.25deg data is 3-hourly.
                season_peak_time_GMT = cube.data.argmax(axis=0) * 3
                season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 1.5) % 24
            elif self.runid == 'cmorph_8km':
                # CMORPH 8km data is 30-min-ly
                season_peak_time_GMT = cube.data.argmax(axis=0) / 2
                season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 0.25) % 24
            else:
                # model data is 1-hourly.
                season_peak_time_GMT = cube.data.argmax(axis=0)
                season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 0.5) % 24

            ax = plt.axes(projection=ccrs.PlateCarree())
            fig = plt.gcf()
            ax.coastlines()

            title = f'{self.runid} diurnal cycle time {mode} {season} LST ppt_thresh={self.precip_thresh} mm hr$^{{-1}}$'
            print(title)
            plt.title(title)

            season_max = cube.data.max(axis=0)
            season_mean = cube.data.mean(axis=0)
            season_strength = season_max / season_mean

            season_strength_filtered = gaussian_filter(season_strength, 3)
            thresh_boundaries = [100 * 1 / 3, 100 * 2 / 3]
            # thresh_boundaries = [100 * 1 / 4, 100 * 1 / 3]
            filtered_med_thresh, filtered_strong_thresh = np.percentile(season_strength_filtered, 
                                                                        thresh_boundaries)
            med_thresh, strong_thresh = np.percentile(season_strength, 
                                                      thresh_boundaries)

            if cmap_name == 'li2018fig3':
                cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
                imshow_kwargs = {'cmap': cmap, 'norm': norm}
                cbar_kwargs['norm'] = norm
            elif cmap_name == 'sky_colour':
                cmap = LinearSegmentedColormap.from_list('cmap',['k','pink','cyan','blue','red','k'], 
                                                         N=24)
                imshow_kwargs = {'cmap': cmap}
                cbar_kwargs = {}

            if overlay_style != 'alpha_overlay':
                im0 = ax.imshow(season_peak_time_LST, origin='lower', extent=extent,
                                vmin=0, vmax=24, **imshow_kwargs)
            elif overlay_style == 'alpha_overlay':
                peak_strong = np.ma.masked_array(season_peak_time_LST, 
                                                 season_strength < strong_thresh)
                peak_med = np.ma.masked_array(season_peak_time_LST, 
                                              ((season_strength > strong_thresh) | 
                                               (season_strength < med_thresh)))
                peak_weak = np.ma.masked_array(season_peak_time_LST, 
                                               season_strength > med_thresh)

                im0 = ax.imshow(peak_strong, origin='lower', extent=extent,
                                vmin=0, vmax=24, **imshow_kwargs)
                ax.imshow(peak_med, origin='lower', extent=extent, alpha=0.66,
                          vmin=0, vmax=24, **imshow_kwargs)
                ax.imshow(peak_weak, origin='lower', extent=extent, alpha=0.33,
                          vmin=0, vmax=24, **imshow_kwargs)

            plt.colorbar(im0, label=f'{mode} peak (hr)', orientation='horizontal', 
                         **cbar_kwargs)

            if overlay_style == 'contour_overlay':
                plt.contour(season_strength_filtered, [filtered_med_thresh, filtered_strong_thresh], colors=['w', 'w'], linestyles=['--', '-'], extent=extent)

            rect = Rectangle((97.5, 18), 125 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
            fig.set_size_inches(12, 8)
            self.savefig(f'ppt_thresh_{self.thresh_text}/diurnal_cycle/asia_diurnal_cycle_{mode}_{season}_peak.{cmap_name}.{overlay_style}.png')
            ax.set_xlim((97.5, 125))
            ax.set_ylim((18, 41))
            ax.set_xticks([100, 110, 120])
            ax.set_yticks([20, 30, 40])
            fig.set_size_inches(6, 8)
            self.savefig(f'ppt_thresh_{self.thresh_text}/diurnal_cycle/china_diurnal_cycle_{mode}_{season}_peak.{cmap_name}.{overlay_style}.png')

            plt.close('all')

            ax = plt.axes(projection=ccrs.PlateCarree())
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            ax.coastlines()
            title = f'{self.runid} diurnal cycle strength {mode} {season} LST ppt_thresh={self.precip_thresh} mm hr$^{{-1}}$'
            plt.title(title)
            if mode == 'freq':
                kwargs = {'vmin': 1, 'vmax': 11, 'norm': LogNorm()}
            elif mode == 'amount':
                kwargs = {'vmin': 1, 'vmax': 22, 'norm': LogNorm()}
            elif mode == 'intensity':
                kwargs = {'vmin': 1, 'vmax': 22, 'norm': LogNorm()}

            im0 = plt.imshow(season_strength, origin='lower', extent=extent, **kwargs)
            rect = Rectangle((97.5, 18), 125 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

            plt.colorbar(im0, label=f'{mode} strength (-)', orientation='horizontal')
            fig.set_size_inches(12, 8)
            self.savefig(f'ppt_thresh_{self.thresh_text}/diurnal_cycle/asia_diurnal_cycle_{mode}_{season}_strength.png')
            ax.set_xlim((97.5, 125))
            ax.set_ylim((18, 41))
            ax.set_xticks([100, 110, 120])
            ax.set_yticks([20, 30, 40])
            fig.set_size_inches(6, 8)
            self.savefig(f'ppt_thresh_{self.thresh_text}/diurnal_cycle/china_diurnal_cycle_{mode}_{season}_strength.png')
            plt.close('all')

    def gen_animations(self):
        for season in self.seasons:
            for mode in MODES:
                for timemode in ['GMT', 'LST']:
                    infiles = f'ppt_thresh_{self.thresh_text}/hourly/{timemode}/asia_{mode}_{season}_hr*_{timemode}.png'
                    outfile = f'ppt_thresh_{self.thresh_text}/anim/asia_{mode}_{season}_anim_{timemode}.gif'
                    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                    cmd = f'convert -delay 30 -loop 0 {infiles} {outfile}'
                    print(cmd)
                    sysrun(cmd)


def main(basepath, runid, daterange, seasons, resolution, precip_threshes=[0.1]):
    if runid == 'cmorph_0p25':
        datadir = Path(f'{basepath}/cmorph_data/0.25deg-3HLY')
    elif runid == 'cmorph_8km':
        datadir = Path(f'{basepath}/cmorph_data/8km-30min')
    else:
        datadir = Path(f'{basepath}/u-{runid}/ap9.pp')

    for precip_thresh in precip_threshes:
        plotter = SeasonAnalysisPlotter(datadir, runid, daterange, seasons, precip_thresh, resolution)
        # plt.ion()

        for mode in MODES:
            # plotter.plot_season_afi_gmt(mode)
            # plotter.plot_season_afi_lst(mode)
            plotter.plot_season_afi_mean(mode)
            plotter.plot_afi_diurnal_cycle(mode, overlay_style='alpha_overlay')

        # gen_animations()

if __name__ == '__main__':
    basepath = sys.argv[1]
    runid = sys.argv[2]
    daterange = sys.argv[3]
    seasons = sys.argv[4]
    resolution = sys.argv[5]
    if resolution == 'None':
        resolution = None
    if len(sys.argv) > 6:
        precip_threshes = [float(v) for v in sys.argv[6:]]
        main(basepath, runid, daterange, seasons, resolution, precip_threshes)
    else:
        main(basepath, runid, daterange, seasons, resolution)

