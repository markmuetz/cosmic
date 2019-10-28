# coding: utf-8
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
import iris
import cartopy.crs as ccrs

from cosmic.util import sysrun

SEASONS = ['jja', 'son', 'djf', 'mam']

li2018_diurnal_colours = ['#035965', '#046a63', '#057c56', '#06a026', '#20ba0d', '#5dcd09',
                          '#89da06', '#b4e704', '#fef500', '#fee000', '#fed000', '#febd00',
                          '#feab00', '#fe8f00', '#fe6a00', '#fe5600', '#fe2c00', '#fe1400',
                          '#ae0040', '#840062', '#5b0085', '#023eaa', '#0123ce', '#0011e6']
li2018_scale_colours = ['#d5e1fe', '#8db1fe', '#7e95fe', '#0062fe', '#009595', '#62fe00',
                        '#95fe00', '#fefe00', '#fec500', '#fe7b00', '#fe1800', '#840062']

li2018_diurnal = LinearSegmentedColormap.from_list('li2018_diurnal', li2018_diurnal_colours)
li2018_scale = LinearSegmentedColormap.from_list('li2018_scale', li2018_scale_colours[:-1])
li2018_scale_freq = LinearSegmentedColormap.from_list('li2018_scale', li2018_scale_colours[3:])


class VrangeNorm(colors.Normalize):
    """Recreate colorbar used in Fig. 2 d e and f.

    Taken from MidpointNormalize: https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html"""
    def __init__(self, vrange, clip=False):
        self.vrange = np.array(vrange)
        colors.Normalize.__init__(self, self.vrange[0], self.vrange[-1], clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # x, y = self.vrange, np.linspace(0, 1, len(self.vrange))
        x = self.vrange
        y = np.linspace(0, 1, len(self.vrange))
        return np.ma.masked_array(np.interp(value, x, y))


def savefig(filename):
    dirname = Path(filename).absolute().parent
    dirname.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)


def plot_season_mean(datadir, runid, precip_thresh=0.1):
    fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 12))
    thresh_text = str(precip_thresh).replace('.', 'p')

    season_mean_min = 1e10
    season_mean_max = 0
    season_std_min = 1e10
    season_std_max = 0

    for i, season in enumerate(SEASONS):
        season_mean = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                                     'precip_flux_mean')
        season_std = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                                    'precip_flux_std')
        season_mean_min = min(season_mean_min, season_mean.data.min() * 3600)
        season_mean_max = max(season_mean_max, season_mean.data.max() * 3600)
        season_std_min = min(season_std_min, season_std.data.min() * 3600)
        season_std_max = max(season_std_max, season_std.data.max() * 3600)
    season_mean_min = 1e-3
    season_mean_max = 3
    season_std_min = 1e-3

    lon_min, lon_max = season_mean.coord('longitude').points[[0, -1]]
    lat_min, lat_max = season_mean.coord('latitude').points[[0, -1]]

    scale_lon = 360 / 2560
    scale_lat = 180 / 1920

    extent = (lon_min, lon_max, lat_min, lat_max)

    plt.subplots_adjust(wspace=0.08, hspace=0.08, bottom=0.15, top=0.99)

    for i, season in enumerate(SEASONS):
        season_mean = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                                     'precip_flux_mean')
        season_std = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                                    'precip_flux_std')
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]
        ax0.set_ylabel(f'{season}'.upper())

        # ax0.set_title(f'{season}'.upper())
        # ax1.set_title(f'{season}'.upper())
        ax0.set_xticks([60, 150])
        ax1.set_xticks([60, 150])
        ax0.set_yticks([10, 50])
        ax1.set_yticks([10, 50])
        # ax0.set_xlim((97.5, 122.5))
        # ax0.set_ylim((18, 41))
        # ax1.set_xlim((97.5, 122.5))
        # ax1.set_ylim((18, 41))

        im0 = ax0.imshow(season_mean.data * 3600, origin='lower', norm=LogNorm(), extent=extent,
                         vmin=season_mean_min, vmax=season_mean_max)
        im1 = ax1.imshow(season_std.data * 3600, origin='lower', norm=LogNorm(), extent=extent,
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
    savefig(f'figs/{runid}/season_mean/asia_{mode}_mean.png')


def plot_season_gmt(datadir, runid, mode, precip_thresh=0.1):
    thresh_text = str(precip_thresh).replace('.', 'p')
    for i, season in enumerate(SEASONS):
        cube = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                              f'{mode}_of_precip_{season}')
        lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
        lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

        extent = (lon_min, lon_max, lat_min, lat_max)
        for j, hour in enumerate(cube.coord('time')):
            plt.clf()
            plt.figure(figsize=(12.8, 9.6))
            title = f'{mode} {season} {j:02d} GMT ppt_thresh={precip_thresh} mm hr$^{{-1}}$'
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
            savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/hourly/GMT/asia_{mode}_{season}_hr{j:02d}_GMT.png')
            plt.close('all')

        plt.clf()
        ax = plt.axes(projection=ccrs.PlateCarree())
        fig = plt.gcf()
        fig.set_size_inches(6, 8)
        ax.coastlines()

        # plt.figure(figsize=(12.8, 9.6))
        title = f'{mode} {season} mean ppt_thresh={precip_thresh} mm hr$^{{-1}}$'
        print(title)
        plt.title(title)
        if mode == 'freq':
            data = cube.data.mean(axis=0) * 100
            kwargs = {'vmin': 0, 'cmap': li2018_scale_freq}
            cbar_kwargs = {}
            units = '%'
            if precip_thresh == 0.1:
                # Consistent with Li 2018.
                kwargs['vmin'] = 3
                kwargs['vmax'] = 40
                ticks = [3, 5, 8, 15, 25, 40]
                kwargs['norm'] = VrangeNorm(ticks)
                cbar_kwargs['extend'] = 'both'
        elif mode == 'amount':
            data = cube.data.mean(axis=0)
            kwargs = {'vmin': 1e-3, 'vmax': 0.5, 'cmap': li2018_scale}
            cbar_kwargs = {'extend': 'max'}
            units = 'mm hr$^{-1}$'
        elif mode == 'intensity':
            data = cube.data.mean(axis=0)
            kwargs = {'vmin': 1e-2, 'vmax': 4, 'cmap': li2018_scale}
            cbar_kwargs = {'extend': 'max'}
            ticks = [0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.4, 3, 4]
            kwargs['norm'] = VrangeNorm(ticks)
            units = 'mm hr$^{-1}$'
        im0 = ax.imshow(data, origin='lower', extent=extent, **kwargs)
        rect = Rectangle((97.5, 18), 122.5 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        plt.colorbar(im0, label=f'{mode} precip. ({units})',
                     orientation='horizontal', **cbar_kwargs)
        if mode == 'amount':
            index_argmax = np.unravel_index(data.argmax(), data.shape)
            print(f'max. amount: {data.max()} mm hr-1')
            lat = cube[:, index_argmax[0], index_argmax[1]].coord('latitude').points[0]
            lon = cube[:, index_argmax[0], index_argmax[1]].coord('longitude').points[0]
            print(f'lat, lon: {lat}, {lon}')
            ax.plot(lon, lat, 'kx')
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/hourly/mean/asia_{mode}_{season}_mean.png')

        # Same as Li 2018.
        ax.set_xlim((97.5, 122.5))
        ax.set_ylim((18, 41))
        ax.set_xticks([100, 110, 120])
        ax.set_yticks([20, 30, 40])
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/hourly/mean/china_{mode}_{season}_mean.png')
        plt.close('all')


def plot_season_lst(datadir, runid, mode, precip_thresh=0.1):
    thresh_text = str(precip_thresh).replace('.', 'p')
    for i, season in enumerate(SEASONS):
        cube = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                              f'{mode}_of_precip_{season}')
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
            title = f'{mode} {season} {j:02d} LST ppt_thresh={precip_thresh} mm hr$^{{-1}}$'
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
            savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/hourly/LST/asia_{mode}_{season}_hr{j:02d}_LST.png')
            plt.close('all')


def gen_animations(datadir, runid, precip_thresh=0.1):
    thresh_text = str(precip_thresh).replace('.', 'p')
    for season in SEASONS:
        for mode in ['freq', 'amount', 'intensity']:
            for timemode in ['GMT', 'LST']:
                infiles = f'figs/{runid}/ppt_thresh_{thresh_text}/hourly/{timemode}/asia_{mode}_{season}_hr*_{timemode}.png'
                outfile = f'figs/{runid}/ppt_thresh_{thresh_text}/anim/asia_{mode}_{season}_anim_{timemode}.gif'
                Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                cmd = f'convert -delay 30 -loop 0 {infiles} {outfile}'
                print(cmd)
                sysrun(cmd)


def plot_diurnal_cycle(datadir, runid, mode, precip_thresh=0.1):
    thresh_text = str(precip_thresh).replace('.', 'p')
    for season in SEASONS:
        cube = iris.load_cube(f'{datadir}/{runid}a.p9{season}.asia_precip.ppt_thresh_{thresh_text}.nc',
                              f'{mode}_of_precip_{season}')
        lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
        lat_min, lat_max = cube.coord('latitude').points[[0, -1]]

        extent = (lon_min, lon_max, lat_min, lat_max)
        season_peak_time_GMT = cube.data.argmax(axis=0)

        t_offset = cube.coord('longitude').points / 180 * 12
        season_peak_time_LST = (season_peak_time_GMT + t_offset[None, :] + 0.5) % 24

        ax = plt.axes(projection=ccrs.PlateCarree())
        fig = plt.gcf()
        fig.set_size_inches(6, 8)
        ax.coastlines()

        title = f'diurnal cycle time {mode} {season} LST ppt_thresh={precip_thresh} mm hr$^{{-1}}$'
        print(title)
        plt.title(title)
        im0 = ax.imshow(season_peak_time_LST, origin='lower', extent=extent, cmap=li2018_diurnal)
        rect = Rectangle((97.5, 18), 122.5 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        plt.colorbar(im0, label=f'{mode} peak (hr)', orientation='horizontal')
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/diurnal_cycle/asia_diurnal_cycle_{mode}_{season}_peak.png')
        ax.set_xlim((97.5, 122.5))
        ax.set_ylim((18, 41))
        ax.set_xticks([100, 110, 120])
        ax.set_yticks([20, 30, 40])
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/diurnal_cycle/china_diurnal_cycle_{mode}_{season}_peak.png')
        plt.close('all')

        season_max = cube.data.max(axis=0)
        season_mean = cube.data.mean(axis=0)
        season_strength = season_max / season_mean

        ax = plt.axes(projection=ccrs.PlateCarree())
        fig = plt.gcf()
        fig.set_size_inches(12.8, 9.6)
        ax.coastlines()
        title = f'diurnal cycle strength {mode} {season} LST ppt_thresh={precip_thresh} mm hr$^{{-1}}$'
        plt.title(title)
        if mode == 'freq':
            kwargs = {'vmin': 1, 'vmax': 11, 'norm': LogNorm()}
        elif mode == 'amount':
            kwargs = {'vmin': 1, 'vmax': 22, 'norm': LogNorm()}
        elif mode == 'intensity':
            kwargs = {'vmin': 1, 'vmax': 22, 'norm': LogNorm()}

        im0 = plt.imshow(season_strength, origin='lower', extent=extent, **kwargs)
        rect = Rectangle((97.5, 18), 122.5 - 97.5, 41 - 18, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        plt.colorbar(im0, label=f'{mode} strength (-)', orientation='horizontal')
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/diurnal_cycle/asia_diurnal_cycle_{mode}_{season}_strength.png')
        ax.set_xlim((97.5, 122.5))
        ax.set_ylim((18, 41))
        ax.set_xticks([100, 110, 120])
        ax.set_yticks([20, 30, 40])
        savefig(f'figs/{runid}/ppt_thresh_{thresh_text}/diurnal_cycle/china_diurnal_cycle_{mode}_{season}_strength.png')
        plt.close('all')


def main(basepath, runid):
    datadir = Path(f'{basepath}/u-{runid}/ap9.pp')

    for precip_thresh in [0.1]:
        # plt.ion()

        for mode in ['amount', 'freq', 'intensity']:
            plot_season_gmt(datadir, runid, mode, precip_thresh)
            plot_season_lst(datadir, runid, mode, precip_thresh)
            plot_diurnal_cycle(datadir, runid, mode, precip_thresh)

        gen_animations(datadir, runid, precip_thresh)

if __name__ == '__main__':
    basepath = sys.argv[1]
    runid = sys.argv[2]
    main(basepath, runid)
