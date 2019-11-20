from argparse import ArgumentParser
import datetime as dt
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from metpy.interpolate import interpolate_to_grid
import numpy as np

from basmati.hydrosheds import load_hydrobasins_geodataframe
from cosmic.util import load_cmap_data, build_raster_from_lon_lat

BASEDIR = Path('/home/markmuetz/Datasets/gauge_china2419/pre/SURF_CLI_CHN_PRE_MUT_HOMO/SURF_CLI_CHN_PRE_MUT_HOMO')


def plot_precip_station_jja_scatter(df_precip_station_jja):
    plt.figure('precip_gauge_china_2419_scatter')
    plt.clf()
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')
    sorted_df_precip_station_jja = df_precip_station_jja.sort_values(by='precip')
    plt.scatter(sorted_df_precip_station_jja.lon, sorted_df_precip_station_jja.lat, c=sorted_df_precip_station_jja.precip, s=300, cmap=cmap, norm=norm)
    plt.xlim((97.5, 125))
    plt.ylim((18, 41))


def plot_precip_station_jja_cressman(ax, df_precip_station_jja, stretch_lat, search_rad, grid_spacing, 
                                     **kwargs):
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')

    if stretch_lat:
        lat = df_precip_station_jja.lat * 1.5
    else:
        lat = df_precip_station_jja.lat

    gx, gy, griddata = interpolate_to_grid(df_precip_station_jja.lon, 
                                           lat, 
                                           df_precip_station_jja.precip, 
                                           interp_type='cressman', minimum_neighbors=1,
                                           hres=grid_spacing, search_radius=search_rad)
    griddata = np.ma.masked_where(np.isnan(griddata), griddata)
    lon_min = df_precip_station_jja.lon.min()
    lon_max = df_precip_station_jja.lon.max()
    lat_min = df_precip_station_jja.lat.min()
    lat_max = df_precip_station_jja.lat.max()
    extent = (lon_min, lon_max, lat_min, lat_max)

    # Use to mask out ocean.
    hb = load_hydrobasins_geodataframe('/home/markmuetz/HydroSHEDS', 'as', [1])
    raster = build_raster_from_lon_lat(lon_min, lon_max, lat_min, lat_max, 
                                       griddata.shape[1], griddata.shape[0], hb)

    griddata = np.ma.masked_array(griddata, raster == 0)
    im = ax.imshow(griddata, origin='lower', 
                   cmap=cmap, norm=norm, extent=extent, interpolation='bilinear')
    # im = ax.imshow(img, origin='lower', cmap=cmap, norm=norm, extent=extent)
    # plt.colorbar(im, label='precip. (mm day$^{-1}$)',
                 # orientation='horizontal', norm=norm, spacing='uniform', **cbar_kwargs)
    return im


def plot_li2018_fig2a_reproduction(df_precip_station_jja, **kwargs):
    # plt.figure('precip_gauge_china_2419_cressman', figsize=(10, 8))
    fig = plt.figure('precip_gauge_china_2419_cressman')
    fig.set_size_inches(10.5, 9.22)
    plt.clf()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')

    # plot_precip_station_jja(df_precip_station_jja)
    im = plot_precip_station_jja_cressman(ax, df_precip_station_jja, **kwargs)

    ax.set_xlim((97.5, 125))
    ax.set_ylim((18, 41))

    ax.set_xticks(np.linspace(100, 120, 3))
    ax.set_xticks(np.linspace(98, 124, 14), minor=True)

    ax.set_yticks(np.linspace(20, 40, 3))
    ax.set_yticks(np.linspace(18, 40, 12), minor=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True, which='both')

    kwargs_str = '.'.join([f'{k}-{i}' for k, i in kwargs.items()])
    plt.tight_layout()
    plt.subplots_adjust(top=0.898, left=0.15, right=0.965, bottom=0.097)
    plt.title(kwargs_str)
    plt.savefig(f'figs/gauge_data/precip_station_jja_cressman.{kwargs_str}.png')


def load_data(basedir):
    df_station_info = pd.read_hdf(basedir / 'station_data.hdf', 'station_info')
    df_precip = pd.read_hdf(basedir / 'station_data.hdf', 'precip')
    df_precip.precip.replace(-999, np.NaN)
    df_precip_jja = df_precip[(df_precip.datetime >= dt.datetime(2009, 6, 1)) & (df_precip.datetime <= dt.datetime(2009, 8, 31))]
    df_precip_station_jja = pd.merge(df_station_info, df_precip_jja.groupby('station_id').mean(), on='station_id')
    return df_station_info, df_precip, df_precip_jja, df_precip_station_jja


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--stretch-lat', action='store_true')
    parser.add_argument('--search-rad', type=float, default=0.48)
    parser.add_argument('--grid-spacing', type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    kwargs = vars(args)
    basedir = Path(kwargs.pop('basedir'))

    try:
        # Save reloading by using 
        #    In [1]: run -i plot_gauge_data.py 
        # from within ipython.
        df_precip_station_jja
    except NameError:
        print('loading data')
        df_station_info, df_precip, df_precip_jja, df_precip_station_jja = load_data(basedir)

    plot_li2018_fig2a_reproduction(df_precip_station_jja, **kwargs)

