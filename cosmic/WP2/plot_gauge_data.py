from argparse import ArgumentParser
import datetime as dt
from pathlib import Path
import pickle

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from metpy.interpolate import interpolate_to_grid
import numpy as np
import rasterio

from basmati.hydrosheds import load_hydrobasins_geodataframe
from basmati.utils import build_raster_from_geometries

BASEDIR = Path('/home/markmuetz/Datasets/gauge_china2419/pre/SURF_CLI_CHN_PRE_MUT_HOMO/SURF_CLI_CHN_PRE_MUT_HOMO')


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


def plot_precip_station_jja(df_precip_station_jja):
    plt.figure('precip_gauge_china_2419_scatter')
    plt.clf()
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('/home/markmuetz/cosmic_ctrl/WP2_analysis/cmap_data/li2018_fig2_cb1.pkl')
    sorted_df_precip_station_jja = df_precip_station_jja.sort_values(by='precip')
    plt.scatter(sorted_df_precip_station_jja.lon, sorted_df_precip_station_jja.lat, c=sorted_df_precip_station_jja.precip, s=300, cmap=cmap, norm=norm)
    plt.xlim((97.5, 125))
    plt.ylim((18, 41))


def build_raster(lon_min, lon_max, lat_min, lat_max, nlon, nlat, hb):
    # TODO: DRY!
    print('Build raster')
    scale_lon = (lon_max - lon_min) / (nlon - 1)
    scale_lat = (lat_max - lat_min) / (nlat - 1)

    affine_tx = rasterio.transform.Affine(scale_lon, 0, lon_min,
                                          0, scale_lat, lat_min)
    raster = build_raster_from_geometries(hb.geometry,
                                          (nlat, nlon),
                                          affine_tx)
    return raster


def plot_precip_station_jja_cressman(df_precip_station_jja):

    plt.figure('precip_gauge_china_2419_cressman')
    plt.clf()
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('/home/markmuetz/cosmic_ctrl/WP2_analysis/cmap_data/li2018_fig2_cb1.pkl')

    sorted_df_precip_station_jja = df_precip_station_jja.sort_values(by='precip')


    gx, gy, img = interpolate_to_grid(sorted_df_precip_station_jja.lon, 
                                      sorted_df_precip_station_jja.lat, 
                                      sorted_df_precip_station_jja.precip, 
                                      interp_type='cressman', minimum_neighbors=1,
                                      hres=0.12, search_radius=0.4)
    img = np.ma.masked_where(np.isnan(img), img)
    lon_min = sorted_df_precip_station_jja.lon.min()
    lon_max = sorted_df_precip_station_jja.lon.max()
    lat_min = sorted_df_precip_station_jja.lat.min()
    lat_max = sorted_df_precip_station_jja.lat.max()
    extent = (lon_min, lon_max, lat_min, lat_max)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')

    hb = load_hydrobasins_geodataframe('/home/markmuetz/HydroSHEDS', 'as', [1])
    raster = build_raster(lon_min, lon_max, lat_min, lat_max, img.shape[1], img.shape[0], hb)

    img = np.ma.masked_array(img, raster == 0)
    im = ax.imshow(img, origin='lower', cmap=cmap, norm=norm, extent=extent, interpolation='bilinear')
    # im = ax.imshow(img, origin='lower', cmap=cmap, norm=norm, extent=extent)
    ax.set_xlim((97.5, 125))
    ax.set_ylim((18, 41))
    plt.colorbar(im, label='precip. (mm day^{-1})',
                 orientation='horizontal', norm=norm, spacing='uniform', **cbar_kwargs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('basedir', action='store_const', const=BASEDIR)
    args = parser.parse_args()
    basedir = Path(args.basedir)

    try:
        df_station_info
        df_precip
    except NameError:
        df_station_info = pd.read_hdf(basedir / 'station_data.hdf', 'station_info')
        df_precip = pd.read_hdf(basedir / 'station_data.hdf', 'precip')
        df_precip_jja = df_precip[(df_precip.datetime >= dt.datetime(2009, 6, 1)) & (df_precip.datetime <= dt.datetime(2009, 8, 31))]
        df_precip_station_jja = pd.merge(df_station_info, df_precip_jja.groupby('station_id').mean(), on='station_id')

    # plot_precip_station_jja(df_precip_station_jja)
    plot_precip_station_jja_cressman(df_precip_station_jja)
    plt.show()


