# coding: utf-8
import datetime as dt
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from metpy.interpolate import interpolate_to_grid
import numpy as np

from read_station_info import read_station_info
from read_station_precip_data import read_station_precip_data


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


def plot_precip_station_jja_cressman(df_precip_station_jja):
    plt.figure('precip_gauge_china_2419_cressman')
    plt.clf()
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('/home/markmuetz/cosmic_ctrl/WP2_analysis/cmap_data/li2018_fig2_cb1.pkl')

    sorted_df_precip_station_jja = df_precip_station_jja.sort_values(by='precip')


    gx, gy, img = interpolate_to_grid(sorted_df_precip_station_jja.lon, 
                                      sorted_df_precip_station_jja.lat, 
                                      sorted_df_precip_station_jja.precip, 
                                      interp_type='cressman', minimum_neighbors=1,
                                      hres=0.12, search_radius=0.35)
    img = np.ma.masked_where(np.isnan(img), img)
    lon_min = sorted_df_precip_station_jja.lon.min()
    lon_max = sorted_df_precip_station_jja.lon.max()
    lat_min = sorted_df_precip_station_jja.lat.min()
    lat_max = sorted_df_precip_station_jja.lat.max()
    extent = (lon_min, lon_max, lat_min, lat_max)

    # plt.imshow(img, origin='lower', cmap=cmap, norm=norm, extent=extent, interpolation='bilinear')
    plt.imshow(img, origin='lower', cmap=cmap, norm=norm, extent=extent)
    plt.xlim((97.5, 125))
    plt.ylim((18, 41))


if __name__ == '__main__':
    try:
        df_station_info
        df_precip
    except NameError:
        df_station_info = read_station_info()
        df_precip = read_station_precip_data()
        df_precip_jja = df_precip[(df_precip.datetime >= dt.datetime(2009, 6, 1)) & (df_precip.datetime <= dt.datetime(2009, 8, 31))]
        df_precip_station_jja = pd.merge(df_station_info, df_precip_jja.groupby('station_id').mean(), on='station_id')

    plot_precip_station_jja(df_precip_station_jja)
    plot_precip_station_jja_cressman(df_precip_station_jja)


