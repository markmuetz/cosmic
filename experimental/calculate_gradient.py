#!/bin/python
"""Taken from /home/users/mmuetz/scripts/bvanniere/sinclair1994_orog_precip_scripts/calculate_gradient

Originally written by B Vanniere"""

import sys
import iris
import numpy as np


def calc_grad(FF, M=None):
    '''
    Usage :
    F : Field given on the latlon C-grid
    M : land mask (1 atmos, 0 land)

    See further description
    By default mask M is set to 1
    '''

    print("Now calculating gradient")
    lon = FF.coord('longitude').points
    lat = FF.coord('latitude').points
    F = FF.data.copy()

    shape = FF.shape
    dFdx = FF.copy()
    dFdx.data = np.zeros(shape)
    dFdy = FF.copy()
    dFdy.data = np.zeros(shape)

    if M is None:
        M = np.ones(F.shape)

    if F.ndim == 2:
        dFdx.data[:, :], dFdy.data[:, :] = calc_grad_2d(lon, lat, F[:, :], M[:, :])
    elif F.ndim == 3:
        for t in range(shape[0]):
            dFdx.data[t, :, :], dFdy.data[t, :, :] = calc_grad_2d(lon, lat, F[t, :, :], M[t, :, :])
    elif F.ndim == 4:
        for t in range(shape[0]):
            for z in range(shape[1]):
                dFdx.data[t, z, :, :], dFdy.data[t, z, :, :] = calc_grad_2d(lon, lat, F[t, z, :, :], M[t, z, :, :])

    return dFdx, dFdy


def calc_grad_2d(lon, lat, F, M=None):
    '''
        Usage :
        F : Field given on the latlon C-grid
        M : land mask (1 atmos, 0 land) 
        
        Description : 
        This function calculate gradient on regular grids only. 

        Written by B. Vanniere on 07 Nov 2016, b.vanniere@reading.ac.uk
        '''

    if M is None:
        M = np.ones(F.shape)

    Nx = len(lon);  # increase eastward
    Ny = len(lat);  # increases northward
    R = 6371. * 1000.;  # Earth's radius

    new_shape = (Ny + 2, Nx + 2)
    lon_ext = np.zeros(Nx + 2)
    lat_ext = np.zeros(Ny + 2)
    M_ext = np.zeros(new_shape)
    F_ext = np.zeros(new_shape)
    dFdx = np.zeros(new_shape)
    dFdy = np.zeros(new_shape)

    lon_ext[1:-1] = lon
    lat_ext[1:-1] = lat
    M_ext[1:-1, 1:-1] = M
    F_ext[1:-1, 1:-1] = F

    # Periodic boundaries
    lat_ext[0] = lat[0]
    lat_ext[-1] = lat[-1]
    lon_ext[0] = lon[-1]
    lon_ext[-1] = lon[0]
    M_ext[1:-1, 0] = M[:, -1]
    M_ext[1:-1, 0] = M[:, -1]
    M_ext[1:-1, -1] = M[:, 0]
    F_ext[1:-1, 0] = F[:, -1]
    F_ext[1:-1, -1] = F[:, 0]

    # Resized matrices
    lon = lon_ext
    lat = lat_ext
    F = F_ext
    M = M_ext

    dtr = np.pi / 180.

    for j in range(1, new_shape[1] - 1):
        for i in range(1, new_shape[0] - 1):
            dlon = lon[j + 1] - lon[j - 1]
            dlon = (dlon + 180.) % 360. - 180.
            dx = R * np.cos(lat[i] * dtr) * (dlon) * dtr;
            dy = R * (lat[i + 1] - lat[i - 1]) * dtr;
            dFdx[i, j] = (F[i, j + 1] - F[i, j - 1]) / dx * M[i, j] * M[i, j - 1] * M[i, j + 1];
            dFdy[i, j] = (F[i + 1, j] - F[i - 1, j]) / dy * M[i, j] * M[i + 1, j] * M[i - 1, j];

    dFdx = dFdx[1:-1, 1:-1]
    dFdy = dFdy[1:-1, 1:-1]

    return dFdx, dFdy
