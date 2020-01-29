import sys
from pathlib import Path
import importlib.util
import pickle
import subprocess as sp
from typing import List, Union
import itertools

import iris
import matplotlib as mpl
import numpy as np
import rasterio
from scipy import stats

from basmati.utils import build_raster_from_geometries

from cosmic.cosmic_errors import CosmicError


def regrid(input_cube, target_cube, scheme=iris.analysis.AreaWeighted(mdtol=0.5)):
    for cube in [input_cube, target_cube]:
        for latlon in ['latitude', 'longitude']:
            if not cube.coord(latlon).has_bounds():
                cube.coord(latlon).guess_bounds()
            if not cube.coord(latlon).coord_system:
                cube.coord(latlon).coord_system = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

    return input_cube.regrid(target_cube, scheme)


def filepath_regrid(input_filepath, target_filepath, scheme=iris.analysis.AreaWeighted(mdtol=0.5)):
    input_cube = iris.load_cube(str(input_filepath))
    target_cube = iris.load_cube(str(target_filepath))
    return regrid(input_cube, target_cube, scheme)


def daily_circular_mean(arr, axis=None):
    """Calculate the mean time (in hr) of a given input array, along any axis

    See: https://en.wikipedia.org/wiki/Mean_of_circular_quantities.
    """
    theta = (arr % 24) / 24 * 2 * np.pi
    arr_x = np.sin(theta)
    arr_y = np.cos(theta)
    theta_mean = np.arctan2(arr_x.mean(axis=axis), arr_y.mean(axis=axis))
    return (theta_mean / (2 * np.pi) * 24) % 24


def build_raster_from_lon_lat(lon_min, lon_max, lat_min, lat_max, nlon, nlat, hb):
    scale_lon = (lon_max - lon_min) / (nlon - 1)
    scale_lat = (lat_max - lat_min) / (nlat - 1)

    affine_tx = rasterio.transform.Affine(scale_lon, 0, lon_min,
                                          0, scale_lat, lat_min)
    raster = build_raster_from_geometries(hb.geometry,
                                          (nlat, nlon),
                                          affine_tx)
    return raster


def build_raster_from_cube(cube, hb):
    nlat = cube.shape[-2]
    nlon = cube.shape[-1]
    lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
    lat_min, lat_max = cube.coord('latitude').points[[0, -1]]
    return build_raster_from_lon_lat(lon_min, lon_max, lat_min, lat_max, nlon, nlat, hb)


def build_raster_cube_from_cube(cube, hb, name):
    nlat = cube.shape[-2]
    nlon = cube.shape[-1]
    lon_min, lon_max = cube.coord('longitude').points[[0, -1]]
    lat_min, lat_max = cube.coord('latitude').points[[0, -1]]
    raster = build_raster_from_lon_lat(lon_min, lon_max, lat_min, lat_max, nlon, nlat, hb)
    raster_cube = iris.cube.Cube(raster, long_name=f'{name}', units='-',
                                 dim_coords_and_dims=[(cube.coord('latitude'), 0),
                                                      (cube.coord('longitude'), 1)])
    return raster_cube


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


def load_config(local_filename):
    config_filepath = Path.cwd() / local_filename
    if not config_filepath.exists():
        raise CosmicError(f'Config file {config_filepath} does not exist')

    # Make sure any local imports in the config script work.
    sys.path.append(str(config_filepath.parent))

    try:
        spec = importlib.util.spec_from_file_location('config', config_filepath)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except SyntaxError as se:
        print(f'Bad syntax in config file {config_filepath}')
        raise

    return config


def sysrun(cmd):
    """Run a system command, returns a CompletedProcess

    raises CalledProcessError if cmd is bad.
    to access output: sysrun(cmd).stdout"""
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


def predominant_pixel_2d(arr: np.ndarray, grain_size: List[int],
                         offset: Union[List[int], str] = 'exact') -> np.ndarray:
    """Find predominant pixel (mode) of a coarse grained a 2D arr based on grain_size

    offsets the input array based on offset, so grain_size does not have to
    divide exactly into array size.

    :param arr: array to analyse
    :param grain_size: 2 value size of grain
    :param offset: one of 'exact', 'centre', 'none' or (offset0, offset1)
    :return: coarse-grained array
    """
    # Doesn't work as stats.mode does not accept a tuple as an axis arg.
    # assert arr.shape[0] % grain_size[0] == 0
    # assert arr.shape[1] % grain_size[1] == 0
    # num0 = arr.shape[0] // grain_size[0]
    # num1 = arr.shape[1] // grain_size[1]
    # return stats.mode(arr.reshape(num0, grain_size[0], num1, grain_size[1]), axis=(1, 3))
    if isinstance(offset, str):
        if offset == 'exact':
            assert arr.shape[0] % grain_size[0] == 0
            assert arr.shape[1] % grain_size[1] == 0
            offset0, offset1 = 0, 0
        elif offset == 'none':
            offset0, offset1 = 0, 0
        elif offset == 'centre':
            offset0 = (arr.shape[0] % grain_size[0]) // 2
            offset1 = (arr.shape[1] % grain_size[1]) // 2
        else:
            raise ValueError(f'Unrecognized offset: {offset}')
    else:
        offset0, offset1 = offset

    num0 = arr.shape[0] // grain_size[0]
    num1 = arr.shape[1] // grain_size[1]
    out_arr = np.zeros((num0, num1))
    for i, j in itertools.product(range(num0), range(num1)):
        arr_slice = (slice(offset0 + i * grain_size[0], offset0 + (i + 1) * grain_size[0]),
                     slice(offset1 + j * grain_size[1], offset1 + (j + 1) * grain_size[1]))
        out_arr[i, j] = stats.mode(arr[arr_slice], axis=None).mode[0]
    return out_arr

