import sys
from pathlib import Path
from hashlib import sha1
import importlib.util
import pickle
import subprocess as sp
from typing import List, Union
import itertools
from timeit import default_timer as timer

import iris
import iris.coords
import matplotlib as mpl
import numpy as np
from scipy import stats

from cosmic.cosmic_errors import CosmicError


def rmse_mask_out_nan(a1, a2):
    nan_mask = np.isnan(a1)
    nan_mask |= np.isnan(a2)
    return rmse(a1[~nan_mask], a2[~nan_mask])


def mae_mask_out_nan(a1, a2):
    nan_mask = np.isnan(a1)
    nan_mask |= np.isnan(a2)
    return mae(a1[~nan_mask], a2[~nan_mask])


def circular_rmse_mask_out_nan(a1, a2):
    nan_mask = np.isnan(a1)
    nan_mask |= np.isnan(a2)
    return circular_rmse(a1[~nan_mask], a2[~nan_mask])


def vrmse_mask_out_nan(a1, a2):
    nan_mask = np.isnan(a1)
    nan_mask |= np.isnan(a2)
    # Need a 1D mask, otherwise choosing values will drop a1/a2 to a 1D array.
    nan_mask = nan_mask.sum(axis=1).astype(bool)
    return vrmse(a1[~nan_mask], a2[~nan_mask])


def mae(a1, a2):
    return np.abs(a1 - a2).mean()


def rmse(a1, a2):
    return np.sqrt(((a1 - a2)**2).mean())


def mae(a1, a2):
    return np.abs(a1 - a2).mean()


def circular_rmse(a1, a2):
    diff = (a1 - a2) % 24
    diff[diff > 12] = (24 - diff)[diff > 12]
    return np.sqrt(((diff**2).mean()))


def vrmse(a1: np.ndarray, a2: np.ndarray) -> float:
    """Vector RMSE: See https://stats.stackexchange.com/questions/449317/calculation-of-vector-rmse

    :param a1: First array, final dimension is dimension of each vector
    :param a2: Second array, shape the same as first
    :return: Vector RMSE of arrays
    """
    assert a1.shape == a2.shape, 'Shapes do not match'
    if a1.ndim > 2:
        # Reduce dims of incoming arrays so that the resulting arrays are (N, D),
        # where N is the number of individual vectors, and D is the dimension of each vector.
        # e.g. a1.shape == (300, 200, 2) -> (60000, 2).
        a1 = a1.reshape(-1, a1.shape[-1])
        a2 = a2.reshape(-1, a2.shape[-1])
    return np.sqrt((((a1 - a2)**2).sum(axis=1)).sum() / a1.shape[0])


def weighted_vrmse(weights: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> float:
    """Vector RMSE: See https://stats.stackexchange.com/questions/449317/calculation-of-vector-rmse

    :param weights: weights to apply to each cell
    :param a1: First array, final dimension is dimension of each vector
    :param a2: Seconda array, shape the same as first
    :return: Vector RMSE of arrays
    """
    assert weights.shape == a1.shape == a2.shape, 'Shapes do not match'
    if a1.ndim > 2:
        # Reduce dims of incoming arrays so that the resulting arrays are (N, D),
        # where N is the number of individual vectors, and D is the dimension of each vector.
        # e.g. a1.shape == (300, 200, 2) -> (60000, 2).
        weights = weights.reshape(-1, weights.shape[-1])
        a1 = a1.reshape(-1, a1.shape[-1])
        a2 = a2.reshape(-1, a2.shape[-1])
    return np.sqrt(((weights * (a1 - a2)**2).sum(axis=1)).sum() / weights.sum())


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


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


def load_module(local_filename):
    module_path = Path.cwd() / local_filename
    if not module_path.exists():
        raise CosmicError(f'Module file {module_path} does not exist')

    # Make sure any local imports in the config script work.
    sys.path.append(str(module_path.parent))
    module_name = Path(local_filename).stem

    try:
        # See: https://stackoverflow.com/a/50395128/54557
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        # N.B. this did not work for me when importing file based on name.
        # Instead used above sys.path.append.
        # sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except SyntaxError as se:
        print(f'Bad syntax in config file {module_path}')
        raise

    return module


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


def calc_2d_dot_product(cubes: iris.cube.CubeList, second_cubes: iris.cube.CubeList, name: str = None) \
        -> iris.cube.Cube:
    """Calc 2D dot product of 2 cubelists.

    :param cubes: input cubes (x2)
    :param second_cubes: second input cubes (x2)
    :param name of output cube
    :return: dot product field
    """
    assert len(cubes) == len(second_cubes) == 2, 'both cubes must have length 2'
    if not name:
        name = f'dot product ' \
               f'({cubes[0].name()}, {cubes[1].name()})x({second_cubes[0].name()}, {second_cubes[1].name()})'

    result = cubes[0].copy()
    result.data = cubes[0].data * second_cubes[0].data + cubes[1].data * second_cubes[1].data
    result.rename(name)
    result.units = cubes[0].units * second_cubes[0].units
    return result


def calc_uniform_lat_lon_grad(cube: iris.cube.Cube) -> iris.cube.CubeList:
    """Use centred difference to calc gradient of cube (with uniform lat/lon coords).

    Note -- discards first and last lat as it is hard to work out dFdy here.

    :param cube: input cube with lat/lon coords as last two coords.
    :return: 2 cubes -- dF/dx, dF/dy.
    """
    R = 6371e3  # Earth radius in km.
    assert cube.coords()[-2] == cube.coord('latitude'), 'cube must have lat,lon as last two coords'
    assert cube.coords()[-1] == cube.coord('longitude'), 'cube must have lat,lon as last two coords'
    lat = cube.coord('latitude').points
    lon = cube.coord('longitude').points
    d2lat = lat[2] - lat[0]  # N.B. 2 grid points diff.
    d2lon = lon[2] - lon[0]  # N.B. 2 grid points diff.
    assert np.all(d2lat == lat[2:] - lat[:-2]), 'Cannot be used on variable grid.'
    assert np.all(d2lon == (np.roll(lon, -1) - np.roll(lon, 1)) % 360), 'Cannot be used on variable grid.'

    # 1 per lat, in metres.
    dx = R * np.cos(lat * np.pi / 180) * d2lon * np.pi / 180
    # 1 only, in metres.
    dy = R * d2lat * np.pi / 180

    # This allows the function to work on *any* cube, provided it has lat/lon as the last two dims.
    lat_slice = tuple([slice(None)] * (cube.ndim - 2) + [slice(1, -1)])
    # Used to broadcast lat (1D) into cube.data.shape (N-D).
    lat_broadcast_slice = tuple([None] * (cube.ndim - 2) + [slice(None)] + [None])

    # Both lat and lon dims are referenced rel to final dim -- allows it to work on *any* cube.
    dfdx = (np.roll(cube.data, -1, axis=-1) - np.roll(cube.data, 1, axis=-1)) / dx[lat_broadcast_slice]
    # N.B. first and last lat values are junk -- latitude is not circular. They are discarded below by using lat_slice.
    dfdy = (np.roll(cube.data, -1, axis=-2) - np.roll(cube.data, 1, axis=-2)) / dy

    ret_cubes = []
    # Make iris cubes with correct name, data and units.
    for var, data in zip(['x', 'y'], [dfdx, dfdy]):
        new_cube = cube[lat_slice].copy()
        new_cube.rename(f'delta {cube.name()} by delta {var}')
        new_cube.data = data[lat_slice]
        new_cube.units = cube.units / 'm'
        ret_cubes.append(new_cube)

    return iris.cube.CubeList(ret_cubes)


class CalcLatLonDistanceMask:
    """Class to allow quick calculation of closeness on lat/lon grid.

    Leverages fact that, for a uniform lat/lon grid, you only need to calculate closeness
    once for each latitude. Caches this result in memory, and rolls this as necessary based
    on the longitude to produce the given mask for a lat/lon index.
    Also, caches the result to a file for quicker subsequent use."""
    def __init__(self, Lat: np.ndarray, Lon: np.ndarray,
                 dist_thresh: int = 100, circular_lon: bool = True, cache_dir: str = '.'):
        self.Lat = Lat
        self.Lon = Lon
        self.dist_thresh = dist_thresh
        self.circular_lon = circular_lon

        # Calculate a cache key based on the hash of all the arguments.
        sha1hash = sha1()
        sha1hash.update(Lat.tobytes())
        sha1hash.update(Lon.tobytes())
        sha1hash.update(bytes(dist_thresh.to_bytes(8, byteorder='big')))
        sha1hash.update(bytes([circular_lon]))
        cache_key = Path(cache_dir) / Path(f'.cache_mask.{sha1hash.hexdigest()}.npy')

        if cache_key.exists():
            print(f'loading cache_mask from file {cache_key}')
            self.cache_mask = np.load(cache_key)
        else:
            print(f'generating cache_mask and saving to {cache_key}')
            self.cache_mask = np.zeros((Lat.shape[0], Lat.shape[0], Lat.shape[1]), dtype=bool)
            R = 6371.  # Earth radius in km.
            # Convert to radians.
            phi2, theta2 = [np.pi / 180 * v for v in (Lat, Lon)]
            for ilat, lat in enumerate(Lat[:, 0]):
                print(ilat)
                phi1 = np.pi / 180 * lat
                theta1 = np.pi / 180 * Lon[0, Lon.shape[1] // 2]
                # Accurate great-circle distance:
                # https://en.wikipedia.org/wiki/Great-circle_distance#Formulae
                self.cache_mask[ilat] = (R * np.arccos(np.cos(phi1) * np.cos(phi2) * np.cos(theta2 - theta1) +
                                                       np.sin(phi1) * np.sin(phi2))) < dist_thresh
            np.save(cache_key, self.cache_mask)

    def calc_mask(self, ilat, ilon):
        # TODO: the + 1 means that this method exactly matches fn calc_close_to_mask
        # TODO: below, but I'm not sure why it's needed.
        ilon_roll = ilon + self.Lon.shape[1] // 2 + 1
        if self.circular_lon:
            return np.roll(self.cache_mask[ilat], ilon_roll, axis=1)
        else:
            circ_mask = np.roll(self.cache_mask[ilat], ilon_roll, axis=1)
            # Using roll means that it will expand a point near the boundary over it.
            # Fix this by zeroing these values.
            if ilon < self.Lon.shape[1] // 2:
                circ_mask[:, self.Lon.shape[1] // 2 + ilon:] = 0
            elif ilon > self.Lon.shape[1] // 2:
                circ_mask[:, :ilon - self.Lon.shape[1] // 2] = 0
            return circ_mask

    def calc_close_to_mask(self, mask: iris.cube.Cube) -> iris.cube.Cube:
        start = timer()
        close_to_mask = mask.copy()
        close_to_mask.data = np.zeros(close_to_mask.shape, dtype=bool)

        latlon_indices = np.where(mask.data)
        # Way faster than a loop over all values, and checking the mask.
        for i, (ilat, jlat) in enumerate(zip(*latlon_indices)):
            if (i + 1) % (latlon_indices[0].size // 10) == 0:
                print(f'{i + 1}/{latlon_indices[0].size} ({100 * (i + 1)/latlon_indices[0].size:.1f}%)')
            close_to_mask.data |= self.calc_mask(ilat, jlat)

        print(f'Completed in {timer() - start:.1f}s')
        return close_to_mask


def calc_latlon_distance(lat1, lat2, lon1, lon2):
    """Accurate lat lon distance in km.

    :param lat1: first lat
    :param lat2: second lat
    :param lon1: first lon
    :param lon2: second lon
    :return: distance in km.
    """
    R = 6371.  # Earth radius in km.
    theta1, theta2, phi1, phi2 = [np.pi / 180 * v for v in (lon1, lon2, lat1, lat2)]

    return R * np.arccos(np.cos(phi1) * np.cos(phi2) * np.cos(theta2 - theta1) + np.sin(phi1) * np.sin(phi2))


def calc_fast_latlon_distance(lat1, lat2, lon1, lon2):
    """Fast lat lon distance in km.

    relies on small angle approx.
    Adapted from B. Vanniere (distance(...))

    :param lat1: first lat
    :param lat2: second lat
    :param lon1: first lon
    :param lon2: second lon
    :return: distance in km.
    """
    R = 6371.
    dlon = (lon2 - lon1)
    dlon = (dlon + 180.) % 360. - 180.
    x = dlon * np.cos(0.5 * (lat2 + lat1) * np.pi / 180.)
    y = lat2 - lat1
    d = np.pi / 180. * R * np.sqrt(x * x + y * y)
    return d


def calc_close_to_mask(mask: iris.cube.Cube, dist_thresh: int = 100, method: str = 'fast') -> iris.cube.Cube:
    """Calc all points that are within a certain distance of masked (True) values.

    Adapted from B. Vanniere (distance_orography(...))
    Note, this is slow for large cubes (e.g. N1280).

    :param mask: input mask
    :param dist_thresh: threshold to use for distance (in km)
    :param method: fast or accurate
    :return: expanded mask based on threshold
    """
    if method not in ['fast', 'accurate']:
        raise ValueError(f'method must be one of fast, accurate')
    start = timer()
    close_to_mask = mask.copy()

    close_to_mask.data = np.zeros(close_to_mask.shape, dtype=int)

    lat = mask.coord('latitude').points
    lon = mask.coord('longitude').points
    Lon, Lat = np.meshgrid(lon, lat)

    calc_dist = calc_latlon_distance if method == 'accurate' else calc_fast_latlon_distance
    # for i, j in itertools.product(range(len(lat)), range(len(lon))):
    #     if mask.data[i, j]:
    # Way faster than a loop over all values, and checking the mask.
    for i, j in zip(*np.where(mask.data)):
        close_to_mask.data += calc_dist(lat[i], Lat, lon[j], Lon) < dist_thresh
    close_to_mask.data = close_to_mask.data.astype(bool)
    print(f'Completed in {timer() - start:.1f}s')
    return close_to_mask


def get_extent_from_cube(cube):
    """ Proper way to work out extent for imshow.

    lat.points contains centres of each cell.
    bounds contains the boundary of the pixel - this is what imshow should take.
    """
    lon = cube.coord('longitude')
    lat = cube.coord('latitude')
    if not lat.has_bounds():
        lat.guess_bounds()
    if not lon.has_bounds():
        lon.guess_bounds()
    lon_min, lon_max = lon.bounds[0, 0], lon.bounds[-1, 1]
    lat_min, lat_max = lat.bounds[0, 0], lat.bounds[-1, 1]
    extent = (lon_min, lon_max, lat_min, lat_max)
    return extent