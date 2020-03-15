from hashlib import sha1


def task_unique_name_from_fn_args_kwargs(fn, args, kwargs):
    task_str = (fn.__code__.co_name +
                ''.join(str(a) for a in args) +
                ''.join(str(k) + str(v) for k, v in kwargs.items()))
    task_unique_filename = sha1(task_str.encode()).hexdigest() + '.task'
    return task_unique_filename


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

