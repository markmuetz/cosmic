import iris

from remake import Task, TaskControl, remake_task_control

from cosmic.config import PATHS, CONSTRAINT_ASIA

BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '02:30',
}


def extract_asia(inputs, outputs):
    asia_cubes = iris.load([str(i) for i in inputs], constraints=CONSTRAINT_ASIA)
    iris.save(asia_cubes, str(outputs[0]))


def regrid_extract_asia(inputs, outputs):
    constraints = iris.Constraint(name='surface_altitude') & CONSTRAINT_ASIA
    target_cube = iris.load_cube(str(inputs[0]), constraint=constraints)

    asia_cubes = iris.load([str(i) for i in inputs[1:]], constraints=CONSTRAINT_ASIA)

    u_cubes = asia_cubes.extract('x_wind')
    v_cubes = asia_cubes.extract('y_wind')

    fixed_cubes = {}
    # Pain in the butt, but one of the cubes is only 2D and the rest are 3D.
    # Also, some attributes are names e.g. time_1 for some reason.
    # Sort both of these out.
    for name, cubes in zip(['x_wind', 'y_wind'], [u_cubes, v_cubes]):
        cube_2ds = [c for c in cubes if c.ndim == 2]
        cube_3ds = [c for c in cubes if c.ndim == 3]
        cube_3d = cube_3ds[0]
        for cube_2d in cube_2ds:
            # Not the same as cube_3d[0]! Retains time dim.
            new_cube_3d = cube_3d[:1].copy()
            new_cube_3d.coord('time').points = cube_2d.coord('time').points
            new_cube_3d[0].data = cube_2d.data
            # I think it is always the first cube -- put first for efficiency.
            # Order shouldn't matter tho.
            cube_3ds.insert(0, new_cube_3d)

        for cube in cube_3ds:
            cube.var_name = name
            cube.coord('time').var_name = 'time'
            cube.coord('forecast_period').var_name = 'forecast_period'
        fixed_cubes[name] = iris.cube.CubeList(cube_3ds).concatenate_cube()

    for cube in [target_cube] + [c for c in fixed_cubes.values()]:
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    print(fixed_cubes)
    scheme = iris.analysis.Linear()
    regridded_asia_cubes = iris.cube.CubeList([c.regrid(target_cube, scheme)
                                               for c in fixed_cubes.values()])
    iris.save(regridded_asia_cubes, str(outputs[0]))


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)

    for model in ['u-al508', 'u-ak543']:
        year = 2006
        for month in [6, 7, 8]:
            nc_files = [PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.orog']
            nc_files.extend(sorted((PATHS['datadir'] / model / 'ap9.pp' /
                                    f'surface_wind_{year}{month:02}')
                                   .glob(f'{model[2:]}a.p9????????.nc')))
            outpath = (PATHS['datadir'] / model / 'ap9.pp' /
                       f'surface_wind_{year}{month:02}' / f'{model[2:]}a.p9{year}{month:02}.asia.nc')
            tc.add(Task(regrid_extract_asia, nc_files, [outpath]))

    return tc
