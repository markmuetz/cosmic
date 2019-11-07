import iris


def regrid(input_filepath, target_filepath):
    input_cubes = iris.load(str(input_filepath))
    target_cubes = iris.load(str(target_filepath))

    input_cube.coord('latitude').guess_bounds()
    input_cube.coord('longitude').guess_bounds()
    target_cube.coord('latitude').guess_bounds()
    target_cube.coord('longitude').guess_bounds()

    target_cube.coord('latitude').coord_system = input_cube.coord('latitude').coord_system
    target_cube.coord('longitude').coord_system = input_cube.coord('longitude').coord_system

    scheme = iris.analysis.AreaWeighted(mdtol=0.5)
    return input_cube.regrid(target_cube, scheme)
