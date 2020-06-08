import iris

from remake import Task, TaskControl, remake_task_control

from cosmic.config import PATHS, CONSTRAINT_ASIA


BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '02:30',
}


def extract_asia(inputs, outputs):
    asia_cubes = iris.load(str(i for i in inputs), constraints=CONSTRAINT_ASIA)
    iris.save(asia_cubes, str(outputs[0]))


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    nc_files = sorted((PATHS['datadir'] / 'u-al508' / 'ap8.pp' / 'lowlevel_wind_200901').glob('al508a.p8200901??.nc'))
    outpath = PATHS['datadir'] / 'u-al508' / 'ap8.pp' / 'lowlevel_wind_200901' / 'al508a.p8200901.asia.nc'
    tc.add(Task(extract_asia, nc_files, [outpath]))
    return tc

