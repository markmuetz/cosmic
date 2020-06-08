from remake import Task, TaskControl, remake_task_control

from cosmic.config import PATHS
from cosmic.processing.convert_pp_to_nc import convert_pp_to_nc


BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '02:30',
}


def convert_wrapper(inputs, outputs):
    convert_pp_to_nc(inputs[0], outputs[0])


@remake_task_control
def gen_task_ctrl():
    tc = TaskControl(__file__)
    pp_files = list((PATHS['datadir'] / 'u-al508' / 'ap8.pp' / 'lowlevel_wind_200901').glob('al508a.p8200901??.pp'))
    for pp_file in pp_files:
        tc.add(Task(convert_wrapper, [pp_file], [pp_file.with_suffix('.nc')]))
    return tc

