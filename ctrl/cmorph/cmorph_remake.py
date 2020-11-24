import calendar
import itertools
from pathlib import Path

import iris

from cosmic.datasets.cmorph.cmorph_downloader import CmorphDownloader
from cosmic.datasets.cmorph.cmorph_convert import convert_cmorph_8km_30min_to_netcdf4_month
from cosmic.datasets.cmorph.cmorph_convert import extract_asia_8km_30min
from cosmic.util import load_module, filepath_regrid

from remake import TaskControl, Task, remake_task_control, remake_required


BASEDIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data/8km-30min')


@remake_required(depends_on=[CmorphDownloader])
def download_year(inputs, outputs, year, month):
    print(f'{year}, {month}')
    dl = CmorphDownloader(BASEDIR / 'raw' / f'precip_{year}{month:02}')
    end_year = year
    end_month = month + 1
    if end_month == 13:
        end_year += 1
        end_month = 1
    dl.download_8km_30min(year, month, outputs[0])


@remake_required(depends_on=[convert_cmorph_8km_30min_to_netcdf4_month])
def convert_year_month(inputs, outputs, year, month):
    print(f'{year}, {month}')
    convert_cmorph_8km_30min_to_netcdf4_month(inputs[0], outputs, year, month)


@remake_required(depends_on=[extract_asia_8km_30min])
def extract_year_month(inputs, outputs, year, month):
    print(f'{year}, {month}')
    extract_asia_8km_30min(inputs, outputs[0], year, month)


def regrid_asia(inputs, outputs):
    target_filepath = inputs['target']
    cmorph_filepath = inputs['cmorph']
    output_filepath = outputs[0]
    print(f'Regrid {cmorph_filepath} -> {output_filepath}')
    print(f'  using {target_filepath} resolution')

    coarse_cube = filepath_regrid(cmorph_filepath, target_filepath)
    iris.save(coarse_cube, str(output_filepath), zlib=True)


@remake_task_control
def gen_task_ctrl():
    years = [2019]
    months = range(1, 13)

    task_ctrl = TaskControl(__file__)
    for year, month in itertools.product(years, months):
        filename = (BASEDIR / 'raw' /
                    f'precip_{year}{month:02}' /
                    f'CMORPH_V1.0_ADJ_8km-30min_{year}{month:02}.tar')
        task_ctrl.add(Task(download_year, [], [filename], func_args=(year, month)))

        raw_filename = (BASEDIR / 'raw' /
                        f'precip_{year}{month:02}' /
                        f'CMORPH_V1.0_ADJ_8km-30min_{year}{month:02}.tar')
        nc_filenames = {day: (BASEDIR /
                              f'precip_{year}{month:02}' /
                              f'cmorph_ppt_{year}{month:02}{day:02}.nc')
                        for day in range(1, calendar.monthrange(year, month)[1] + 1)}
        task_ctrl.add(Task(convert_year_month,
                           [raw_filename],
                           nc_filenames,
                           func_args=(year, month)))

        nc_asia_filename = (BASEDIR / f'precip_{year}{month:02}' /
                            f'cmorph_ppt_{year}{month:02}.asia.nc')
        task_ctrl.add(Task(extract_year_month,
                           nc_filenames,
                           [nc_asia_filename],
                           func_args=(year, month)))

        regrid_inputs = {}
        regrid_inputs['target'] = Path(f'/gws/nopw/j04/cosmic/mmuetz/data/u-al508/ap9.pp/precip_200501/al508a.p9200501.asia_precip.nc')
        regrid_inputs['cmorph'] = BASEDIR / f'precip_{year}{month:02}/cmorph_ppt_{year}{month:02}.asia.nc'
        regrid_output = regrid_inputs['cmorph'].parent / (regrid_inputs['cmorph'].stem + '.N1280.nc')
        task_ctrl.add(Task(regrid_asia,
                           regrid_inputs,
                           [regrid_output]))

    return task_ctrl

