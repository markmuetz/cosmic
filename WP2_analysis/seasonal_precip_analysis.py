import sys
import iris

import cosmic.WP2.seasonal_precip_analysis as spa
from remake import Task, TaskControl

from paths import PATHS


DATASETS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
    'u-ak543',
    'u-al508',
    'cmorph',
]


def fmt_year_month(year, month):
    return f'{year}{month:02}'


def gen_seasonal_precip_analysis(inputs, outputs, season, precip_thresh):
    season_cube = iris.load([str(p) for p in inputs]).concatenate_cube()
    num_per_day = 48

    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                           num_per_day=num_per_day,
                                                           convert_kgpm2ps1_to_mmphr=False,
                                                           calc_method='low_mem')

    iris.save(analysis_cubes, str(outputs[0]))


class CmorphSpaTask(Task):
    def __init__(self, start_year_month, end_year_month, resolution, precip_thresh, cmorph_dataset, season):
        datadir = PATHS['datadir'] / 'cmorph_data' / cmorph_dataset
        if resolution:
            file_tpl = 'cmorph_ppt_{year}{month:02}.asia.{resolution}.nc'
            nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                    file_tpl=file_tpl, resolution=resolution)
        else:
            file_tpl = 'cmorph_ppt_{year}{month:02}.asia.nc'
            nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                    file_tpl=file_tpl)
        daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
        thresh_text = str(precip_thresh).replace('.', 'p')
        output_path = (PATHS['datadir'] / 'cmorph_data' /
                       f'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_{thresh_text}.{resolution}.nc')
        super().__init__(gen_seasonal_precip_analysis, nc_season, [output_path], func_args=(precip_thresh,))


def gen_task_ctrl():
    task_ctrl = TaskControl(enable_file_task_content_checks=True, dotremake_dir='.remake.seasonal_precip_analysis')

    start_year_month = (1998, 1)
    end_year_month = (2018, 12)
    resolution = 'N1280'
    precip_thresh = 0.1
    cmorph_dataset = '8km-30min'
    season = 'jja'

    task_ctrl.add(CmorphSpaTask(start_year_month, end_year_month, resolution, precip_thresh, cmorph_dataset, season))
    return task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl()
    if len(sys.argv) == 2 and sys.argv[1] == 'finalize':
        task_ctrl.finalize()
    elif len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.finalize()
        task_ctrl.run()
    else:
        task_ctrl.build_task_DAG()
