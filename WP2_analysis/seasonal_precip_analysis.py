import sys
import iris

import cosmic.WP2.seasonal_precip_analysis as spa
from remake import Task, TaskControl

from paths import PATHS

BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '10:00',
}


def fmt_year_month(year, month):
    return f'{year}{month:02}'


def fmt_thresh_text(precip_thresh):
    return str(precip_thresh).replace('.', 'p')


def gen_seasonal_precip_analysis(inputs, outputs, season, precip_thresh, num_per_day, convert_kgpm2ps1_to_mmphr):
    season_cube = iris.load([str(p) for p in inputs]).concatenate_cube()
    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                           num_per_day=num_per_day,
                                                           convert_kgpm2ps1_to_mmphr=convert_kgpm2ps1_to_mmphr,
                                                           calc_method='low_mem')

    iris.save(analysis_cubes, str(outputs[0]))


class CmorphSpaTask(Task):
    def __init__(self, start_year_month, end_year_month, precip_thresh, season):
        datadir = PATHS['datadir'] / 'cmorph_data' / '8km-30min'
        file_tpl = 'cmorph_ppt_{year}{month:02}.asia.N1280.nc'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                file_tpl=file_tpl)

        num_per_day = 48
        daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
        thresh_text = fmt_thresh_text(precip_thresh)
        output_path = (PATHS['datadir'] / 'cmorph_data' /
                       f'cmorph_ppt_{season}.{daterange}.asia_precip.ppt_thresh_{thresh_text}.N1280.nc')

        super().__init__(gen_seasonal_precip_analysis, nc_season, [output_path],
                         func_args=(season, precip_thresh, num_per_day, False))


class UmN1280SpaTask(Task):
    RUNIDS = [
        'ak543',
        'al508',
    ]

    def __init__(self, start_year_month, end_year_month, precip_thresh, season, runid):
        suite = f'u-{runid}'
        split_stream = 'a.p9'
        loc = 'asia'

        datadir = PATHS['datadir'] / suite / 'ap9.pp'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                runid=runid, split_stream=split_stream, loc=loc)

        num_per_day = 24
        daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
        thresh_text = fmt_thresh_text(precip_thresh)
        output_path = (PATHS['datadir'] / suite / 'ap9.pp' /
                       f'{runid}{split_stream}{season}.{daterange}.{loc}_precip.ppt_thresh_{thresh_text}.nc')

        super().__init__(gen_seasonal_precip_analysis, nc_season, [output_path],
                         func_args=(season, precip_thresh, num_per_day, True))


class UmHadgemSpaTask(Task):
    MODELS = {
        'HadGEM3-GC31-HM': {
            'input_dir': PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-HM',
        },
        'HadGEM3-GC31-MM': {
            'input_dir': PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-MM',
        },
        'HadGEM3-GC31-LM': {
            'input_dir': PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/local/HadGEM3-GC31-LM',
        },
    }

    def __init__(self, start_year_month, end_year_month, precip_thresh, season, model):
        input_dir = UmHadgemSpaTask.MODELS[model]['input_dir']
        input_filenames = [input_dir / f'{model}.highresSST-present.r1i1p1f1.{year}.{season}.asia_precip.nc'
                           for year in range(start_year_month[0], end_year_month[0] + 1)]

        num_per_day = 24
        daterange = f'{start_year_month[0]}-{end_year_month[0]}'

        thresh_text = fmt_thresh_text(precip_thresh)
        output_path = (input_dir / f'{model}.highresSST-present.r1i1p1f1.{daterange}.'
                                   f'{season}.asia_precip.ppt_thresh_{thresh_text}.nc')

        super().__init__(gen_seasonal_precip_analysis, input_filenames, [output_path],
                         func_args=(season, precip_thresh, num_per_day, True))


def gen_task_ctrl():
    task_ctrl = TaskControl(enable_file_task_content_checks=True, dotremake_dir='.remake.seasonal_precip_analysis')
    precip_thresh = 0.1
    season = 'jja'

    start_year_month = (1998, 1)
    end_year_month = (2018, 12)
    task_ctrl.add(CmorphSpaTask(start_year_month, end_year_month, precip_thresh, season))

    start_year_month = (2005, 6)
    end_year_month = (2008, 8)
    for runid in UmN1280SpaTask.RUNIDS:
        task_ctrl.add(UmN1280SpaTask(start_year_month, end_year_month, precip_thresh, season, runid))

    for model in UmHadgemSpaTask.MODELS.keys():
        task_ctrl.add(UmHadgemSpaTask(start_year_month, end_year_month, precip_thresh, season, model))

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
