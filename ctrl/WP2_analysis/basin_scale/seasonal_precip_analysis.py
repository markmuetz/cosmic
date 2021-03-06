import sys
import iris
from iris.experimental.equalise_cubes import equalise_attributes

import cosmic.WP2.seasonal_precip_analysis as spa
from remake import Task, TaskControl, remake_task_control, remake_required

from cosmic.config import PATHS

BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '10:00',
}


def fmt_year_month(year, month):
    return f'{year}{month:02}'


def fmt_thresh_text(precip_thresh):
    return str(precip_thresh).replace('.', 'p')


def fmt_afi_output_filename(dataset, start_year_month, end_year_month, precip_thresh, season, region='asia'):
    daterange = fmt_year_month(*start_year_month) + '-' + fmt_year_month(*end_year_month)
    thresh_text = fmt_thresh_text(precip_thresh)

    output_path = f'{dataset}.{daterange}.{season}.{region}_precip_afi.ppt_thresh_{thresh_text}.nc'
    return output_path


@remake_required(depends_on=[spa.calc_precip_amount_freq_intensity])
def gen_seasonal_precip_analysis(inputs, outputs, season, precip_thresh, num_per_day, convert_kgpm2ps1_to_mmphr):
    season_cubes = iris.load([str(p) for p in inputs])
    # Needed for HadGEM cubes, won't adversely affect others.
    equalise_attributes(season_cubes)
    coord_system = None
    # Needed for CMORPH N1280, for which only one has coord_system set.
    for cube in season_cubes:
        if cube.coord('latitude').coord_system:
            coord_system = cube.coord('latitude').coord_system
            assert cube.coord('longitude').coord_system == coord_system
            break

    if coord_system:
        for cube in season_cubes:
            if cube.coord('latitude').coord_system:
                assert cube.coord('latitude').coord_system == coord_system
                assert cube.coord('longitude').coord_system == coord_system
            else:
                cube.coord('latitude').coord_system = coord_system
                cube.coord('longitude').coord_system = coord_system

    season_cube = season_cubes.concatenate_cube()

    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                           num_per_day=num_per_day,
                                                           convert_kgpm2ps1_to_mmphr=convert_kgpm2ps1_to_mmphr,
                                                           calc_method='low_mem')

    iris.save(analysis_cubes, str(outputs[0]))


class CmorphSpaTask(Task):
    def __init__(self, start_year_month, end_year_month, precip_thresh, season, region):
        datadir = PATHS['datadir'] / 'cmorph_data' / '8km-30min'
        file_tpl = 'cmorph_ppt_{year}{month:02}.{region}.N1280.nc'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                file_tpl=file_tpl, region=region)

        output_filename = fmt_afi_output_filename('cmorph_8km_N1280',
                                                  start_year_month, end_year_month, precip_thresh, season, region)
        self.output_path = PATHS['datadir'] / 'cmorph_data' / '8km-30min' / output_filename

        num_per_day = 48
        super().__init__(gen_seasonal_precip_analysis, nc_season, [self.output_path],
                         func_args=(season, precip_thresh, num_per_day, False))


class UmN1280SpaTask(Task):
    RUNIDS = [
        'ak543',
        'al508',
        'aj399',
        'az035',
        'am754',
    ]

    def __init__(self, start_year_month, end_year_month, precip_thresh, season, region, runid):
        suite = f'u-{runid}'
        split_stream = 'a.p9'

        datadir = PATHS['datadir'] / suite / 'ap9.pp'
        nc_season = spa.gen_nc_precip_filenames(datadir, season, start_year_month, end_year_month,
                                                runid=runid, split_stream=split_stream, loc=region)

        output_filename = fmt_afi_output_filename(runid, start_year_month, end_year_month, precip_thresh, season, region)
        self.output_path = PATHS['datadir'] / suite / 'ap9.pp' / output_filename

        num_per_day = 24
        super().__init__(gen_seasonal_precip_analysis, nc_season, [self.output_path],
                         func_args=(season, precip_thresh, num_per_day, True))


class UmHadgemSpaTask(Task):
    MODELS = [
        'HadGEM3-GC31-HM',
        'HadGEM3-GC31-MM',
        'HadGEM3-GC31-LM',
    ]

    def __init__(self, start_year_month, end_year_month, precip_thresh, season, model):
        season = season.upper()
        input_dir = PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC' / 'local' / model
        input_filenames = [input_dir / f'{model}.highresSST-present.r1i1p1f1.{year}.{season}.asia_precip.nc'
                           for year in range(start_year_month[0], end_year_month[0] + 1)]

        output_filename = fmt_afi_output_filename(f'{model}.highresSST-present.r1i1p1f1',
                                                  start_year_month, end_year_month, precip_thresh, season)
        self.output_path = input_dir / output_filename

        num_per_day = 24
        super().__init__(gen_seasonal_precip_analysis, input_filenames, [self.output_path],
                         func_args=(season, precip_thresh, num_per_day, True))


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    precip_thresh = 0.1
    season = 'jja'
    runids = UmN1280SpaTask.RUNIDS
    # runids = ['aj399']
    # regions = ['asia', 'europe']
    regions = ['asia']

    for region in regions:
        start_year_month = (1998, 1)
        end_year_month = (2018, 12)
        task_ctrl.add(CmorphSpaTask(start_year_month, end_year_month, precip_thresh, season, region))

        for start_year in range(1998, 2016):
            start_year_month = (start_year, 6)
            end_year_month = (start_year + 3, 8)
            task_ctrl.add(CmorphSpaTask(start_year_month, end_year_month, precip_thresh, season, region))

        start_year_month = (2005, 6)
        end_year_month = (2008, 8)
        for runid in runids:
            task_ctrl.add(UmN1280SpaTask(start_year_month, end_year_month, precip_thresh, season, region, runid))

        # Disabled for now.
        # for model in UmHadgemSpaTask.MODELS:
        #     task_ctrl.add(UmHadgemSpaTask(start_year_month, end_year_month, precip_thresh, season, region, model))

    return task_ctrl

    # Individual analysis for each JJA season.
    for year in range(1998, 2019):
        start_year_month = (year, 6)
        end_year_month = (year, 8)
        task_ctrl.add(CmorphSpaTask(start_year_month, end_year_month, precip_thresh, season))

    for year in range(2005, 2009):
        start_year_month = (year, 6)
        end_year_month = (year, 8)
        for runid in runids:
            task_ctrl.add(UmN1280SpaTask(start_year_month, end_year_month, precip_thresh, season, runid))

        for model in UmHadgemSpaTask.MODELS:
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
