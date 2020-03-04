import sys

import iris
from iris.experimental.equalise_cubes import equalise_attributes

from cosmic.util import load_module
import cosmic.WP2.seasonal_precip_analysis as spa


def main(models_settings, start_year, end_year, model, precip_thresh, season):
    input_dir = models_settings[model]['input_dir']
    input_filenames = sorted([fn for fn in
                              input_dir.glob(f'{model}.highresSST-present.r1i1p1f1.????.{season}.asia_precip.N1280.nc')
                              if start_year <= int(fn.stem.split('.')[3]) <= end_year])

    thresh_text = str(precip_thresh).replace('.', 'p')
    output_filename = input_dir / f'{model}.highresSST-present.r1i1p1f1.{start_year}-{end_year}.' \
                                  f'{season}.asia_precip.N1280.ppt_thresh_{thresh_text}.nc'
    done_filename = (output_filename.parent / (output_filename.name + '.done'))

    if done_filename.exists():
        print(f'Skipping: {done_filename.name} exists')
        return

    season_cubes = iris.load([str(fn) for fn in input_filenames])
    equalise_attributes(season_cubes)
    season_cube = season_cubes.concatenate_cube()
    analysis_cubes = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh)

    iris.save(analysis_cubes, str(output_filename))
    done_filename.touch()


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config.MODELS, config.START_YEAR, config.END_YEAR,
         *config.SCRIPT_ARGS[config_key])
