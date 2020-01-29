import sys

import iris

from cosmic.util import load_config, filepath_regrid


def main(target_filename, models_settings, model, year, season):
    input_dir = models_settings[model]['input_dir']
    print(f'{model}, {year}, {season}')
    input_filename = input_dir / f'{model}.highresSST-present.r1i1p1f1.{year}.{season}.asia_precip.nc'
    output_filename = input_filename.parent / f'{input_filename.stem}.N1280.nc'
    done_filename = (output_filename.parent / (output_filename.name + '.done'))

    if done_filename.exists():
        print(f'Skipping: {done_filename.name} exists')
        return

    regridded_cube = filepath_regrid(input_filename, target_filename)
    iris.save(regridded_cube, str(output_filename), zlib=True)
    done_filename.touch()


if __name__ == '__main__':
    config = load_config(sys.argv[1])
    config_key = sys.argv[2]
    main(config.TARGET_FILENAME, config.MODELS, *config.SCRIPT_ARGS[config_key])
