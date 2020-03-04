import sys

from cosmic.util import load_module
from cosmic.processing.extract_asia import HadGEM3_extract_asia_precip


def main(models_settings, model, year, season):
    datadir = models_settings[model]['datadir']
    output_dir = models_settings[model]['output_dir']
    print(f'{model}, {year}, {season}')
    HadGEM3_extract_asia_precip(model, datadir, output_dir, year, season)


if __name__ == '__main__':
    config = load_module(sys.argv[1])
    config_key = sys.argv[2]
    main(config.MODELS, *config.SCRIPT_ARGS[config_key])
